import os
import json
import random
import re
import argparse
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from recovery.inference import InferenceAPI
from evaluation.metrics import MetricsEvaluator
from config.settings import DATA_DIR, BEST_MODEL_DIR

def simulate_word_packet_loss(text: str, loss_rate: float, seed: int) -> str:
    random.seed(seed)
    if loss_rate <= 0.0:
        return text
        
    words = text.split()
    if not words: return text
    
    target_drop_count = max(1, int(len(words) * loss_rate))
    
    dropped = 0
    words_copy = words[:]
    
    # Randomly select chunks of words to drop to simulate packet bursts
    # We will try to drop 1 to 3 words at a time
    attempts = 0
    while dropped < target_drop_count and attempts < 100:
        attempts += 1
        span_len = min(random.randint(1, 3), target_drop_count - dropped)
        start_idx = random.randint(0, len(words_copy) - span_len)
        
        # Check if already dropped
        already_dropped = any(w == "<MISSING>" for w in words_copy[start_idx:start_idx+span_len])
        if not already_dropped:
            for i in range(start_idx, start_idx + span_len):
                words_copy[i] = "<MISSING>"
                dropped += 1
                
    corrupted_str = " ".join(words_copy)
    corrupted_str = re.sub(r'(<MISSING>\s*)+', '<MISSING> ', corrupted_str).strip()
    return corrupted_str

def run_loss_robustness_experiment(args):
    results_dir = args.out_dir
    os.makedirs(results_dir, exist_ok=True)
    
    test_file = args.test_file
    
    # Load test sentences
    clean_sentences = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                clean_sentences.append(json.loads(line)['original_text'])
                
    total_samples = len(clean_sentences)
    print(f"Loaded {total_samples} clean sentences for experiment.")
    
    loss_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    
    inference_api = InferenceAPI(model_path=args.model_path)
    evaluator = MetricsEvaluator()
    
    predictions_file = os.path.join(results_dir, "packet_loss_predictions.jsonl")
    out_f = open(predictions_file, "w", encoding="utf-8")
    
    all_metrics = []
    
    for loss_rate in loss_rates:
        print(f"\n--- Testing Packet Loss Rate: {int(loss_rate*100)}% ---")
        
        total_word_acc_corr = 0.0
        total_word_acc_rec = 0.0
        total_char_sim_corr = 0.0
        total_char_sim_rec = 0.0
        total_bleu = 0.0
        total_rouge = 0.0
        total_gen_len = 0.0
        exact_matches = 0
        
        for idx, sentence in enumerate(tqdm(clean_sentences, desc=f"Evaluating {loss_rate*100}%")):
            # Simulate loss
            corrupted = simulate_word_packet_loss(sentence, loss_rate, seed=42+idx+int(loss_rate*100))
            
            # Recover
            rec_res = inference_api.recover_message(corrupted)
            recovered = rec_res['recovered_text']
            
            # Evaluate
            metrics = evaluator.evaluate_message(sentence, corrupted, recovered)
            
            if metrics['exact_match_recovered']:
                exact_matches += 1
                
            total_word_acc_corr += metrics['word_accuracy_corrupted']
            total_word_acc_rec += metrics['word_accuracy_recovered']
            total_char_sim_corr += metrics['char_similarity_corrupted']
            total_char_sim_rec += metrics['char_similarity_recovered']
            total_bleu += metrics['bleu_recovered']
            total_rouge += metrics['rougeL_recovered']
            total_gen_len += metrics['generated_length_chars']
            
            out_f.write(json.dumps({
                "loss_rate": loss_rate,
                "original_text": sentence,
                "corrupted_text": corrupted,
                "recovered_text": recovered,
                "metrics": metrics
            }) + "\n")
            
        n = total_samples
        avg_word_acc_corr = total_word_acc_corr / n
        avg_word_acc_rec = total_word_acc_rec / n
        avg_char_sim_corr = total_char_sim_corr / n
        avg_char_sim_rec = total_char_sim_rec / n
        avg_bleu = total_bleu / n
        avg_rouge = total_rouge / n
        avg_gen_len = total_gen_len / n
        exact_match_acc = exact_matches / n
        
        ml_improvement = avg_word_acc_rec - avg_word_acc_corr
        
        all_metrics.append({
            "Loss Rate": f"{int(loss_rate*100)}%",
            "Float Loss Rate": loss_rate,
            "Baseline Word Accuracy": avg_word_acc_corr * 100,
            "ML Word Accuracy": avg_word_acc_rec * 100,
            "Baseline Char Similarity": avg_char_sim_corr * 100,
            "ML Char Similarity": avg_char_sim_rec * 100,
            "Exact Match": exact_match_acc * 100,
            "BLEU": avg_bleu * 100,
            "ROUGE-L": avg_rouge * 100,
            "Avg Gen Len": avg_gen_len,
            "ML Improvement": ml_improvement * 100
        })
        
    out_f.close()
    
    # Save JSON summary
    summary_file = os.path.join(results_dir, "packet_loss_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
        
    # Print and save CSV
    csv_file = os.path.join(results_dir, "packet_loss_results.csv")
    keys = all_metrics[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_metrics)
        
    print("\n--- FINAL ROBUSTNESS RESULTS ---")
    headers = ["Loss Rate", "Baseline Word Accuracy", "ML Word Accuracy", "Baseline Char Similarity", "ML Char Similarity", "Exact Match", "BLEU", "ROUGE-L", "Avg Gen Len", "ML Improvement"]
    header_str = " | ".join([h.ljust(20) for h in headers])
    print(header_str)
    print("-" * len(header_str))
    for m in all_metrics:
        row = []
        for h in headers:
            val = m[h]
            if isinstance(val, float):
                if h == "Avg Gen Len":
                    row.append(f"{val:.1f}".ljust(20))
                else:
                    row.append(f"{val:.2f}%".ljust(20))
            else:
                row.append(str(val).ljust(20))
        print(" | ".join(row))
    
    # Generate Graphs
    x_rates = [m["Float Loss Rate"]*100 for m in all_metrics]
    
    # Graph 1: Word Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["Baseline Word Accuracy"] for m in all_metrics], 'o--', label="Baseline")
    plt.plot(x_rates, [m["ML Word Accuracy"] for m in all_metrics], 's-', label="ML Recovery")
    plt.title("Packet Loss vs Word Accuracy")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Word Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "packet_loss_vs_word_accuracy.png"))
    plt.close()
    
    # Graph 2: Char Similarity
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["Baseline Char Similarity"] for m in all_metrics], 'o--', label="Baseline")
    plt.plot(x_rates, [m["ML Char Similarity"] for m in all_metrics], 's-', label="ML Recovery")
    plt.title("Packet Loss vs Character Similarity")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Character Similarity (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "packet_loss_vs_char_similarity.png"))
    plt.close()
    
    # Graph 3: Exact Match
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["Exact Match"] for m in all_metrics], '^-', color='purple')
    plt.title("Packet Loss vs Exact Match Accuracy")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Exact Match Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "packet_loss_vs_exact_match.png"))
    plt.close()
    
    print(f"\nSaved all results and graphs to {results_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate packet loss robustness.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model directory.")
    parser.add_argument("--test-file", type=str, default=os.path.join(DATA_DIR, "test.jsonl"), help="Path to the test JSONL file.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save evaluation outputs.")
    
    args = parser.parse_args()
    run_loss_robustness_experiment(args)
