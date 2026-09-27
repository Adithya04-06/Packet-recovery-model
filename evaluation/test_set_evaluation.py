import os
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from recovery.inference import InferenceAPI
from evaluation.metrics import MetricsEvaluator
from config.settings import DATA_DIR

def run_test_evaluation(args):
    test_file = args.test_file
    if not os.path.exists(test_file):
        print(f"Error: Test file not found at {test_file}")
        return
        
    os.makedirs(args.out_dir, exist_ok=True)
        
    inference_api = InferenceAPI(model_path=args.model_path)
    evaluator = MetricsEvaluator()
    
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    total_samples = len(samples)
    print(f"Loaded {total_samples} test samples.")
    
    predictions_file = os.path.join(args.out_dir, "predictions.jsonl")
    out_f = open(predictions_file, "w", encoding="utf-8")
    
    exact_matches = 0
    total_word_acc_corr = 0.0
    total_word_acc_rec = 0.0
    total_char_sim_corr = 0.0
    total_char_sim_rec = 0.0
    total_bleu = 0.0
    total_rouge = 0.0
    total_gen_len_chars = 0.0
    
    correct_examples = []
    incorrect_examples = []
    
    print("Running inference on test set...")
    for item in tqdm(samples, desc="Evaluating"):
        original = item['original_text']
        corrupted = item['corrupted_text']
        
        # Recover
        rec_res = inference_api.recover_message(corrupted)
        recovered = rec_res['recovered_text']
        
        # Evaluate
        metrics = evaluator.evaluate_message(original, corrupted, recovered)
        
        if metrics['exact_match_recovered']:
            exact_matches += 1
            if len(correct_examples) < 5:
                correct_examples.append({"original": original, "corrupted": corrupted, "recovered": recovered})
        else:
            if len(incorrect_examples) < 5:
                incorrect_examples.append({"original": original, "corrupted": corrupted, "recovered": recovered, "metrics": metrics})
            
        total_word_acc_corr += metrics['word_accuracy_corrupted']
        total_word_acc_rec += metrics['word_accuracy_recovered']
        total_char_sim_corr += metrics['char_similarity_corrupted']
        total_char_sim_rec += metrics['char_similarity_recovered']
        total_bleu += metrics['bleu_recovered']
        total_rouge += metrics['rougeL_recovered']
        total_gen_len_chars += metrics['generated_length_chars']
        
        # Save
        out_item = {
            "original_text": original,
            "corrupted_text": corrupted,
            "recovered_text": recovered,
            "is_exact_match": metrics['exact_match_recovered'],
            "metrics": metrics
        }
        out_f.write(json.dumps(out_item) + "\n")
        
    out_f.close()
    
    # Save Error Analysis
    error_analysis_file = os.path.join(args.out_dir, "error_analysis.json")
    with open(error_analysis_file, "w", encoding="utf-8") as ea_f:
        json.dump({
            "correct_examples": correct_examples,
            "incorrect_examples": incorrect_examples
        }, ea_f, indent=2)
    
    # Aggregates
    exact_match_acc = exact_matches / total_samples
    avg_word_acc_corr = total_word_acc_corr / total_samples
    avg_word_acc_rec = total_word_acc_rec / total_samples
    avg_char_sim_corr = total_char_sim_corr / total_samples
    avg_char_sim_rec = total_char_sim_rec / total_samples
    avg_bleu = total_bleu / total_samples
    avg_rouge = total_rouge / total_samples
    avg_gen_len = total_gen_len_chars / total_samples
    
    improvement_word = avg_word_acc_rec - avg_word_acc_corr
    improvement_char = avg_char_sim_rec - avg_char_sim_corr
    
    print("\n--- TEST SET EVALUATION REPORT ---")
    print(f"Total test samples:       {total_samples}")
    print(f"Exact matches:            {exact_matches}")
    print(f"Exact Match Accuracy:     {exact_match_acc*100:.2f}%")
    print(f"Average Word Accuracy:    {avg_word_acc_rec*100:.2f}% (Baseline: {avg_word_acc_corr*100:.2f}%)")
    print(f"Average Char Similarity:  {avg_char_sim_rec*100:.2f}% (Baseline: {avg_char_sim_corr*100:.2f}%)")
    print(f"Average BLEU:             {avg_bleu*100:.2f}%")
    print(f"Average ROUGE-L:          {avg_rouge*100:.2f}%")
    print(f"Average Gen Length:       {avg_gen_len:.1f} chars")
    print(f"Improvement (Word Acc):   {improvement_word*100:.2f}%")
    
    # Table and Graph
    results_data = {
        "Metric": ["Word Accuracy", "Char Similarity"],
        "Baseline (Corrupted)": [avg_word_acc_corr*100, avg_char_sim_corr*100],
        "ML Recovered": [avg_word_acc_rec*100, avg_char_sim_rec*100],
        "Improvement": [improvement_word*100, improvement_char*100]
    }
    df = pd.DataFrame(results_data)
    print("\n--- RESULTS TABLE ---")
    print(df.to_string(index=False))
    
    # Plot
    labels = df["Metric"]
    baseline_vals = df["Baseline (Corrupted)"]
    recovered_vals = df["ML Recovered"]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar([p - width/2 for p in x], baseline_vals, width, label='Baseline (No ML)')
    rects2 = ax.bar([p + width/2 for p in x], recovered_vals, width, label='ML Recovered')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Baseline vs ML Text Recovery Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plot_path = os.path.join(args.out_dir, "test_set_evaluation.png")
    plt.savefig(plot_path)
    print(f"\nSaved graph to {plot_path}")
    print(f"Saved predictions to {predictions_file}")
    print(f"Saved error analysis to {error_analysis_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text recovery model on test set.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model directory.")
    parser.add_argument("--test-file", type=str, default=os.path.join(DATA_DIR, "test.jsonl"), help="Path to the test JSONL file.")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save evaluation outputs.")
    
    args = parser.parse_args()
    run_test_evaluation(args)
