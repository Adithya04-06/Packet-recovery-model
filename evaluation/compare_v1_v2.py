import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
from recovery.inference import InferenceAPI
from evaluation.metrics import MetricsEvaluator
from config.settings import BASE_DIR
from evaluation.packet_loss_experiment import simulate_word_packet_loss

def compare_v1_v2():
    v1_model_dir = os.path.join(BASE_DIR, "models", "v1", "best_model")
    v2_model_dir = os.path.join(BASE_DIR, "models", "v2", "gpu_model")
    v2_data_dir = os.path.join(BASE_DIR, "data", "v2")
    
    results_dir = os.path.join(BASE_DIR, "evaluation", "results", "v2_comparison")
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading V1 Model...")
    api_v1 = InferenceAPI(model_path=v1_model_dir)
    print("Loading V2 Model...")
    api_v2 = InferenceAPI(model_path=v2_model_dir)
    
    evaluator = MetricsEvaluator()
    
    # Load V2 test set
    test_file = os.path.join(v2_data_dir, "test.jsonl")
    sentences = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sentences.append(json.loads(line)['original_text'])
                
    loss_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    
    intermediate_file = os.path.join(results_dir, "intermediate_metrics.json")
    all_metrics = []
    completed_loss_rates = set()
    
    if os.path.exists(intermediate_file):
        print(f"Found existing intermediate results at {intermediate_file}. Resuming...")
        with open(intermediate_file, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
        for m in all_metrics:
            completed_loss_rates.add(m["Float Loss Rate"])
            
    total_samples = len(loss_rates) * len(sentences)
    total_evaluations = total_samples * 2 # V1 and V2
    
    # Calculate how many have already been completed for overall progress
    completed_samples_so_far = len(completed_loss_rates) * len(sentences)
    overall_evals = completed_samples_so_far * 2
    
    for loss_rate in loss_rates:
        if loss_rate in completed_loss_rates:
            print(f"\n--- Skipping Completed Packet Loss Rate: {int(loss_rate*100)}% ---")
            continue
            
        print(f"\n--- Testing Packet Loss Rate: {int(loss_rate*100)}% ---")
        
        base_word = 0.0; base_char = 0.0; base_exact = 0; base_bleu = 0.0
        v1_word = 0.0; v1_char = 0.0; v1_exact = 0; v1_bleu = 0.0
        v2_word = 0.0; v2_char = 0.0; v2_exact = 0; v2_bleu = 0.0
        
        n = len(sentences)
        for idx, sentence in enumerate(sentences):
            # Print progress
            if idx % 5 == 0 or idx == n - 1:
                current_overall = overall_evals + (idx * 2)
                sys.stdout.write(f"\rLoss rate: {int(loss_rate*100)}% | Sample: {idx+1}/{n} | Overall progress: {current_overall}/{total_evaluations}")
                sys.stdout.flush()
                
            corrupted = simulate_word_packet_loss(sentence, loss_rate, seed=42+idx+int(loss_rate*100))
            
            # V1 Inference
            rec_v1 = api_v1.recover_message(corrupted)['recovered_text']
            met_v1 = evaluator.evaluate_message(sentence, corrupted, rec_v1)
            
            base_word += met_v1['word_accuracy_corrupted']
            base_char += met_v1['char_similarity_corrupted']
            if met_v1['exact_match_corrupted']: base_exact += 1
            met_base = evaluator.evaluate_message(sentence, corrupted, corrupted)
            base_bleu += met_base['bleu_recovered']
            
            v1_word += met_v1['word_accuracy_recovered']
            v1_char += met_v1['char_similarity_recovered']
            v1_bleu += met_v1['bleu_recovered']
            if met_v1['exact_match_recovered']: v1_exact += 1
            
            # V2 Inference
            rec_v2 = api_v2.recover_message(corrupted)['recovered_text']
            met_v2 = evaluator.evaluate_message(sentence, corrupted, rec_v2)
            v2_word += met_v2['word_accuracy_recovered']
            v2_char += met_v2['char_similarity_recovered']
            v2_bleu += met_v2['bleu_recovered']
            if met_v2['exact_match_recovered']: v2_exact += 1
            
        print() # New line after the progress bar finishes
        overall_evals += n * 2 # Add the completed ones for the next loss rate
        
        # Aggregation
        all_metrics.append({
            "Loss Rate": f"{int(loss_rate*100)}%",
            "Float Loss Rate": loss_rate,
            "Baseline Word Accuracy": (base_word/n)*100,
            "V1 Word Accuracy": (v1_word/n)*100,
            "V2 Word Accuracy": (v2_word/n)*100,
            "V1 vs Baseline Imp": ((v1_word/n)*100) - ((base_word/n)*100),
            "V2 vs Baseline Imp": ((v2_word/n)*100) - ((base_word/n)*100),
            "V1 vs V2 Imp": ((v2_word/n)*100) - ((v1_word/n)*100),
            "Baseline Char Similarity": (base_char/n)*100,
            "V1 Char Similarity": (v1_char/n)*100,
            "V2 Char Similarity": (v2_char/n)*100,
            "Baseline Exact Match": (base_exact/n)*100,
            "V1 Exact Match": (v1_exact/n)*100,
            "V2 Exact Match": (v2_exact/n)*100,
            "Baseline BLEU": (base_bleu/n)*100,
            "V1 BLEU": (v1_bleu/n)*100,
            "V2 BLEU": (v2_bleu/n)*100
        })
        
        # Save intermediate results
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Saved intermediate results for {int(loss_rate*100)}% loss rate.")
        
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("Float Loss Rate")
    all_metrics = df.to_dict('records')
    
    display_df = df.drop(columns=['Float Loss Rate'])
    
    for col in display_df.columns:
        if col != 'Loss Rate':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
            
    print("\n--- V1 VS V2 COMPARISON RESULTS ---")
    print(display_df.to_string(index=False))
    
    x_rates = [m["Float Loss Rate"]*100 for m in all_metrics]
    
    # Graphs
    # Word Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["V1 Word Accuracy"] for m in all_metrics], 'o--', label="V1 Model (Literary)")
    plt.plot(x_rates, [m["V2 Word Accuracy"] for m in all_metrics], 's-', label="V2 Model (Conversational)")
    plt.title("V1 vs V2 Word Accuracy")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Word Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "v1_vs_v2_word_accuracy.png"))
    plt.close()
    
    # Char Similarity
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["V1 Char Similarity"] for m in all_metrics], 'o--', label="V1 Model (Literary)")
    plt.plot(x_rates, [m["V2 Char Similarity"] for m in all_metrics], 's-', label="V2 Model (Conversational)")
    plt.title("V1 vs V2 Character Similarity")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Character Similarity (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "v1_vs_v2_char_similarity.png"))
    plt.close()
    
    # Exact Match
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["V1 Exact Match"] for m in all_metrics], 'o--', label="V1 Model")
    plt.plot(x_rates, [m["V2 Exact Match"] for m in all_metrics], 's-', label="V2 Model")
    plt.title("V1 vs V2 Exact Match Accuracy")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Exact Match Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "v1_vs_v2_exact_match.png"))
    plt.close()
    
    # Save to final JSON
    with open(os.path.join(results_dir, "v1_vs_v2_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
        
    # Save to CSV
    df.to_csv(os.path.join(results_dir, "v1_vs_v2_metrics.csv"), index=False)
        
    # Generate Markdown Report
    with open(os.path.join(results_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# V1 vs V2 Comparison Report\n\n")
        f.write("## Metrics Summary\n")
        
        # Convert DataFrame to markdown table manually
        f.write("| " + " | ".join(display_df.columns) + " |\n")
        f.write("|" + "|".join(["---" for _ in display_df.columns]) + "|\n")
        for _, row in display_df.iterrows():
            f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
            
        f.write("\n\n## Visualizations\n")
        f.write("![Word Accuracy](v1_vs_v2_word_accuracy.png)\n\n")
        f.write("![Character Similarity](v1_vs_v2_char_similarity.png)\n\n")
        f.write("![Exact Match](v1_vs_v2_exact_match.png)\n\n")
        
    print(f"\nSaved complete comparison results to {results_dir}/")

if __name__ == "__main__":
    compare_v1_v2()

