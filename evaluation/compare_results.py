import os
import json
import csv
import matplotlib.pyplot as plt

def run_comparison():
    v1_file = os.path.join("evaluation", "results", "packet_loss_results.json")
    v2_file = os.path.join("evaluation", "v2", "packet_loss_results.json")
    
    if not os.path.exists(v1_file) or not os.path.exists(v2_file):
        print(f"Missing one of the results files:\nV1: {v1_file}\nV2: {v2_file}")
        return
        
    with open(v1_file, "r") as f:
        v1_data = json.load(f)
        
    with open(v2_file, "r") as f:
        v2_data = json.load(f)
        
    v1_dict = {m["Float Loss Rate"]: m for m in v1_data}
    v2_dict = {m["Float Loss Rate"]: m for m in v2_data}
    
    all_metrics = []
    for rate in sorted(v2_dict.keys()):
        m1 = v1_dict.get(rate, {})
        m2 = v2_dict[rate]
        
        all_metrics.append({
            "Loss Rate": f"{int(rate*100)}%",
            "Float Loss Rate": rate,
            "Baseline Word Acc": m2.get("Baseline Word Accuracy", 0),
            "V1 Word Acc": m1.get("ML Word Accuracy", 0),
            "V2 Word Acc": m2.get("ML Word Accuracy", 0),
            "V1 Char Sim": m1.get("ML Char Similarity", 0),
            "V2 Char Sim": m2.get("ML Char Similarity", 0),
            "V1 Exact Match": m1.get("Exact Match", 0),
            "V2 Exact Match": m2.get("Exact Match", 0),
            "V1 BLEU": m1.get("BLEU", 0),
            "V2 BLEU": m2.get("BLEU", 0)
        })
        
    out_dir = os.path.join("evaluation", "v2", "comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_file = os.path.join(out_dir, "v1_vs_v2_comparison.csv")
    keys = all_metrics[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_metrics)
    
    print("\n--- V1 VS V2 COMPARISON RESULTS ---")
    headers = ["Loss Rate", "Baseline Word Acc", "V1 Word Acc", "V2 Word Acc", "V1 Exact Match", "V2 Exact Match", "V1 BLEU", "V2 BLEU"]
    header_str = " | ".join([h.ljust(18) for h in headers])
    print(header_str)
    print("-" * len(header_str))
    for m in all_metrics:
        row = []
        for h in headers:
            val = m[h]
            if isinstance(val, float):
                row.append(f"{val:.2f}%".ljust(18))
            else:
                row.append(str(val).ljust(18))
        print(" | ".join(row))
    
    x_rates = [m["Float Loss Rate"]*100 for m in all_metrics]
    
    # Graphs
    # Word Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["V1 Word Acc"] for m in all_metrics], 'o--', label="V1 Model")
    plt.plot(x_rates, [m["V2 Word Acc"] for m in all_metrics], 's-', label="V2 Model")
    plt.plot(x_rates, [m["Baseline Word Acc"] for m in all_metrics], 'x:', label="Baseline", alpha=0.6)
    plt.title("V1 vs V2 Word Accuracy")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Word Accuracy (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "v1_vs_v2_word_accuracy.png"))
    plt.close()
    
    # Exact Match
    plt.figure(figsize=(10, 6))
    plt.plot(x_rates, [m["V1 Exact Match"] for m in all_metrics], 'o--', label="V1 Model")
    plt.plot(x_rates, [m["V2 Exact Match"] for m in all_metrics], 's-', label="V2 Model")
    plt.title("V1 vs V2 Exact Match")
    plt.xlabel("Packet Loss Rate (%)")
    plt.ylabel("Exact Match (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "v1_vs_v2_exact_match.png"))
    plt.close()

    print(f"\nSaved comparison files to {out_dir}/")

if __name__ == "__main__":
    run_comparison()
