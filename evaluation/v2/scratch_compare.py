import json
import sys

def main():
    try:
        with open("evaluation/results/packet_loss_results.json", "r") as f:
            v1_data = {m["Loss Rate"]: m for m in json.load(f)}
            
        with open("evaluation/v2/packet_loss_results.json", "r") as f:
            v2_data = {m["Loss Rate"]: m for m in json.load(f)}
            
        rates = ["0%", "5%", "10%", "15%", "20%", "25%", "30%", "40%"]
        
        for r in rates:
            m1 = v1_data.get(r, {})
            m2 = v2_data.get(r, {})
            print(f"Loss Rate: {r}")
            print(f"V1 ML Word Accuracy: {m1.get('ML Word Accuracy', 'N/A')}")
            print(f"V2 ML Word Accuracy: {m2.get('ML Word Accuracy', 'N/A')}")
            print(f"V1 Exact Match: {m1.get('Exact Match', 'N/A')}")
            print(f"V2 Exact Match: {m2.get('Exact Match', 'N/A')}")
            print(f"V1 BLEU: {m1.get('BLEU', 'N/A')}")
            print(f"V2 BLEU: {m2.get('BLEU', 'N/A')}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
