import json

count = 0
print("| Original | Corrupted | Recovered |")
print("|---|---|---|")

with open('predictions.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        orig = data['original_text']
        corr = data['corrupted_text']
        rec = data['recovered_text']
        print(f"| {orig} | {corr} | {rec} |")
        count += 1
        if count >= 10:
            break
