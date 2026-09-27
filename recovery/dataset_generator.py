import os
import json
import random
import urllib.request
import re
from config.settings import DATA_DIR

class DatasetGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(self.seed)

    def fetch_base_corpus(self, num_samples=10000):
        print("Fetching dataset without pyarrow...")
        # Download a public domain book (e.g. Alice's Adventures in Wonderland)
        url = "https://www.gutenberg.org/files/11/11-0.txt"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response:
                text = response.read().decode('utf-8')
        except Exception as e:
            print("Failed to download text:", e)
            # Fallback to hardcoded synthetics if no internet
            text = "The quick brown fox jumps over the lazy dog. " * 1000
            
        # Basic sentence splitting
        raw_sentences = re.split(r'(?<=[.!?]) +', text)
        sentences = []
        for sentence in raw_sentences:
            sentence = sentence.replace('\r', ' ').replace('\n', ' ').strip()
            # Filter sentences
            words = sentence.split()
            if 5 <= len(words) <= 30:
                sentences.append(sentence)
                if len(sentences) >= num_samples:
                    break
        
        # If we didn't get enough, repeat the sentences (it's just a prototype test)
        while len(sentences) < num_samples:
            sentences.extend(sentences[:num_samples - len(sentences)])
            
        print(f"Fetched {len(sentences)} base sentences.")
        return sentences

    def corrupt_sentence(self, sentence: str) -> dict:
        words = sentence.split()
        if len(words) < 3:
            return None
            
        # Define corruption patterns
        # 1. single word
        # 2. multiple consecutive
        # 3. random spans
        
        corruption_type = random.choices(
            ['single', 'consecutive', 'random_spans'],
            weights=[0.4, 0.4, 0.2]
        )[0]
        
        target_text = sentence
        
        if corruption_type == 'single':
            idx = random.randint(0, len(words) - 1)
            words[idx] = "<MISSING>"
        elif corruption_type == 'consecutive':
            span_len = random.randint(2, min(4, len(words)-1))
            start_idx = random.randint(0, len(words) - span_len)
            for i in range(start_idx, start_idx + span_len):
                words[i] = "<MISSING>"
        else: # random spans
            num_spans = random.randint(2, 3)
            for _ in range(num_spans):
                idx = random.randint(0, len(words) - 1)
                words[idx] = "<MISSING>"
                
        # Clean up multiple <MISSING> tokens into one
        corrupted_str = " ".join(words)
        import re
        corrupted_str = re.sub(r'(<MISSING>\s*)+', '<MISSING> ', corrupted_str).strip()
        
        if corrupted_str == target_text or "<MISSING>" not in corrupted_str:
            return None
            
        return {
            "original_text": target_text,
            "corrupted_text": corrupted_str,
            "language": "en",
            "corruption_type": corruption_type
        }

    def generate_splits(self, num_train=5000, num_val=500, num_test=500):
        total_needed = num_train + num_val + num_test
        sentences = self.fetch_base_corpus(total_needed * 2)
        
        random.shuffle(sentences)
        
        dataset = []
        for sent in sentences:
            sample = self.corrupt_sentence(sent)
            if sample:
                dataset.append(sample)
            if len(dataset) >= total_needed:
                break
                
        train_data = dataset[:num_train]
        val_data = dataset[num_train:num_train+num_val]
        test_data = dataset[num_train+num_val:num_train+num_val+num_test]
        
        self._save_jsonl(train_data, os.path.join(DATA_DIR, "train.jsonl"))
        self._save_jsonl(val_data, os.path.join(DATA_DIR, "validation.jsonl"))
        self._save_jsonl(test_data, os.path.join(DATA_DIR, "test.jsonl"))
        
        print(f"Generated {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.")

    def _save_jsonl(self, data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_splits(num_train=5000, num_val=500, num_test=500)
