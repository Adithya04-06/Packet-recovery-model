import os
import json
import random
import re
from config.settings import DATA_DIR

class DatasetGeneratorV2:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(self.seed)
        
    def generate_synthetic_sentences(self, num_samples=24000):
        print("Generating conversational iTantra dataset...")
        
        locations = ["railway station", "hospital", "airport", "bus stand", "main gate", "highway", "school", "office", "hotel", "market", "clinic", "pharmacy", "police station", "subway"]
        cities = ["New York", "London", "Mumbai", "Delhi", "Tokyo", "Paris", "Berlin", "Dubai", "Singapore", "Sydney", "Toronto", "Chicago", "Los Angeles", "Chennai", "Bangalore"]
        transports = ["bus", "train", "flight", "ferry", "cab", "taxi", "shuttle", "metro"]
        weathers = ["sunny", "raining heavily", "storming", "snowing", "clear", "cloudy", "foggy", "windy", "very cold", "extremely hot"]
        emergencies = ["Send an ambulance", "We need police assistance", "Fire reported", "Accident happened", "Please send help", "Need a doctor immediately"]
        directions = ["Turn left at", "Go straight past", "Take a right after", "It is located near", "Head north towards", "Keep going until you see"]
        greetings = ["Hello", "Hi there", "Good morning", "Good evening", "Hey"]
        questions = ["Where are you?", "What time are we meeting?", "Did you get the package?", "Are you okay?", "When does it arrive?", "How much does it cost?", "Can you help me?"]
        answers = ["I am on my way.", "Yes, I got it.", "No, not yet.", "It will take 10 minutes.", "Everything is fine here.", "I don't know.", "Maybe tomorrow."]
        healthcare = ["The patient is stable.", "Blood pressure is normal.", "Take this medicine twice a day.", "We need more bandages.", "Doctor is in surgery."]
        education = ["The class is cancelled today.", "Did you finish the assignment?", "The exam starts at 9 AM.", "Bring your textbooks tomorrow."]
        
        templates = [
            "I have reached the {location}.",
            "{greeting}, I am waiting at the {location}.",
            "{emergency} at the {location}!",
            "The weather in {city} is {weather}.",
            "What time does the {transport} leave for {city}?",
            "{direction} the {location}.",
            "My {transport} to {city} is delayed.",
            "{question}",
            "{answer}",
            "{health}",
            "{edu}",
            "Can you book a {transport} to the {location}?",
            "The {location} in {city} is closed due to {weather}.",
            "Please deliver the items to the {location} in {city}."
        ]
        
        sentences = set()
        
        # Keep generating until we hit the target
        while len(sentences) < num_samples:
            template = random.choice(templates)
            
            sentence = template.format(
                location=random.choice(locations),
                city=random.choice(cities),
                transport=random.choice(transports),
                weather=random.choice(weathers),
                emergency=random.choice(emergencies),
                direction=random.choice(directions),
                greeting=random.choice(greetings),
                question=random.choice(questions),
                answer=random.choice(answers),
                health=random.choice(healthcare),
                edu=random.choice(education)
            )
            
            # Optionally combine a greeting or an answer to make longer sentences
            if random.random() < 0.2:
                sentence = f"{random.choice(greetings)}, {sentence.lower()}"
            if random.random() < 0.1:
                sentence = f"{sentence} {random.choice(answers)}"
                
            sentences.add(sentence)
            
        sentences_list = list(sentences)
        random.shuffle(sentences_list)
        return sentences_list

    def corrupt_sentence(self, sentence: str) -> dict:
        words = sentence.split()
        if len(words) < 3:
            return None
            
        # Realistic corruption patterns
        # single missing word
        # 2 consecutive
        # 3 consecutive
        # short phrase
        # beginning deletion
        # middle deletion
        # ending deletion
        # multiple spans
        
        corruption_type = random.choices(
            ['single', '2_consecutive', '3_consecutive', 'beginning', 'end', 'multiple_spans'],
            weights=[0.3, 0.2, 0.1, 0.1, 0.1, 0.2]
        )[0]
        
        words_copy = words[:]
        
        if corruption_type == 'single':
            idx = random.randint(0, len(words) - 1)
            words_copy[idx] = "<MISSING>"
            
        elif corruption_type == '2_consecutive':
            if len(words) >= 2:
                idx = random.randint(0, len(words) - 2)
                words_copy[idx] = "<MISSING>"
                words_copy[idx+1] = "<MISSING>"
                
        elif corruption_type == '3_consecutive':
            if len(words) >= 3:
                idx = random.randint(0, len(words) - 3)
                for i in range(idx, idx+3):
                    words_copy[i] = "<MISSING>"
                    
        elif corruption_type == 'beginning':
            span = random.randint(1, min(3, len(words)-1))
            for i in range(span):
                words_copy[i] = "<MISSING>"
                
        elif corruption_type == 'end':
            span = random.randint(1, min(3, len(words)-1))
            for i in range(len(words)-span, len(words)):
                words_copy[i] = "<MISSING>"
                
        elif corruption_type == 'multiple_spans':
            num_spans = 2
            for _ in range(num_spans):
                span = random.randint(1, 2)
                start = random.randint(0, max(0, len(words) - span))
                for i in range(start, min(len(words), start+span)):
                    words_copy[i] = "<MISSING>"
                    
        corrupted_str = " ".join(words_copy)
        corrupted_str = re.sub(r'(<MISSING>\s*)+', '<MISSING> ', corrupted_str).strip()
        
        if corrupted_str == sentence or "<MISSING>" not in corrupted_str:
            # Fallback single
            idx = random.randint(0, len(words) - 1)
            words_copy = words[:]
            words_copy[idx] = "<MISSING>"
            corrupted_str = " ".join(words_copy)
            
        return {
            "original_text": sentence,
            "corrupted_text": corrupted_str,
            "language": "en",
            "corruption_type": corruption_type
        }

    def generate_splits(self, num_train=20000, num_val=2000, num_test=2000):
        total_needed = num_train + num_val + num_test
        sentences = self.generate_synthetic_sentences(total_needed)
        
        dataset = []
        for sent in sentences:
            sample = self.corrupt_sentence(sent)
            if sample:
                dataset.append(sample)
                
        # Ensure we have exactly the numbers needed
        train_data = dataset[:num_train]
        val_data = dataset[num_train:num_train+num_val]
        test_data = dataset[num_train+num_val:num_train+num_val+num_test]
        
        self._save_jsonl(train_data, os.path.join(DATA_DIR, "train.jsonl"))
        self._save_jsonl(val_data, os.path.join(DATA_DIR, "validation.jsonl"))
        self._save_jsonl(test_data, os.path.join(DATA_DIR, "test.jsonl"))
        
        print(f"Generated {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples in {DATA_DIR}.")

    def _save_jsonl(self, data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    generator = DatasetGeneratorV2()
    generator.generate_splits(num_train=20000, num_val=2000, num_test=2000)
