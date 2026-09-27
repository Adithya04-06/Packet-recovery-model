import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config.settings import MODEL_NAME, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH

class TextRecoveryModel:
    def __init__(self, model_name=MODEL_NAME, device=None):
        """
        Initializes the sequence-to-sequence model for text recovery.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def prepare_data(self, corrupted_text, target_text=None):
        """
        Tokenizes the input. If T5 is used, <MISSING> is mapped to <extra_id_0>.
        """
        # Internal mapping
        input_text = corrupted_text.replace("<MISSING>", "<extra_id_0>")
        
        inputs = self.tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = None
        if target_text is not None:
            labels = self.tokenizer(
                target_text,
                max_length=MAX_TARGET_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # HF expects pad tokens to be -100 for calculating loss
            labels_ids = labels.input_ids
            labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
            labels = labels_ids
            
        return inputs, labels

    def predict(self, corrupted_text, num_return_sequences=1):
        """
        Generates reconstruction.
        """
        inputs, _ = self.prepare_data(corrupted_text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=max(5, num_return_sequences),
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )
            
        decoded = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        # Calculate a pseudo-confidence score using sequences scores
        # Sequence scores are the sum of log probabilities.
        # This is a heuristic.
        results = []
        if 'sequences_scores' in outputs:
            scores = torch.exp(outputs.sequences_scores).cpu().numpy()
            for seq, score in zip(decoded, scores):
                results.append({"text": seq.strip(), "confidence": float(score)})
        else:
            for seq in decoded:
                results.append({"text": seq.strip(), "confidence": 1.0})
                
        return results

if __name__ == "__main__":
    # Test initialization
    m = TextRecoveryModel()
    res = m.predict("I <MISSING> pen")
    print(res)
