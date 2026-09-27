import os
from recovery.model import TextRecoveryModel
from config.settings import BEST_MODEL_DIR, MODEL_NAME

class InferenceAPI:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = BEST_MODEL_DIR
            
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise ValueError(f"Error: Trained model not found at {model_path}. Expected trained model directory: {BEST_MODEL_DIR}")
            
        print(f"Loading trained model from {os.path.abspath(model_path)}...")
        self.model = TextRecoveryModel(model_name=model_path)

    def recover_message(self, corrupted_text: str, language="en"):
        """
        Inference API to reconstruct corrupted text.
        """
        # If there's no missing token, nothing to recover
        if "<MISSING>" not in corrupted_text:
            return {
                "input": corrupted_text,
                "recovered_text": corrupted_text,
                "confidence": 1.0,
                "language": language
            }
            
        results = self.model.predict(corrupted_text, num_return_sequences=3)
        top_result = results[0]
        
        return {
            "input": corrupted_text,
            "recovered_text": top_result["text"],
            "confidence": top_result["confidence"],
            "language": language,
            "candidates": results
        }

if __name__ == "__main__":
    api = InferenceAPI()
    res = api.recover_message("I <MISSING> pen")
    print(res)
