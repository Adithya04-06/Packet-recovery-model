from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class MetricsEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def exact_match(self, pred: str, target: str) -> bool:
        return pred.strip().lower() == target.strip().lower()

    def word_accuracy(self, pred: str, target: str) -> float:
        pred_words = pred.strip().lower().split()
        target_words = target.strip().lower().split()
        
        if not target_words:
            return 1.0 if not pred_words else 0.0
            
        matches = sum(1 for w1, w2 in zip(pred_words, target_words) if w1 == w2)
        return matches / max(len(target_words), len(pred_words))

    def character_similarity(self, pred: str, target: str) -> float:
        import difflib
        sm = difflib.SequenceMatcher(None, pred, target)
        return sm.ratio()

    def evaluate_message(self, original: str, corrupted: str, recovered: str):
        em_corrupted = self.exact_match(corrupted, original)
        em_recovered = self.exact_match(recovered, original)
        
        word_acc_corr = self.word_accuracy(corrupted, original)
        word_acc_rec = self.word_accuracy(recovered, original)
        
        char_sim_corr = self.character_similarity(corrupted, original)
        char_sim_rec = self.character_similarity(recovered, original)
        
        # Calculate BLEU using NLTK
        try:
            ref = [original.lower().split()]
            hyp = recovered.lower().split()
            cc = SmoothingFunction()
            bleu_rec = sentence_bleu(ref, hyp, smoothing_function=cc.method1)
        except Exception:
            bleu_rec = 0.0
            
        # Calculate ROUGE-L
        rouge_scores = self.rouge_scorer.score(original, recovered)
        rouge_l = rouge_scores['rougeL'].fmeasure
            
        return {
            "exact_match_corrupted": em_corrupted,
            "exact_match_recovered": em_recovered,
            "word_accuracy_corrupted": word_acc_corr,
            "word_accuracy_recovered": word_acc_rec,
            "char_similarity_corrupted": char_sim_corr,
            "char_similarity_recovered": char_sim_rec,
            "bleu_recovered": bleu_rec,
            "rougeL_recovered": rouge_l,
            "improvement": word_acc_rec - word_acc_corr,
            "generated_length_chars": len(recovered),
            "generated_length_words": len(recovered.split())
        }

if __name__ == "__main__":
    ev = MetricsEvaluator()
    res = ev.evaluate_message("I have a pen", "I <MISSING> pen", "I have a pen")
    print(res)
