import os
import matplotlib.pyplot as plt
import pandas as pd
from packet.pipeline import CommunicationPipeline
from evaluation.metrics import MetricsEvaluator
from recovery.inference import InferenceAPI
from config.settings import AES_KEY

class ExperimentRunner:
    def __init__(self):
        self.pipeline = CommunicationPipeline(AES_KEY)
        self.evaluator = MetricsEvaluator()
        self.inference = InferenceAPI()
        
    def run_experiment(self, sentences, loss_rates=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]):
        results = []
        
        for loss_rate in loss_rates:
            print(f"Running experiment for loss rate: {loss_rate*100}%")
            
            total_corrupted_acc = 0
            total_recovered_acc = 0
            total_improvement = 0
            total_exact_match = 0
            
            for sent in sentences:
                # 1. Transmit through simulated channel
                pipe_res = self.pipeline.transmit(sent, drop_probability=loss_rate)
                corrupted = pipe_res['corrupted_text']
                
                # 2. Recover using ML
                rec_res = self.inference.recover_message(corrupted)
                recovered = rec_res['recovered_text']
                
                # 3. Evaluate
                metrics = self.evaluator.evaluate_message(sent, corrupted, recovered)
                
                total_corrupted_acc += metrics['word_accuracy_corrupted']
                total_recovered_acc += metrics['word_accuracy_recovered']
                total_improvement += metrics['improvement']
                if metrics['exact_match_recovered']:
                    total_exact_match += 1
                    
            n = len(sentences)
            results.append({
                "Packet Loss": f"{int(loss_rate*100)}%",
                "Loss Rate Float": loss_rate,
                "Corrupted Accuracy": f"{(total_corrupted_acc/n)*100:.2f}%",
                "Recovered Accuracy": f"{(total_recovered_acc/n)*100:.2f}%",
                "Improvement": f"{(total_improvement/n)*100:.2f}%",
                "Exact Match": f"{(total_exact_match/n)*100:.2f}%",
                "_corr_acc": total_corrupted_acc/n,
                "_rec_acc": total_recovered_acc/n
            })
            
        self.generate_report(results)
        
    def generate_report(self, results):
        df = pd.DataFrame(results)
        display_df = df.drop(columns=['Loss Rate Float', '_corr_acc', '_rec_acc'])
        print("\n--- EXPERIMENT RESULTS ---")
        print(display_df.to_string(index=False))
        
        # Plot
        loss_rates = [r['Loss Rate Float']*100 for r in results]
        corr_accs = [r['_corr_acc']*100 for r in results]
        rec_accs = [r['_rec_acc']*100 for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_rates, corr_accs, marker='o', linestyle='--', label='Before ML Recovery (Corrupted)')
        plt.plot(loss_rates, rec_accs, marker='s', linestyle='-', label='After ML Recovery')
        
        plt.title('Packet Loss vs Text Recovery Accuracy')
        plt.xlabel('Packet Loss Rate (%)')
        plt.ylabel('Word-level Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plot_path = "evaluation_graph.png"
        plt.savefig(plot_path)
        print(f"\nSaved evaluation graph to {plot_path}")

if __name__ == "__main__":
    from recovery.dataset_generator import DatasetGenerator
    gen = DatasetGenerator()
    sents = gen.fetch_base_corpus(num_samples=50) # Use 50 samples for quick experiment
    runner = ExperimentRunner()
    runner.run_experiment(sents)
