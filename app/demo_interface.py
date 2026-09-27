import gradio as gr
from packet.pipeline import CommunicationPipeline
from evaluation.metrics import MetricsEvaluator
from recovery.inference import InferenceAPI
from config.settings import AES_KEY
import pandas as pd
import json

class DemoApp:
    def __init__(self):
        self.pipeline = CommunicationPipeline(AES_KEY)
        self.evaluator = MetricsEvaluator()
        print("Loading Model...")
        self.inference = InferenceAPI()
        
    def process(self, original_text, loss_rate):
        loss_rate_float = float(loss_rate) / 100.0
        
        # 1. Pipeline simulation
        pipe_res = self.pipeline.transmit(original_text, drop_probability=loss_rate_float)
        corrupted_text = pipe_res['corrupted_text']
        
        # 2. Recovery
        rec_res = self.inference.recover_message(corrupted_text)
        recovered_text = rec_res['recovered_text']
        confidence = rec_res.get('confidence', 1.0)
        
        # 3. Evaluation
        metrics = self.evaluator.evaluate_message(original_text, corrupted_text, recovered_text)
        
        # Format outputs
        packet_info = f"Total Packets: {pipe_res['total_packets']}\n"
        packet_info += f"Received Packets: {pipe_res['received_packets']}\n"
        packet_info += f"Lost Packets: {pipe_res['lost_packets']}\n"
        
        improvement_val = metrics['improvement'] * 100
        
        metrics_info = f"Original vs Corrupted Accuracy: {metrics['word_accuracy_corrupted']*100:.1f}%\n"
        metrics_info += f"Original vs Recovered Accuracy: {metrics['word_accuracy_recovered']*100:.1f}%\n"
        metrics_info += f"Recovery Improvement: {improvement_val:.1f}%\n"
        metrics_info += f"Exact Match: {'Yes' if metrics['exact_match_recovered'] else 'No'}\n"
        
        if confidence < 0.3:
            metrics_info += "\n[!] Low-confidence reconstruction."
            
        candidates_str = ""
        if 'candidates' in rec_res and len(rec_res['candidates']) > 1:
            candidates_str = "\nTop Candidates:\n"
            for i, cand in enumerate(rec_res['candidates']):
                candidates_str += f"{i+1}. {cand['text']} (conf: {cand['confidence']:.2f})\n"
                
        return corrupted_text, recovered_text, packet_info, metrics_info, candidates_str

    def launch(self):
        with gr.Blocks(title="iTantra Offline Text Recovery") as app:
            gr.Markdown("# iTantra: Offline ML-based Text Recovery Prototype")
            gr.Markdown("Simulates packet loss over a wireless channel, AES-GCM encryption/decryption, and text reconstruction using a local sequence-to-sequence model.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Transmitter")
                    input_text = gr.Textbox(label="Original Message", lines=3, placeholder="e.g. I have a pen")
                    loss_slider = gr.Slider(minimum=0, maximum=50, step=5, value=20, label="Packet Loss Rate (%)")
                    submit_btn = gr.Button("Transmit")
                    
                    gr.Examples(
                        examples=[
                            ["I have a pen", 20],
                            ["Please send the document tomorrow", 30],
                            ["The meeting starts at ten in the morning", 15],
                            ["I am going to college today", 25],
                            ["I have a new blue pen for my college", 30]
                        ],
                        inputs=[input_text, loss_slider]
                    )
                
                with gr.Column():
                    gr.Markdown("### Receiver")
                    corrupted_box = gr.Textbox(label="Simulated Received Message (After Reassembly)", lines=3)
                    recovered_box = gr.Textbox(label="ML Recovered Message", lines=3)
                    
                    packet_box = gr.Textbox(label="Packet Information", lines=3)
                    metrics_box = gr.Textbox(label="Accuracy Metrics", lines=4)
                    candidates_box = gr.Textbox(label="Model Candidates (if any)", lines=4)
                    
            submit_btn.click(
                fn=self.process,
                inputs=[input_text, loss_slider],
                outputs=[corrupted_box, recovered_box, packet_box, metrics_box, candidates_box]
            )
            
        app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)

if __name__ == "__main__":
    demo = DemoApp()
    demo.launch()
