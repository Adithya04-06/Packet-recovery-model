import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from recovery.model import TextRecoveryModel
from config.settings import BEST_MODEL_DIR, GPU_MODEL_DIR, DATA_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE

class RecoveryDataset(Dataset):
    def __init__(self, filepath, model_wrapper: TextRecoveryModel):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.model_wrapper = model_wrapper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs, labels = self.model_wrapper.prepare_data(
            item["corrupted_text"], item["original_text"]
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def train_model():
    checkpoint_path = os.path.join(GPU_MODEL_DIR, "checkpoint.pt")
    
    if not os.path.exists(checkpoint_path) and os.path.exists(os.path.join(BEST_MODEL_DIR, "config.json")):
        print(f"Loading previous CPU best_model weights from {BEST_MODEL_DIR}...")
        model_wrapper = TextRecoveryModel(model_name=BEST_MODEL_DIR)
    else:
        model_wrapper = TextRecoveryModel()
        
    device = model_wrapper.device
    print(f"Device: {device}")
    if device == "cuda" or device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_dataset = RecoveryDataset(os.path.join(DATA_DIR, "train.jsonl"), model_wrapper)
    val_dataset = RecoveryDataset(os.path.join(DATA_DIR, "validation.jsonl"), model_wrapper)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model_wrapper.model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    print("Starting pure PyTorch training loop...")
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    patience = 2
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading true resume state from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from Epoch {start_epoch + 1}")
    elif os.path.exists(os.path.join(BEST_MODEL_DIR, "config.json")):
        # We loaded the CPU weights above, we continue from epoch 3 effectively,
        # but to keep EPOCHS clean, we just start at 0 and run 8 *more* epochs.
        print("Continuing from CPU weights. Starting fresh 8-epoch loop.")
    
    for epoch in range(start_epoch, EPOCHS):
        # Training
        model_wrapper.model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_wrapper.model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            
            if step % 200 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Train Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation
        model_wrapper.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model_wrapper.model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"New best validation loss! Saving model to {GPU_MODEL_DIR}...")
            model_wrapper.model.save_pretrained(GPU_MODEL_DIR)
            model_wrapper.tokenizer.save_pretrained(GPU_MODEL_DIR)
            
            # Save true resume checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_wrapper.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(GPU_MODEL_DIR, "checkpoint.pt"))
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
            
    # Plot training logs
    plt.figure()
    plt.plot(range(start_epoch+1, EPOCHS+1), train_loss_history, label="Train Loss")
    plt.plot(range(start_epoch+1, EPOCHS+1), val_loss_history, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(GPU_MODEL_DIR, "loss_graph.png"))
    print("Saved loss graph.")

if __name__ == "__main__":
    train_model()
