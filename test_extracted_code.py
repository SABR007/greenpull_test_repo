import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

def train_model():
    print("⏳ Loading a tiny subset of IMDB data (100 samples)...")
    # We use a tiny dataset so you can actually test this quickly
    dataset = load_dataset("imdb", split="train[:100]") 
    
    # ❌ INEFFICIENT: Loading the full model in 32-bit floating point
    model_name = "bert-base-uncased"
    print(f"⏳ Loading {model_name} in full FP32 precision...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
# ✅ Fixed lines: Rename to what PyTorch expects, then drop the raw text
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])    
    tokenized_datasets.set_format("torch")
    
    dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True)
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    # ❌ INEFFICIENT: Optimizing all 109 million parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    print(f"🚀 Starting full-parameter training on {device}...")
    for epoch in range(2): # Just 2 epochs for testing
        total_loss = 0
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Average Loss: {total_loss/len(dataloader):.4f}")
        
    print("✅ Training complete!")
    return model

if __name__ == "__main__":
    train_model()
 OPTIMIZEDDDDD UEESSESSSS