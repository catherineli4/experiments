from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# ==== Configuration ====

MODEL_NAME = "allenai/OLMo-2-0425-1B"
DATASET_PATH = "/home/catheri4/alpaca_split_perplexity/datasets/quartiles/q50_75.json"  # JSON or JSONL with "prompt" and "response"
OUTPUT_DIR = "./olmo1b_alpaca_3perp_SFT"
TEXT_FIELD = "prompt"
RESPONSE_FIELD = "response"


# ==== Load Model and Tokenizer ====

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # OLMo doesn't have a pad_token by default

# Automatically infers format (JSON or JSONL)
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def formatting_func(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return {
        "prompt": prompt,
        "completion": example["output"]
    }



# Add this after loading your dataset to inspect the data:
print("Dataset sample:", dataset[0])
print("Formatted sample:", formatting_func(dataset[0]))

# Check if your JSON actually has the expected fields
sample = dataset[0]
print("Available keys:", sample.keys())

# Test tokenization
sample_text = formatting_func(dataset[0])
# tokens = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
# print("Tokenized length:", tokens['input_ids'].shape)
# print("Sample tokens:", tokens['input_ids'][0][:20])  # First 20 tokens

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Smaller batch size for stability
    
    # KEY FIX 2: Much lower learning rate to prevent gradient explosion
    learning_rate=1e-5,  # Reduced from 1e-4 to 1e-5
    
    logging_steps=10,
    save_strategy="epoch",
    
    # KEY FIX 3: Longer sequences with packing
    max_seq_length=1024,
    packing=True,  # Enable packing to combine short sequences
    
    # KEY FIX 4: Gradient clipping to prevent explosion
    max_grad_norm=1.0,  # Clip gradients to prevent NaN
    
    # KEY FIX 5: More conservative training settings
    warmup_steps=100,  # More warmup steps for stability
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    
    # KEY FIX 6: Disable mixed precision to avoid NaN
    fp16=False,
    bf16=False,
    
    # KEY FIX 7: Training stability options
    dataloader_drop_last=True,
    gradient_checkpointing=False,  # Disable to avoid potential issues
    
    # Better logging to monitor training
    logging_first_step=True,
    load_best_model_at_end=False,

    completion_only_loss=True,
)

# ==== Train ====

dataset = dataset.map(formatting_func)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args = training_args,

)

trainer.train()

# ==== Save Final Model ====

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
