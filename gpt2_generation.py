import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

model_name = "distilgpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token to avoid open-ended generation alignment errors
tokenizer.pad_token = tokenizer.eos_token

custom_data = {
    "text": [
        "Recipe: Mix the flour and sugar together in a large bowl.",
        "Recipe: Bake the mixture at 180 degrees for twenty minutes.",
        "Recipe: Pour the melted chocolate slowly over the fluffy cake.",
        "Recipe: Stir the soup consistently until it reaches a soft boil."
    ]
}
dataset = Dataset.from_dict(custom_data)

# Process sentences into numbers (tokens) for the neural network
def tokenize_function(examples):
    mapping = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)
    mapping["labels"] = mapping["input_ids"].copy()
    return mapping

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./gpt2_chef",
    num_train_epochs=5,          # Trains across the 4 sentences 5 times
    per_device_train_batch_size=2, 
    logging_steps=1,             
    fp16=True,                   # Uses half-precision via T4 GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

print("--- Starting Fine-Tuning ---")
trainer.train()
print("--- Training Complete! ---")

model.eval()

# Prompt word given to the model
prompt = "Recipe:"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Let the AI construct a text generation chain
output = model.generate(
    input_ids, 
    max_length=30,             
    do_sample=True,            
    top_k=50,                  
    top_p=0.95,                
    temperature=0.3          # Forces strict adherence to the recipe style
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n--- AI Generated Recipe Output ---")
print(generated_text)
