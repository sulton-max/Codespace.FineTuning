import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

# Step 1: Load and prepare the data
df = pd.read_csv('test-data.csv')

def combine_text(row):
    return "Input: Pochta bo'limi: {0}\nOutput: Manzil: {1}, Xizmat: {2}, Yangi Indeks: {3}, Eski Indeks: {4}".format(
        row["Pochta bo'limi"],
        row["Tuman / Manzil"],
        row["Xizmat ko'rsatuvchi pochta bo'limi"],
        row["Yangi Indekslar"],
        row["Eski Indekslar"]
    )

df['text'] = df.apply(combine_text, axis=1)

# Step 2: Create Hugging Face dataset
dataset = Dataset.from_pandas(df[['text']])

# Step 3: Load tokenizer and model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Step 4: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Step 5: Split the dataset
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Step 6: Define training arguments
training_args = TrainingArguments(
    output_dir="./results_phi_mini",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs_phi_mini",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=2,
    save_steps=2,
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,
)

# Step 7: Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 8: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# Step 9: Train the model
trainer.train()


# Step 10: Save the model
trainer.save_model("./my_fine_tuned_phi_mini")

# Step 11: Test the model
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test example
test_input = "Input: Pochta bo'limi: Toshkent - 1"
generated_output = generate_response(test_input)
print(f"Input: {test_input}")
print(f"Generated Output: {generated_output}")