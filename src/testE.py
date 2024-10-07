import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Step 1: Load the CSV file
df = pd.read_csv('test-data.csv')

# Step 2: Prepare the data
def combine_text(row):
    return "Pochta bo'limi: {}, Manzil: {}, Xizmat: {}, Yangi Indeks: {}, Eski Indeks: {}".format(
        row["Pochta bo'limi"],
        row["Tuman / Manzil"],
        row["Xizmat ko'rsatuvchi pochta bo'limi"],
        row["Yangi Indekslar"],
        row["Eski Indekslar"]
    )

df['text'] = df.apply(combine_text, axis=1)

# Step 3: Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 4: Load a pre-trained model and tokenizer
model_name = "gpt2"  # You can replace this with any model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Step 5: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Step 6: Split the dataset into train and test sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Step 7: Configure the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
)

# Step 8: Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Step 9: Train the model
trainer.train()

# Optional: Save the model
trainer.save_model("./my_fine_tuned_model")