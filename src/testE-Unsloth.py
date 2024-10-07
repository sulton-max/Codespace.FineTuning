# Import necessary libraries
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Set training parameters
max_seq_length = 2048  # Set to any desired sequence length
dtype = None  # None for auto-detection; use 'float16' for Tesla T4, V100; 'bfloat16' for Ampere+
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

# Load the Phi-3.5 model
model_name = "phi-3.5"  # Replace with the actual model name if different
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Configure LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank of the LoRA adapters
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized gradient checkpointing
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Prepare the data in Phi-3 format
tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

# Load and preprocess the dataset
dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Set to True if you want to enable sequence packing
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1000,  # Adjust the number of training steps as needed
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine-tuned-phi-3.5")
tokenizer.save_pretrained("fine-tuned-phi-3.5")

# Load the fine-tuned model for evaluation
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="fine-tuned-phi-3.5",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)

# Prepare the tokenizer for inference
tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

# Evaluate the model with a sample input
from transformers import TextStreamer

messages = [
    {"from": "human", "value": "Can you explain the theory of relativity in simple terms?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
output = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
)

# Print the generated response
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)