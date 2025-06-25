from datasets import load_dataset
import os
import requests

# ✅ Download Tiny Shakespeare Dataset manually (Colab-safe)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
shakespeare_path = "tiny_shakespeare.txt"

if not os.path.exists(shakespeare_path):
    response = requests.get(url)
    with open(shakespeare_path, "w") as f:
        f.write(response.text)

# Load lines manually
with open(shakespeare_path, "r") as f:
    all_lines = f.readlines()

# Extract training/validation texts
train_texts = all_lines[:5000]
val_texts = all_lines[5000:5500]

# Prompt-format
train_texts = ["user: " + t.strip() + "\nbot: [response] end" for t in train_texts]
val_texts = ["user: " + t.strip() + "\nbot: [response] end" for t in val_texts]

from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
val_encodings = tokenizer(val_texts, padding=True, truncation=True, return_tensors='pt')

# Mask user input
def mask_user_input(input_ids_batch, texts):
    labels = input_ids_batch.clone()
    for i, text in enumerate(texts):
        bot_index = text.find("bot:")
        if bot_index == -1:
            continue
        tokens_before_bot = tokenizer(text[:bot_index], truncation=True)['input_ids']
        mask_len = len(tokens_before_bot)
        labels[i, :mask_len] = -100
    return labels

# Dataset
class ConversationDataset(Dataset):
    def __init__(self, encodings, texts):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = mask_user_input(self.input_ids, texts)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Dataset init
train_dataset = ConversationDataset(train_encodings, train_texts)
val_dataset = ConversationDataset(val_encodings, val_texts)

# ✅ Load full GPT-2 (124M) without LoRA
from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config.from_pretrained("gpt2")
config.pad_token_id = tokenizer.pad_token_id
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False
model.gradient_checkpointing_enable()  # optional for long seq

# Trainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer
from torch.utils.data import RandomSampler, DataLoader

# Collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Args
training_args = TrainingArguments(
    output_dir="./gpt2_full_finetuned",
    per_device_train_batch_size=12,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    fp16=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    disable_tqdm=False,
    learning_rate=5e-5,  # 🚨 Slower LR when full model trains
    weight_decay=0.01,
    warmup_steps=200,
    lr_scheduler_type='linear'
)

# Custom Trainer
class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# Init Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 🔥 Train full model
trainer.train()

# Save
model.save_pretrained("./gpt2_full_finetuned")
tokenizer.save_pretrained("./gpt2_full_finetuned")
