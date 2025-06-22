# PRODIGY_GA_02
# 🧠 GPT-2 LoRA Chatbot Finetuner

This is a **plug-and-play Python script** to finetune GPT-2 using your own chatbot-style data (like user-bot messages) with **LoRA (Low-Rank Adaptation)** for fast and memory-efficient training. Built for internships, GenAI projects, or just flexing your AI skills 💻🔥

---

## 🚀 Features

- 📁 File picker to load your CSV (no hardcoded paths)
- 🧠 Label-masking before `bot:` — trains the model to only predict bot replies
- 🔄 Custom train-validation split
- 🧰 LoRA via PEFT for efficient finetuning (smaller VRAM needed)
- 🤖 GPT-2 based transformer model
- ✅ Final model + tokenizer saved in `./gpt2_lora_model`

---

## 🗂️ Expected CSV Format

Your dataset must have two columns:

```csv
user,bot
hi,hello there!
how are you?,i'm doing great! what about you?
