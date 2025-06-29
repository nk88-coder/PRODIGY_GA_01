# PRODIGY_GA_02
# 🧠 GPT-2 Chatbot Finetuner (No LoRA)

This is a **plug-and-play Python script** to finetune GPT-2 using your own chatbot-style data (like user-bot messages). Designed for internships, GenAI projects, or to show off your AI game 💬⚡

✅ Currently fine-tuned on Shakespearean text for poetic and dramatic conversations.

---

## 🚀 Features

- 📁 File picker to load your CSV (no hardcoded paths)
- 🧠 Label-masking before `bot:` — trains the model to only predict bot replies
- 🔄 Custom train-validation split
- 🤖 GPT-2 based transformer model
- ✅ Final model + tokenizer saved in `./gpt2_chatbot_model`

---

## 🗂️ Expected CSV Format

Your dataset must have two columns:

```csv
user,bot
hi,hello there!
how are you?,i'm doing great! what about you?


