# 🌍 Arabic to English Neural Machine Translator | Transformer from Scratch

A powerful, minimal, and educational implementation of a **Neural Machine Translation (NMT)** system that translates **Arabic ➝ English** using a **Transformer-based Seq2Seq model**, built completely **from scratch** in PyTorch — no pretrained models or external tokenizers used.

Deployed with an intuitive **Streamlit Web App** for live interaction and demonstration.

---

## 📌 Features

✅ Transformer architecture (Encoder-Decoder)  
✅ Custom Tokenizer (Word-level with special tokens)  
✅ Trained on Tatoeba (Helsinki-NLP) Arabic-English dataset  
✅ Greedy decoding for inference  
✅ Clean Streamlit app interface for live translation  
✅ Easy to extend, customize, and learn from

---

## 🚀 Demo

Try it yourself:

🔗 **[Live Streamlit App](#)** *https://transmind-arabic-english-h9pqu2tvwjjqdxeq4jan3s.streamlit.app/*
🔗 **[See On Linkedln](#)** *https://transmind-arabic-english-h9pqu2tvwjjqdxeq4jan3s.streamlit.app/*
🔗 **[See Medium Blog](#)** *https://transmind-arabic-english-h9pqu2tvwjjqdxeq4jan3s.streamlit.app/*

---

## 🧠 Concepts Used

### ✨ Transformer (Seq2Seq) – From Scratch
- Encoder-Decoder architecture using `torch.nn.Transformer`
- Positional Encoding to retain sequence order
- Multi-head attention and feed-forward layers
- Greedy decoding loop for generating output

### ✨ Tokenization
- Custom word-level tokenizer
- Handles Arabic and English text
- Special token handling: `<sos>`, `<eos>`, `<pad>`, `<unk>`

### ✨ Dataset
- **Helsinki-NLP / Tatoeba** dataset via HuggingFace `datasets` library
- Combined `validation` and `test` sets for training

---

## 🧪 Model Architecture

```text
[Arabic Text] → Tokenizer → Embedding + Positional Encoding → Encoder → Decoder → Linear Layer → Softmax → [English Translation]
