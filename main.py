import streamlit as st
import torch
import json
import re

# ============================
# 🔹 1. Page Configuration
# ============================
st.set_page_config(
    page_title="Arabic to English Translator",
    page_icon="🌐",
    layout="centered"
)

# ============================
# 🔹 2. Custom Styling
# ============================
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 14px;
        color: gray;
        text-align: center;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# 🔹 3. Load Custom Tokenizer
# ============================
class CustomTokenizer:
    def __init__(self):
        with open("tokenizer.json", "r") as f:
            tokenizer_dict = json.load(f)
        self.word2idx = tokenizer_dict["word2idx"]
        self.idx2word = {int(k): v for k, v in tokenizer_dict["idx2word"].items()}
        self.vocab_size = tokenizer_dict["vocab_size"]
        self.special_tokens = tokenizer_dict["special_tokens"]

    def tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.idx2word.get(idx, "<unk>") for idx in token_ids]
        return " ".join(tokens)

tokenizer = CustomTokenizer()
SOS_TOKEN_ID = tokenizer.word2idx["<sos>"]
EOS_TOKEN_ID = tokenizer.word2idx["<eos>"]
PAD_TOKEN_ID = tokenizer.word2idx["<pad>"]
UNK_TOKEN_ID = tokenizer.word2idx["<unk>"]

# ============================
# 🔹 4. Load Transformer Model
# ============================
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class Transformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size):
        super().__init__()
        self.encoder_embedding = torch.nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = torch.nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = torch.nn.Transformer(d_model, num_heads, num_layers, num_layers, dff, batch_first=True)
        self.fc_out = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.positional_encoding(self.encoder_embedding(src))
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))
        src_padding_mask = (src == PAD_TOKEN_ID)
        tgt_padding_mask = (tgt == PAD_TOKEN_ID)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.fc_out(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    num_layers=6, d_model=256, num_heads=8, dff=1024,
    input_vocab_size=tokenizer.vocab_size,
    target_vocab_size=tokenizer.vocab_size
).to(device)
model.load_state_dict(torch.load("transformer_ara_eng_custom_tokenizer.pth", map_location=device))
model.eval()

# ============================
# 🔹 5. Translation Function
# ============================
def translate(model, source_text, max_len=100):
    model.eval()
    with torch.no_grad():
        source_tokens = [SOS_TOKEN_ID] + tokenizer.encode(source_text) + [EOS_TOKEN_ID]
        source = torch.tensor([source_tokens]).to(device)
        target = torch.tensor([[SOS_TOKEN_ID]]).to(device)

        for _ in range(max_len):
            output = model(source, target)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            if next_token == EOS_TOKEN_ID:
                break
            target = torch.cat([target, torch.tensor([[next_token]]).to(device)], dim=1)

        translated_text = tokenizer.decode(target[0].tolist())
        translated_text = translated_text.replace("<sos>", "").replace("<eos>", "").strip()
        return translated_text

# ============================
# 🔹 6. Streamlit App UI
# ============================
st.markdown('<div class="title">🌐 Arabic ➜ English Translator</div>', unsafe_allow_html=True)
st.markdown("Translate Arabic text into English using a Transformer-based model built from scratch.")

source_text = st.text_area("📝 Enter Arabic text", placeholder="مثال: مرحبًا بك في عالم الذكاء الاصطناعي...", height=120)

if st.button("🔁 Translate"):
    if source_text.strip():
        with st.spinner("Translating..."):
            translated = translate(model, source_text)
        st.success("✅ Translation Completed!")
        st.subheader("🗣️ English Translation:")
        st.write(f"**{translated}**")
    else:
        st.warning("⚠️ Please enter some Arabic text to translate.")

# Footer
st.markdown('<div class="footer">Developed by Sami | CS Department | GenAI Assignment 3</div>', unsafe_allow_html=True)
