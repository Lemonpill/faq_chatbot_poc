import os
import re
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv


# -----------------------
# Configure client
# -----------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# -----------------------
# Mock FAQ Data
# -----------------------
DATA = {
    "categories": [
        {"name": "Orders", "questions": [{"q": "How do I place an order?", "a": "To place an order, select a product and click 'Buy Now'."}, {"q": "Can I cancel my order?", "a": "Yes, you can cancel your order within 24 hours."}]},
        {"name": "Shipping", "questions": [{"q": "How long does delivery take?", "a": "Delivery usually takes 3-5 business days."}, {"q": "Do you ship internationally?", "a": "Yes, we ship worldwide."}]},
        {"name": "Payments", "questions": [{"q": "What payment methods are supported?", "a": "We accept credit cards, PayPal, and Apple Pay."}]},
    ]
}


# -----------------------
# Embedding Utils
# -----------------------
def embed(text: str) -> List[float]:
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def tokenize(text: str):
    return set(re.findall(r"\w+", text.lower()))


# -----------------------
# Index (RAG)
# -----------------------
class FAQIndex:
    def __init__(self, data):
        self.entries = []
        self._build(data)

    def _build(self, data):
        print("🔧 Building index...")

        for cat in data["categories"]:
            for item in cat["questions"]:
                text = f"Category: {cat['name']}\nQ: {item['q']}\nA: {item['a']}"
                emb = embed(text)

                self.entries.append({"text": text, "embedding": emb, "category": cat["name"], "question": item["q"], "answer": item["a"]})

        print(f"✅ Indexed {len(self.entries)} FAQ entries\n")

    def search(self, query: str, top_k=3):
        query_emb = embed(query)
        query_tokens = tokenize(query)

        scored = []

        for entry in self.entries:
            emb_score = cosine_similarity(query_emb, entry["embedding"])

            entry_tokens = tokenize(entry["question"])
            overlap = len(query_tokens & entry_tokens)

            # 🔥 hybrid score
            score = emb_score * 0.7 + overlap * 0.3

            scored.append((score, entry, emb_score, overlap))

        scored.sort(key=lambda x: x[0], reverse=True)

        # debug (optional, helps a LOT)
        print("\n--- DEBUG SCORES ---")
        for s in scored[:3]:
            print(f"{s[1]['question']} | total={s[0]:.3f} emb={s[2]:.3f} overlap={s[3]}")
        print("--------------------\n")

        return scored[:top_k]


# -----------------------
# LLM Answer
# -----------------------
def generate_answer(query: str, contexts: List[Dict]):
    context_text = "\n\n".join([c[1]["text"] for c in contexts])

    prompt = f"""
You are a helpful customer support assistant.

- Detect the user's language
- Answer in the SAME language
- Use ONLY the provided context
- If unsure, say you don't know

--- CONTEXT ---
{context_text}
----------------

User question: {query}
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return res.choices[0].message.content.strip()


# -----------------------
# Voice: Record
# -----------------------
def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("✅ Recording saved\n")


# -----------------------
# Voice: Transcribe
# -----------------------
def transcribe(filename="input.wav"):
    with open(filename, "rb") as f:
        res = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
    return res.text


# -----------------------
# Voice: Speak
# -----------------------
def speak(text, filename="output.mp3"):
    res = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)

    with open(filename, "wb") as f:
        f.write(res.read())

    print("🔊 Response saved to output.mp3\n")


# -----------------------
# Main Chat Loop
# -----------------------
def run():
    index = FAQIndex(DATA)

    print("🤖 AI FAQ Bot (text + voice)")
    print("Commands:")
    print("  /voice  -> speak instead of typing")
    print("  /exit   -> quit\n")

    while True:
        user_input = input("You: ")

        if user_input == "/exit":
            break

        # Voice mode
        if user_input == "/voice":
            record_audio()
            user_input = transcribe()
            print(f"📝 You (transcribed): {user_input}")

        # Search
        results = index.search(user_input)

        top_score = results[0][0] if results else 0

        # Confidence check
        if top_score < 0.25:
            answer = "Sorry, I couldn't find a relevant answer."
        else:
            answer = generate_answer(user_input, results)

        print(f"\nBot: {answer}\n")

        # Speak response
        speak(answer)


# -----------------------
# Entry
# -----------------------
if __name__ == "__main__":
    run()
