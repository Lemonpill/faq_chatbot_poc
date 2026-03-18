import os
import re
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from typing import List, Dict
from openai import OpenAI
from openai.types.create_embedding_response import Usage
from openai.types.completion_usage import CompletionUsage
from dotenv import load_dotenv
from mutagen.mp3 import MP3
import wave

EMBED_MODEL = "text-embedding-3-small"
AUDIO_MODEL = "gpt-4o-mini-tts"
TEXT_MODEL = "gpt-4.1-nano"
TRANS_MODEL = "gpt-4o-mini-transcribe"


MODEL_PRICE = {  # USD / Per 1M tokens
    EMBED_MODEL: {
        "Cost": 0.02,
        "Batch cost": 0.01,
    },
    AUDIO_MODEL: {  # Audio
        "Input": None,
        "Output": 12.00,
        "Estimated cost": 0.015,  # / minute
    },
    TEXT_MODEL: {
        "Input": 0.10,
        "Cached input": 0.025,
        "Output": 0.40,
    },
    TRANS_MODEL: {
        "Input": 1.25,
        "Output": 5.00,
        "Estimated cost": 0.003,  # / minute
    },
}


def get_mp3_duration(filename: str):
    audio = MP3(filename)
    return audio.info.length


def get_wav_duration(filename: str):
    duration_seconds = None
    with wave.open(filename, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration_seconds = frames / float(rate)
        print(f"Audio duration: {duration_seconds:.2f} seconds")
    return duration_seconds


def get_usage_cost_embedding(usage: Usage):
    costs = MODEL_PRICE[EMBED_MODEL]
    cost_per_million = costs["Cost"]  # USD per 1M tokens
    cost = cost_per_million * usage.total_tokens / 1000000
    print(f"get_usage_cost_embedding: total ${cost:.6f}")
    return cost


def get_usage_cost_speak(duration: float):
    minutes = duration / 60
    cost = MODEL_PRICE[AUDIO_MODEL]["Estimated cost"] * minutes
    print(f"get_usage_cost_speak: duration {duration}; cost {cost}")
    return cost


def get_usage_cost_text(usage: CompletionUsage):
    tokens_in = usage.prompt_tokens
    tokens_out = usage.completion_tokens
    cost_in = MODEL_PRICE[TEXT_MODEL]["Input"]
    cost_out = MODEL_PRICE[TEXT_MODEL]["Output"]
    total_in = cost_in * tokens_in / 1000000
    total_out = cost_out * tokens_out / 1000000
    cost = total_in + total_out
    print(f"get_usage_cost_text: total {cost} (in {total_in}; out {total_out})")
    return cost


def get_usage_cost_transcribe(duration: float):
    minutes = duration / 60
    cost = MODEL_PRICE[TRANS_MODEL]["Estimated cost"] * minutes
    print(f"get_usage_cost_transcribe: duration {duration}; cost {cost}")
    return cost


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
    get_usage_cost_embedding(res.usage)
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
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    usage = res.usage
    get_usage_cost_text(usage)

    return res.choices[0].message.content.strip()


# -----------------------
# Voice: Record
# -----------------------
def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")  # ✅ int16
    sd.wait()
    wav.write(filename, fs, recording)
    print(f"✅ Recording saved ({duration}s) to {filename}")


# -----------------------
# Voice: Transcribe
# -----------------------
def transcribe(filename="input.wav"):
    with open(filename, "rb") as f:
        res = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
    duration = get_wav_duration(filename)
    get_usage_cost_transcribe(duration)
    return res.text


# -----------------------
# Voice: Speak
# -----------------------
def speak(text, filename="output.mp3"):
    res = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    with open(filename, "wb") as f:
        f.write(res.read())
    duration_out = get_mp3_duration(filename)
    get_usage_cost_speak(duration_out)
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
