"""
chatbot_pdf_groq.py
Usage:
  - Set environment variable GROQ_API_KEY to your Groq key (ne pas committer).
    Exemple (Linux/macOS): export GROQ_API_KEY="gsk_..."
  - Puis: python chatbot_pdf_groq.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# extraction
import PyPDF2

# retrieval
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# détection de langue
from langdetect import detect

# Groq client
try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

load_dotenv()

# --- CONFIG ---
PDF_PATH = "eGov Society - 2015_2024.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --- EXTRACTION TEXTE PDF ---
def extract_text_from_pdf(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable: {path}")
    text_pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in range(len(reader.pages)):
            try:
                page = reader.pages[p]
                txt = page.extract_text() or ""
                text_pages.append(txt)
            except Exception:
                text_pages.append("")
    return "\n\n".join(text_pages)


# --- CHUNKING ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = i + chunk_size
        chunk = text[i:end]
        chunks.append(chunk.strip())
        i = end - overlap
        if i < 0:
            i = 0
    return [c for c in chunks if c]


# --- RETRIEVER ---
class Retriever:
    def __init__(self, passages):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fit(passages)

    def _fit(self, passages):
        self.tfidf = self.vectorizer.fit_transform(passages)

    def query(self, q, top_k=3):
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.tfidf)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i]), self.passages[i]) for i in idx]


# --- PROMPT BUILDER ---
def build_prompt(context_passages, user_question, lang):
    context_text = "\n\n---\n\n".join([f"Passage {i}:\n{p}" for i, _, p in context_passages])

    # Système multilingue selon la langue détectée
    if lang == "fr":
        system = (
            "Tu es un assistant francophone qui répond uniquement en français, "
            "en utilisant UNIQUEMENT les passages fournis comme contexte. "
            "Si la réponse n’est pas dans le contexte, dis 'Je ne sais pas'."
        )
    elif lang == "ar":
        system = (
            "أنت مساعد ذكي يجيب فقط باللغة العربية باستخدام المقاطع المقدمة. "
            "إذا لم تجد الجواب في النص، قل إنك لا تعرف."
        )
    else:
        system = (
            "You are an English-speaking assistant that answers using ONLY the provided context passages. "
            "If the answer is not in the context, say 'I don’t know'."
        )

    user = f"Context:\n{context_text}\n\nQuestion: {user_question}"
    return system, user


# --- GROQ CALL ---
def call_groq(system_prompt, user_prompt, api_key, model="llama-3.3-70b-versatile", max_tokens=512):
    if not HAS_GROQ:
        raise RuntimeError("Le package 'groq' n'est pas installé. pip install groq")
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = client.chat.completions.create(messages=messages, model=model, max_tokens=max_tokens)
    return completion.choices[0].message.content


# --- MAIN ---
def main():
    print("=== Chatbot PDF multilingue (Groq) ===")
    print("Extraction du PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Longueur texte extrait: {len(text)} caractères")

    chunks = chunk_text(text)
    print(f"{len(chunks)} morceaux (chunks) créés.")

    retriever = Retriever(chunks)
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        print("✅ GROQ_API_KEY détectée — utilisation de Groq.")
    else:
        print("⚠️ Pas de clé Groq : fallback local.")

    while True:
        try:
            q = input("\nPose ta question (ou 'exit' pour quitter) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSortie.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Bye 👋")
            break

        # Détection de la langue automatiquement
        try:
            lang = detect(q)
        except Exception:
            lang = "en"

        if lang.startswith("fr"):
            lang = "fr"
        elif lang.startswith("ar"):
            lang = "ar"
        else:
            lang = "en"

        print(f"🌐 Langue détectée : {lang.upper()}")

        top = retriever.query(q, top_k=3)
        print("\nPassages pertinents (score):")
        for i, score, p in top:
            preview = p[:120].replace("\n", " ")
            print(f" - Passage {i} (score {score:.3f}) -> {preview}...")

        if groq_key:
            system_prompt, user_prompt = build_prompt(top, q, lang)
            try:
                answer = call_groq(system_prompt, user_prompt, groq_key)
                print("\n=== Réponse ===")
                print(answer.strip())
            except Exception as e:
                print("Erreur Groq:", e)
                print("Fallback local:")
                print(top[0][2][:200])
        else:
            concat = "\n\n".join([p for (_, _, p) in top])
            print("\n=== Réponse (fallback local) ===")
            print(concat[:1500])


if __name__ == "__main__":
    main()
