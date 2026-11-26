"""Reproducible corpus fetch: public Wikipedia extracts via the MediaWiki API.
Run: python tools/fetch_corpus.py  (writes data/corpus/*.md)"""
import json, urllib.request, urllib.parse
from pathlib import Path

TOPICS = {
    "rag.md": "Retrieval-augmented generation",
    "vector_db.md": "Vector database",
    "faiss.md": "FAISS",
    "llm.md": "Large language model",
    "transformer.md": "Transformer (deep learning)",
    "cosine_similarity.md": "Cosine similarity",
    "gradient_descent.md": "Gradient descent",
    "python.md": "Python (programming language)",
    "fastapi.md": "FastAPI",
    "word_embedding.md": "Word embedding",
    "postgresql.md": "PostgreSQL",
    "rnn.md": "Recurrent neural network",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "corpus"
OUT.mkdir(parents=True, exist_ok=True)
for fname, title in TOPICS.items():
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "titles": title, "redirects": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "rag-eval-corpus/1.0 (educational take-home)"})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    page = next(iter(data["query"]["pages"].values()))
    text = f"# {page['title']}\n\nSource: Wikipedia, article '{page['title']}' (public, CC BY-SA)\n\n" + page.get("extract", "")
    (OUT / fname).write_text(text)
    print(fname, len(text), "chars")
