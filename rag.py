"""
Hybrid RAG + Finetune LLM Pipeline

Kullanım:
  1. pip install chromadb sentence-transformers
  2. python rag.py index      → Makale chunk'larını vectorize et
  3. python rag.py chat       → RAG + finetune model ile sohbet
  4. python rag.py search     → Sadece retrieval test

Apple Silicon M4 Mini 16GB için optimize.
"""
import sys, json, re
from pathlib import Path

import yaml

# --- Config ---
def load_config(path="config.yaml"):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    ft = cfg["finetune"]
    rag = cfg.get("rag", {})
    return {
        "model": ft["model"],
        "adapter_dir": Path(ft["adapter_dir"]),
        "fused_dir": Path(ft["fused_dir"]),
        "processed_dir": Path(cfg["paths"]["output_dir"]),
        "system_prompt": ft["system_prompt"],
        "generation": ft["generation"],
        # RAG settings
        "embedding_model": rag.get("embedding_model", "BAAI/bge-small-en-v1.5"),
        "chunk_size": rag.get("chunk_size", 512),
        "chunk_overlap": rag.get("chunk_overlap", 50),
        "top_k": rag.get("top_k", 3),
        "chroma_dir": Path(rag.get("chroma_dir", "./chroma_db")),
    }

CFG = load_config()


# --- Chunking ---
def chunk_text(text: str, size: int, overlap: int) -> list[dict]:
    """Token-aware chunking with metadata preservation."""
    chunks = []
    # Section başlıklarını bul
    lines = text.split("\n")
    current_section = ""
    current_text = []

    for line in lines:
        if line.startswith("## ") or line.startswith("### ") or line.startswith("#### "):
            # Önceki section'ı chunk'la
            if current_text:
                section_text = "\n".join(current_text)
                words = section_text.split()
                for i in range(0, len(words), size - overlap):
                    chunk = " ".join(words[i:i + size])
                    if len(chunk) > 50:
                        chunks.append({
                            "text": chunk,
                            "section": current_section,
                        })
            current_section = line.lstrip("#").strip()
            current_text = []
        else:
            current_text.append(line)

    # Son section
    if current_text:
        section_text = "\n".join(current_text)
        words = section_text.split()
        for i in range(0, len(words), size - overlap):
            chunk = " ".join(words[i:i + size])
            if len(chunk) > 50:
                chunks.append({
                    "text": chunk,
                    "section": current_section,
                })

    return chunks


# --- Index ---
def index():
    """Makale .txt dosyalarını chunk'la ve ChromaDB'ye indexle."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install chromadb sentence-transformers")

    print(f"Loading embedding model: {CFG['embedding_model']}")
    embedder = SentenceTransformer(CFG["embedding_model"])

    # ChromaDB
    CFG["chroma_dir"].mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CFG["chroma_dir"]))

    # Mevcut collection varsa sil
    try:
        client.delete_collection("papers")
    except Exception:
        pass
    collection = client.create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"},
    )

    # Tüm processed .txt dosyalarını oku
    txt_files = sorted(CFG["processed_dir"].glob("*.txt"))
    if not txt_files:
        sys.exit(f"No .txt files in {CFG['processed_dir']}")

    all_chunks = []
    for f in txt_files:
        arxiv_id = f.stem
        text = f.read_text(encoding="utf-8")

        # Title çıkar
        title = ""
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        chunks = chunk_text(text, CFG["chunk_size"], CFG["chunk_overlap"])
        for i, chunk in enumerate(chunks):
            chunk["arxiv_id"] = arxiv_id
            chunk["title"] = title
            chunk["chunk_id"] = f"{arxiv_id}_chunk_{i}"
            all_chunks.append(chunk)

        print(f"  {arxiv_id}: {len(chunks)} chunks ({title[:60]})")

    # Embed ve kaydet
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "arxiv_id": c["arxiv_id"],
            "title": c["title"],
            "section": c["section"],
        } for c in all_chunks],
    )

    print(f"\n✓ Indexed {len(all_chunks)} chunks from {len(txt_files)} papers → {CFG['chroma_dir']}")


# --- Search ---
def search(query: str = None):
    """Retrieval test."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install chromadb sentence-transformers")

    if query is None:
        query = input("Query: ").strip()

    embedder = SentenceTransformer(CFG["embedding_model"])
    client = chromadb.PersistentClient(path=str(CFG["chroma_dir"]))
    collection = client.get_collection("papers")

    q_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=CFG["top_k"],
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nTop {CFG['top_k']} results for: '{query}'\n")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        doc = results["documents"][0][i][:200]
        print(f"  [{i+1}] (score: {1-dist:.3f}) {meta['title'][:50]} | {meta['section']}")
        print(f"      {doc}...\n")

    return results


def retrieve(query: str) -> str:
    """Query → context string for LLM."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return ""

    embedder = SentenceTransformer(CFG["embedding_model"])
    client = chromadb.PersistentClient(path=str(CFG["chroma_dir"]))
    collection = client.get_collection("papers")

    q_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=CFG["top_k"],
        include=["documents", "metadatas"],
    )

    # Context oluştur
    context_parts = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        context_parts.append(
            f"[Source: {meta['title']} | Section: {meta['section']}]\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


# --- Chat (Hybrid RAG + Fine-tuned LLM) ---
def chat():
    """RAG context + fine-tuned model ile sohbet."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        sys.exit("pip install mlx-lm")

    # Model yükle
    fused = CFG["fused_dir"]
    if fused.exists():
        print(f"Loading fused model: {fused}")
        model, tokenizer = load(str(fused))
    elif CFG["adapter_dir"].exists() and (CFG["adapter_dir"] / "adapters.safetensors").exists():
        print(f"Loading {CFG['model']} + adapter")
        model, tokenizer = load(CFG["model"], adapter_path=str(CFG["adapter_dir"]))
    else:
        print(f"Loading base model: {CFG['model']}")
        model, tokenizer = load(CFG["model"])

    gen = CFG["generation"]
    print("Hybrid RAG Chat started (type 'q' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            break

        # 1. Retrieve relevant context
        context = retrieve(user_input)

        # 2. Build prompt with context
        if context:
            system = (
                f"{CFG['system_prompt']}\n\n"
                f"Use the following research paper excerpts to answer the question. "
                f"Be specific and cite details from the sources when possible.\n\n"
                f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
            )
        else:
            system = CFG["system_prompt"]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=gen["max_tokens"], verbose=False,
        )
        print(f"AI: {response}\n")


# --- Main ---
if __name__ == "__main__":
    commands = {"index": index, "search": search, "chat": chat}

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python rag.py [{' | '.join(commands)}]")
        print("\n  1. index  → Chunk & embed papers into ChromaDB")
        print("  2. search → Test retrieval")
        print("  3. chat   → Hybrid RAG + fine-tuned LLM chat")
        sys.exit(1)

    commands[sys.argv[1]]()