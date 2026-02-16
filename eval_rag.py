"""
RAG Evaluation — Two-Phase Pipeline

Phase 1: Qwen (finetune) → cevap üret
Phase 2: Llama-3.1 (judge) → faithfulness + relevance puanla
         BGE → semantic similarity

Kullanım:
  python eval_rag.py                → full eval
"""
import json, sys, re, gc
from pathlib import Path

import yaml
import numpy as np


# --- Config ---
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
QA_DIR = Path("qa_data")
TOP_K = cfg.get("rag", {}).get("top_k", 3)
EVAL_CFG = cfg.get("eval", {})
JUDGE_MODEL = EVAL_CFG.get("judge_model", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")


def load_qa_pairs() -> list[dict]:
    pairs = []
    for f in sorted(QA_DIR.glob("qa_*.json")):
        text = f.read_text(encoding="utf-8")
        text = text.replace("\\\\", "%%D%%")
        text = text.replace("\\n", "%%N%%").replace("\\\"", "%%Q%%").replace("\\t", "%%T%%")
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
        text = text.replace("%%D%%", "\\\\").replace("%%N%%", "\\n")
        text = text.replace("%%Q%%", "\\\"").replace("%%T%%", "\\t")
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            continue
        for item in items:
            msgs = item["messages"]
            pairs.append({
                "question": msgs[1]["content"],
                "expected": msgs[2]["content"],
                "source": item.get("source", ""),
            })
    return pairs


# --- Retrieval ---
def get_retriever():
    import chromadb
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(cfg["rag"]["embedding_model"])
    client = chromadb.PersistentClient(path=cfg["rag"]["chroma_dir"])
    collection = client.get_collection("papers")
    return embedder, collection


def retrieve(embedder, collection, query: str) -> tuple[str, list[str]]:
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb, n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    parts, ids = [], []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        parts.append(f"[Source: {meta['title']} | Section: {meta['section']}]\n{doc}")
        ids.append(meta["arxiv_id"])
    return "\n\n---\n\n".join(parts), ids


def free_model():
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass


# --- Context Recall ---
def eval_context_recall(pairs, embedder, collection):
    print(f"\n{'='*50}")
    print("CONTEXT RECALL")
    print(f"{'='*50}\n")

    hits, total = 0, 0
    per_source = {}

    for p in pairs:
        _, retrieved_ids = retrieve(embedder, collection, p["question"])
        hit = p["source"] in retrieved_ids
        hits += int(hit)
        total += 1
        if p["source"] not in per_source:
            per_source[p["source"]] = {"hit": 0, "total": 0}
        per_source[p["source"]]["total"] += 1
        if hit:
            per_source[p["source"]]["hit"] += 1

    recall = hits / total if total > 0 else 0
    print(f"  Context Recall@{TOP_K}: {recall:.1%} ({hits}/{total})\n")
    for src, c in sorted(per_source.items()):
        r = c["hit"] / c["total"]
        bar = "█" * int(r * 20) + "░" * (20 - int(r * 20))
        print(f"  {src}: {bar} {r:.0%} ({c['hit']}/{c['total']})")

    return recall


# =============================================
# PHASE 1: Generate answers with Qwen (fine-tuned)
# =============================================
def phase1_generate(pairs, embedder, collection, max_samples=15) -> list[dict]:
    from mlx_lm import load, generate

    ft = cfg["finetune"]
    adapter = Path(ft["adapter_dir"])
    print(f"\n{'='*50}")
    print(f"PHASE 1: Generating answers — {ft['model']}")
    print(f"{'='*50}\n")

    if adapter.exists() and (adapter / "adapters.safetensors").exists():
        model, tokenizer = load(ft["model"], adapter_path=str(adapter))
    else:
        model, tokenizer = load(ft["model"])

    results = []
    for i, p in enumerate(pairs[:max_samples]):
        context, retrieved_ids = retrieve(embedder, collection, p["question"])

        sys_prompt = ft["system_prompt"]
        if context:
            sys_prompt += (
                f"\n\nUse the following research paper excerpts to answer the question. "
                f"Be specific and cite details from the sources when possible.\n\n"
                f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
            )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": p["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=prompt,
                          max_tokens=ft["generation"]["max_tokens"], verbose=False)

        results.append({
            "question": p["question"],
            "expected": p["expected"],
            "response": response,
            "context": context,
            "source": p["source"],
        })
        print(f"  [{i+1}/{min(len(pairs), max_samples)}] ✓ | {p['question'][:70]}")

    del model, tokenizer
    free_model()
    print(f"\n  ✓ {len(results)} answers generated. Qwen unloaded.\n")
    return results


# =============================================
# PHASE 2: Judge with Llama-3.1
# =============================================
def phase2_judge(results: list[dict], embedder) -> dict:
    from mlx_lm import load, generate

    scores = {"faithfulness": [], "relevance": [], "similarity": []}

    # --- Load Llama judge ---
    print(f"{'='*50}")
    print(f"PHASE 2: Judging — {JUDGE_MODEL}")
    print(f"{'='*50}\n")

    judge_m, judge_tok = load(JUDGE_MODEL)

    for i, r in enumerate(results):
        # --- Faithfulness ---
        faith_prompt = (
            f"You are an evaluation judge. Your task is to assess if an answer is faithful to the given context.\n"
            f"A faithful answer ONLY contains information that is supported by the context.\n"
            f"An unfaithful answer contains hallucinated or made-up information not in the context.\n\n"
            f"CONTEXT:\n{r['context'][:2000]}\n\n"
            f"ANSWER:\n{r['response'][:1000]}\n\n"
            f"Rate faithfulness from 1 to 5:\n"
            f"1 = Completely hallucinated, no connection to context\n"
            f"2 = Mostly hallucinated, few supported claims\n"
            f"3 = Mixed, some claims supported and some not\n"
            f"4 = Mostly faithful, minor unsupported details\n"
            f"5 = Completely faithful, everything is in the context\n\n"
            f"Reply with ONLY a single number (1-5):"
        )
        messages = [{"role": "user", "content": faith_prompt}]
        prompt = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(judge_m, judge_tok, prompt=prompt, max_tokens=5, verbose=False)
        nums = re.findall(r'[1-5]', resp)
        faith = int(nums[0]) if nums else 3
        scores["faithfulness"].append(faith)

        # --- Relevance ---
        rel_prompt = (
            f"You are an evaluation judge. Your task is to assess if an answer is relevant to the question.\n"
            f"A relevant answer directly addresses what was asked.\n"
            f"An irrelevant answer talks about unrelated topics or misses the point.\n\n"
            f"QUESTION:\n{r['question']}\n\n"
            f"ANSWER:\n{r['response'][:1000]}\n\n"
            f"Rate relevance from 1 to 5:\n"
            f"1 = Completely irrelevant\n"
            f"2 = Barely relevant\n"
            f"3 = Partially relevant\n"
            f"4 = Mostly relevant\n"
            f"5 = Perfectly relevant, fully addresses the question\n\n"
            f"Reply with ONLY a single number (1-5):"
        )
        messages = [{"role": "user", "content": rel_prompt}]
        prompt = judge_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = generate(judge_m, judge_tok, prompt=prompt, max_tokens=5, verbose=False)
        nums = re.findall(r'[1-5]', resp)
        rel = int(nums[0]) if nums else 3
        scores["relevance"].append(rel)

        f_status = "✓" if faith >= 4 else "⚠" if faith >= 3 else "✗"
        r_status = "✓" if rel >= 4 else "⚠" if rel >= 3 else "✗"
        print(f"  [{i+1}/{len(results)}] Faith={faith}{f_status} Rel={rel}{r_status} | {r['question'][:60]}")

    del judge_m, judge_tok
    free_model()
    print(f"\n  ✓ Judging complete. Llama unloaded.\n")

    # --- Semantic Similarity ---
    print(f"--- Semantic Similarity ---\n")
    for i, r in enumerate(results):
        emb_exp = embedder.encode([r["expected"]])
        emb_res = embedder.encode([r["response"]])
        sim = float(np.dot(emb_exp[0], emb_res[0]) /
                    (np.linalg.norm(emb_exp[0]) * np.linalg.norm(emb_res[0])))
        scores["similarity"].append(sim)
        status = "✓" if sim > 0.85 else "⚠" if sim > 0.7 else "✗"
        print(f"  [{i+1}/{len(results)}] {status} Sim={sim:.2f} | {r['question'][:70]}")

    return scores


# --- Summary ---
def print_summary(recall, scores):
    avg_faith = np.mean(scores["faithfulness"]) if scores["faithfulness"] else 0
    avg_rel = np.mean(scores["relevance"]) if scores["relevance"] else 0
    avg_sim = np.mean(scores["similarity"]) if scores["similarity"] else 0

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Context Recall@{TOP_K}:    {recall:.1%}")
    print(f"  Faithfulness:          {avg_faith:.1f}/5  (judge: Llama-3.1-8B)")
    print(f"  Answer Relevance:      {avg_rel:.1f}/5  (judge: Llama-3.1-8B)")
    print(f"  Semantic Similarity:   {avg_sim:.2f}  (BGE cosine)")
    print(f"{'='*50}")


# --- Main ---
def run_retrieval():
    embedder, collection = get_retriever()
    pairs = load_qa_pairs()
    eval_context_recall(pairs, embedder, collection)


def run_generation():
    embedder, collection = get_retriever()
    pairs = load_qa_pairs()
    results = phase1_generate(pairs, embedder, collection)
    scores = phase2_judge(results, embedder)
    print_summary(0, scores)


def run_full():
    embedder, collection = get_retriever()
    pairs = load_qa_pairs()
    recall = eval_context_recall(pairs, embedder, collection)
    results = phase1_generate(pairs, embedder, collection)
    scores = phase2_judge(results, embedder)
    print_summary(recall, scores)


if __name__ == "__main__":
    commands = {"retrieval": run_retrieval, "generation": run_generation}
    if len(sys.argv) < 2:
        run_full()
    elif sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        print(f"Usage: python eval_rag.py [{' | '.join(commands)}]")