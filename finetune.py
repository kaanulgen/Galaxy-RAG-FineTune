"""
Finetune Pipeline: arxiv data → mlx-lm QLoRA → domain expert LLM

Kullanım:
  1. pip install "mlx-lm[train]"
  2. python arxiv_tex_data.py         → makale indir & train.jsonl üret
  3. python finetune.py prepare       → train/valid/test split
  4. python finetune.py train         → QLoRA fine-tune
  5. python finetune.py test          → validation loss
  6. python finetune.py fuse          → adapter + model birleştir
  7. python finetune.py chat          → interaktif sohbet
"""
import sys, json, random, subprocess
from pathlib import Path

import yaml


# --- Config ---
def load_config(path="config.yaml"):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    ft = cfg["finetune"]
    return {
        "model": ft["model"],
        "data_dir": Path(ft["data_dir"]),
        "adapter_dir": Path(ft["adapter_dir"]),
        "fused_dir": Path(ft["fused_dir"]),
        "source_jsonl": Path(cfg["paths"]["output_dir"]) / "train.jsonl",
        "train_split": ft["train_split"],
        "valid_split": ft["valid_split"],
        "training": ft["training"],
        "lora": ft["lora"],
        "generation": ft["generation"],
        "system_prompt": ft["system_prompt"],
    }

CFG = load_config()


# --- 1. Data Preparation ---
def prepare():
    """train.jsonl → data/{train,valid,test}.jsonl"""
    CFG["data_dir"].mkdir(exist_ok=True)

    items = [json.loads(l) for l in CFG["source_jsonl"].read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(42)
    random.shuffle(items)

    n = len(items)
    t, v = int(n * CFG["train_split"]), int(n * (CFG["train_split"] + CFG["valid_split"]))
    splits = {"train": items[:t], "valid": items[t:v], "test": items[v:]}

    for name, data in splits.items():
        path = CFG["data_dir"] / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} examples → {path}")

    # LoRA config
    CFG["adapter_dir"].mkdir(exist_ok=True)
    lora_cfg = {
        "num_layers": CFG["lora"]["num_layers"],
        "lora_parameters": {
            "rank": CFG["lora"]["rank"],
            "scale": CFG["lora"]["scale"],
            "dropout": CFG["lora"]["dropout"],
        },
    }
    with open(CFG["adapter_dir"] / "adapter_config.json", "w") as f:
        json.dump(lora_cfg, f, indent=2)
    print(f"  LoRA config → {CFG['adapter_dir'] / 'adapter_config.json'}")


# --- 2. Train ---
def train():
    t = CFG["training"]
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", CFG["model"],
        "--data", str(CFG["data_dir"]),
        "--adapter-path", str(CFG["adapter_dir"]),
        "--train",
        "--iters", str(t["iters"]),
        "--batch-size", str(t["batch_size"]),
        "--learning-rate", str(t["learning_rate"]),
        "--steps-per-report", str(t["steps_per_report"]),
        "--steps-per-eval", str(t["steps_per_eval"]),
        "--save-every", str(t["save_every"]),
        "--num-layers", str(CFG["lora"]["num_layers"]),
        "--max-seq-length", str(t["max_seq_length"]),
    ]
    if t.get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")
    if t.get("mask_prompt"):
        cmd.append("--mask-prompt")

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


# --- 3. Test ---
def test():
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", CFG["model"],
        "--adapter-path", str(CFG["adapter_dir"]),
        "--data", str(CFG["data_dir"]),
        "--test",
        "--max-seq-length", str(CFG["training"]["max_seq_length"]),
    ]
    subprocess.run(cmd)


# --- 4. Fuse ---
def fuse():
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", CFG["model"],
        "--adapter-path", str(CFG["adapter_dir"]),
        "--save-path", str(CFG["fused_dir"]),
        "--dequantize",
    ]
    print(f"Fusing → {CFG['fused_dir']}")
    subprocess.run(cmd)


# --- 5. Chat ---
def chat():
    try:
        from mlx_lm import load, generate
    except ImportError:
        sys.exit("pip install mlx-lm")

    fused = CFG["fused_dir"]
    if fused.exists():
        print(f"Loading fused: {fused}")
        model, tokenizer = load(str(fused))
    else:
        print(f"Loading {CFG['model']} + adapter")
        model, tokenizer = load(CFG["model"], adapter_path=str(CFG["adapter_dir"]))

    gen = CFG["generation"]
    print("Chat started (type 'q' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            break
        messages = [
            {"role": "system", "content": CFG["system_prompt"]},
            {"role": "user", "content": user_input},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=gen["max_tokens"], verbose=False)
        print(f"AI: {response}\n")


# --- Main ---
if __name__ == "__main__":
    commands = {"prepare": prepare, "train": train, "test": test, "fuse": fuse, "chat": chat}

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python finetune.py [{' | '.join(commands)}]")
        print("\n  1. prepare → Split data  2. train → QLoRA  3. test → Evaluate  4. fuse → Merge  5. chat → Chat")
        sys.exit(1)

    commands[sys.argv[1]]()