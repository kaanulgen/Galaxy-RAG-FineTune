"""
Q&A JSON Birleştirici: qa_*.json → data/{train,valid,test}.jsonl

Kullanım:
  1. qa_*.json dosyalarını qa_data/ klasörüne koy
  2. python merge_qa.py
"""
import json, random, glob
from pathlib import Path
import re

import yaml

# Config
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
QA_DIR = Path("qa_data")           # qa_*.json dosyalarının olduğu yer
DATA_DIR = Path(cfg["finetune"]["data_dir"])
TRAIN_SPLIT = cfg["finetune"]["train_split"]
VALID_SPLIT = cfg["finetune"]["valid_split"]

# 1. Tüm qa_*.json dosyalarını oku ve birleştir
all_items = []
for f in sorted(QA_DIR.glob("qa_*.json")):
    text = f.read_text(encoding="utf-8")
    # LaTeX kaçış karakterlerini düzelt: \sigma → \\sigma vb.
    text = text.replace("\\\\", "%%DOUBLE%%")  # mevcut \\ koru
    text = text.replace("\\n", "%%NEWLINE%%")   # \n koru
    text = text.replace("\\\"", "%%QUOTE%%")    # \" koru
    text = text.replace("\\t", "%%TAB%%")       # \t koru
    # Kalan tek \ karakterlerini \\ yap
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    text = text.replace("%%DOUBLE%%", "\\\\")
    text = text.replace("%%NEWLINE%%", "\\n")
    text = text.replace("%%QUOTE%%", "\\\"")
    text = text.replace("%%TAB%%", "\\t")
    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] {f.name}: {e}")
        continue
    all_items.extend(items)
    print(f"  {f.name}: {len(items)} Q&A")

print(f"\nToplam: {len(all_items)} Q&A")

# 2. Shuffle & split
random.seed(42)
random.shuffle(all_items)

n = len(all_items)
t = int(n * TRAIN_SPLIT)
v = int(n * (TRAIN_SPLIT + VALID_SPLIT))
splits = {"train": all_items[:t], "valid": all_items[t:v], "test": all_items[v:]}

# 3. JSONL olarak kaydet
DATA_DIR.mkdir(exist_ok=True)
for name, data in splits.items():
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(data)} examples → {path}")