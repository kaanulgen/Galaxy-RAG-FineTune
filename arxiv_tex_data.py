"""arXiv TeX → Finetune Data"""
import re, json, tarfile, gzip, time, urllib.request
from pathlib import Path
from dataclasses import dataclass, field

import yaml

# --- Config & Data ---
@dataclass
class Config:
    arxiv_ids: list[str]
    sources_dir: Path
    output_dir: Path
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_len: int = 50
    delay: int = 2

    @classmethod
    def from_yaml(cls, path="config.yaml"):
        c = yaml.safe_load(open(path, encoding="utf-8"))
        return cls(
            arxiv_ids=c["arxiv_ids"],
            sources_dir=Path(c["paths"]["sources_dir"]),
            output_dir=Path(c["paths"]["output_dir"]),
            **{k: c["processing"][k] for k in c.get("processing", {})},
        )

@dataclass
class Paper:
    arxiv_id: str
    title: str = ""
    abstract: str = ""
    sections: list[dict] = field(default_factory=list)

    @property
    def full_text(self):
        parts = [f"# {self.title}"] if self.title else []
        if self.abstract:
            parts.append(f"\n## Abstract\n{self.abstract}")
        for s in self.sections:
            h = {"section": "##", "subsection": "###", "subsubsection": "####"}.get(s["type"], "##")
            parts.append(f"\n{h} {s['title']}\n{s['content']}")
        return "\n".join(parts)

# --- TeX Cleaning ---
_TEX_SUBS = [
    (r'(?<!\\)%.*?$',                                   ''),
    (r'\\text(?:bf|it|rm|tt|)\{([^}]*)\}',              r'\1'),
    (r'\\(?:emph|underline)\{([^}]*)\}',                r'\1'),
    (r'\\(?:cite[pt]?|ref|eqref|autoref)\{[^}]*\}',    '[REF]'),
    (r'\\(?:label|footnote)\{[^}]*\}',                  ''),
    (r'\\begin\{(figure|table)\}.*?\\end\{\1\}',        ''),
    (r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}',    r' [EQ] '),
    (r'\$\$(.+?)\$\$',                                  r' [MATH] '),
    (r'\$(.+?)\$',                                      r' \1 '),
    (r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?',   ''),
    (r'[{}]',                                            ''),
]

def clean_tex(text: str) -> str:
    for p, r in _TEX_SUBS:
        text = re.sub(p, r, text, flags=re.DOTALL | re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()

# --- Download & Extract ---
def download(arxiv_id: str, dest: Path) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{arxiv_id}.tar.gz"
    if path.exists():
        return path
    try:
        req = urllib.request.Request(f"https://arxiv.org/e-print/{arxiv_id}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
        return path
    except Exception as e:
        print(f"  [ERR] {arxiv_id}: {e}")
        return None

def extract_tex(archive: Path) -> str | None:
    try:
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as tar:
                texs = [m for m in tar.getmembers() if m.name.endswith(".tex") and not m.name.startswith(".")]
                if not texs:
                    return None
                pick = next((f for f in texs if any(x in f.name.lower() for x in ["main", "paper", "ms.tex"])),
                            max(texs, key=lambda x: x.size))
                if f := tar.extractfile(pick):
                    return f.read().decode("utf-8", errors="ignore")
        else:
            with gzip.open(archive, "rt", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "\\begin{document}" in content:
                    return content
    except Exception:
        pass
    return None

# --- Parse & Chunk ---
def parse_tex(tex: str, arxiv_id: str, min_len: int) -> Paper:
    paper = Paper(arxiv_id=arxiv_id)
    if m := re.search(r'\\title\s*(?:\[.*?\])?\s*\{(.+?)\}', tex, re.DOTALL):
        paper.title = clean_tex(m.group(1))
    if m := re.search(r'\\begin\{abstract\}(.+?)\\end\{abstract\}', tex, re.DOTALL):
        paper.abstract = clean_tex(m.group(1))

    body = m.group(1) if (m := re.search(r'\\begin\{document\}(.+?)\\end\{document\}', tex, re.DOTALL)) else tex
    matches = list(re.finditer(r'\\(section|subsection|subsubsection)\*?\{([^}]+)\}', body))

    for i, mt in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = clean_tex(body[mt.end():end])
        if len(content) > min_len:
            paper.sections.append({"type": mt.group(1), "title": clean_tex(mt.group(2)), "content": content})
    return paper

def chunk(text: str, size: int, overlap: int, min_len: int) -> list[str]:
    words = text.split()
    return [c for i in range(0, len(words), size - overlap)
            if len(c := " ".join(words[i:i + size])) > min_len]

# --- Training Data
SECTION_PROMPTS = {
    "introduction": "Explain the introduction and motivation of this paper.",
    "method":       "Describe the methodology used in this paper.",
    "result":       "Summarize the key results of this paper.",
    "experiment":   "Describe the experimental setup and findings.",
    "discussion":   "Explain the discussion section of this paper.",
    "conclusion":   "Summarize the conclusions of this paper.",
    "related":      "Summarize the related work discussed in this paper.",
}

def make_chat(system: str, user: str, assistant: str, source: str) -> dict:
    """ChatML format — mlx-lm ve çoğu model bunu bekler."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
    }

def create_training_data(papers: dict[str, Paper], cfg: Config) -> list[dict]:
    # config'den system prompt oku (yoksa default)
    try:
        ft_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")).get("finetune", {})
        sys_prompt = ft_cfg.get("system_prompt", "You are a research assistant specializing in galaxy clusters and cosmology.")
    except Exception:
        sys_prompt = "You are a research assistant specializing in galaxy clusters and cosmology."
    data = []

    for p in papers.values():
        if p.abstract:
            data.append(make_chat(
                sys_prompt,
                f"Summarize the paper titled '{p.title}'.",
                p.abstract,
                p.arxiv_id,
            ))

        # Section-level Q&A
        for s in p.sections:
            if len(s["content"]) < 100:
                continue
            prompt = next((v for k, v in SECTION_PROMPTS.items() if k in s["title"].lower()),
                          f"Explain the '{s['title']}' section.")
            data.append(make_chat(
                sys_prompt,
                f"Paper: {p.title}\nSection: {s['title']}\n\n{prompt}",
                s["content"][:4000],
                p.arxiv_id,
            ))

        # Chunked context
        for i, c in enumerate(chunk(p.full_text, cfg.chunk_size, cfg.chunk_overlap, cfg.min_len)):
            data.append(make_chat(
                sys_prompt,
                "Analyze the following academic text and explain its key points.",
                c,
                f"{p.arxiv_id}_chunk_{i}",
            ))
    return data

# --- Main ---
def main():
    cfg = Config.from_yaml()
    cfg.sources_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    papers = {}
    for i, aid in enumerate(cfg.arxiv_ids, 1):
        print(f"[{i}/{len(cfg.arxiv_ids)}] {aid}", end=" ")
        if (path := download(aid, cfg.sources_dir)) and (tex := extract_tex(path)):
            papers[aid] = parse_tex(tex, aid, cfg.min_len)
            (cfg.output_dir / f"{aid}.txt").write_text(papers[aid].full_text, encoding="utf-8")
            print(f"✓ {len(papers[aid].sections)} sections")
        else:
            print("✗")
        time.sleep(cfg.delay)

    data = create_training_data(papers, cfg)

    out = cfg.output_dir / "train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✓ {len(papers)}/{len(cfg.arxiv_ids)} papers → {len(data)} examples → {out}")
    return papers, data

if __name__ == "__main__":
    main()