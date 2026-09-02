#!/usr/bin/env python3
"""Convert MiniMind .pth weights and upload to a Hugging Face model repo.

Reads HF_TOKEN from the environment. Never writes the token to disk.

Usage:
  export HF_TOKEN=hf_...
  python scripts/push_to_hub.py --repo-id meet447/minimind
"""
import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import HfApi, login

from model.model_minimind import MiniMindConfig

_spec = importlib.util.spec_from_file_location("convert_model", ROOT / "scripts" / "convert_model.py")
convert_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_model)


def _write_model_card(path: Path, repo_id: str, hidden_size: int, num_layers: int, use_moe: bool):
    moe = "MoE" if use_moe else "dense"
    path.write_text(
        f"""---
license: apache-2.0
language:
- en
- zh
library_name: transformers
pipeline_tag: text-generation
tags:
- minimind
- pytorch
base_model: jingyaogong/minimind-3
---

# MiniMind ({repo_id})

~64M MiniMind-3 {moe} checkpoint from the English-first fork
[{repo_id.split('/')[0]}/minimind](https://github.com/{repo_id.split('/')[0]}/minimind).

This Hub repo is the **living weight store**. New pretrain / SFT / later English
runs overwrite `main` so you can always pull the latest.

## Current weights (T4, 2026-09-02)

| File | Stage | Notes |
|------|--------|--------|
| root (`config.json` + `model.safetensors`) | full SFT | Transformers / Qwen3-compatible layout |
| `pytorch/full_sft_{hidden_size}.pth` | full SFT | Native MiniMind trainer format |
| `pytorch/pretrain_{hidden_size}.pth` | pretrain | Native MiniMind trainer format |

**Architecture:** {num_layers} × {hidden_size}, GQA 8Q/4KV, vocab 6400, SwiGLU, QK-Norm, RoPE 1e6, tied embeddings, no MoE.

**This run:** 1 epoch packed pretrain on `pretrain_t2t_mini.jsonl` (Tesla T4, fp16, batch 96) then 1 epoch SFT on `sft_t2t_mini.jsonl` (batch 16, lr 3e-5). Trainer extras: sequence packing, 3% warmup, AdamW param groups, `torch.compile`.

**Honest quality note:** the tokenizer and mini datasets are still the upstream **Chinese-heavy** ones. These weights validate the fork trainer. They are **not** an English-from-scratch model yet. English tokenizer + FineWeb-Edu / SmolTalk data come next ([docs/ROADMAP.md](https://github.com/{repo_id.split('/')[0]}/minimind/blob/master/docs/ROADMAP.md)).

## Load (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype="auto")
messages = [{{"role": "user", "content": "Why is the sky blue?"}}]
inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
out = model.generate(inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Load (this repo's trainer)

```bash
# download native .pth into ./out
huggingface-cli download {repo_id} pytorch/full_sft_{hidden_size}.pth --local-dir .
# then
python eval_llm.py --weight full_sft --hidden_size {hidden_size} --num_hidden_layers {num_layers}
```

## Update this repo

From a checkout that has `out/full_sft_{hidden_size}.pth` (and optionally `out/pretrain_{hidden_size}.pth`):

```bash
export HF_TOKEN=hf_...
python scripts/push_to_hub.py --repo-id {repo_id}
```

License: Apache 2.0. Architecture and tokenizer follow [jingyaogong/minimind](https://github.com/jingyaogong/minimind).
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Convert MiniMind weights and push to Hugging Face")
    parser.add_argument("--repo-id", default="meet447/minimind", help="Hugging Face model repo id")
    parser.add_argument("--save-dir", default=str(ROOT / "out"), help="Directory with native .pth files")
    parser.add_argument("--weight", default="full_sft", help="Primary weight prefix to convert (default: full_sft)")
    parser.add_argument("--hidden-size", default=768, type=int)
    parser.add_argument("--num-hidden-layers", default=8, type=int)
    parser.add_argument("--use-moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--export-dir", default=str(ROOT / "out" / "hf_export"), help="Local export folder")
    parser.add_argument("--native-format", action="store_true", help="Export MiniMind AutoModel layout instead of Qwen3")
    parser.add_argument("--private", action="store_true", help="Create/keep the Hub repo private")
    parser.add_argument("--commit-message", default="", help="Hub commit message")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the environment. Do not put it in git.")

    moe_suffix = "_moe" if args.use_moe else ""
    torch_name = f"{args.weight}_{args.hidden_size}{moe_suffix}.pth"
    torch_path = Path(args.save_dir) / torch_name
    if not torch_path.is_file():
        raise SystemExit(f"Missing weights: {torch_path}")

    export_dir = Path(args.export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    (export_dir / "pytorch").mkdir()

    convert_model.lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    transformers_dir = export_dir / "transformers"
    transformers_dir.mkdir()
    old_cwd = os.getcwd()
    os.chdir(ROOT / "scripts")
    try:
        dest = str(transformers_dir.resolve())
        src = str(torch_path.resolve())
        if args.native_format:
            convert_model.convert_torch2transformers_minimind(src, dest, dtype=torch.float16)
        else:
            try:
                convert_model.convert_torch2transformers(src, dest, dtype=torch.float16)
            except Exception as exc:
                print(f"Qwen3 export failed ({exc}); falling back to MiniMind native format")
                shutil.rmtree(transformers_dir)
                transformers_dir.mkdir()
                convert_model.convert_torch2transformers_minimind(src, dest, dtype=torch.float16)
    finally:
        os.chdir(old_cwd)

    for item in transformers_dir.iterdir():
        dest = export_dir / item.name
        if dest.exists():
            dest.unlink() if dest.is_file() else shutil.rmtree(dest)
        shutil.move(str(item), str(dest))
    transformers_dir.rmdir()

    shutil.copy2(torch_path, export_dir / "pytorch" / torch_name)
    pretrain_path = Path(args.save_dir) / f"pretrain_{args.hidden_size}{moe_suffix}.pth"
    if pretrain_path.is_file():
        shutil.copy2(pretrain_path, export_dir / "pytorch" / pretrain_path.name)

    shutil.copy2(ROOT / "model" / "model_minimind.py", export_dir / "model_minimind.py")
    _write_model_card(export_dir / "README.md", args.repo_id, args.hidden_size, args.num_hidden_layers, bool(args.use_moe))

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    message = args.commit_message or f"Update {args.weight} {args.hidden_size}{moe_suffix} weights"
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        commit_message=message,
        ignore_patterns=[".git*", "__pycache__*", "*.pyc"],
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
