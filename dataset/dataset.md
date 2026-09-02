# MiniMind Datasets

Place downloaded jsonl files in this directory. Nothing is vendored in git.

**Upstream Chinese files** (original MiniMind): [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [Hugging Face](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

```text
pretrain_t2t_mini.jsonl   # 1.2GB  quick pretrain
sft_t2t_mini.jsonl        # 1.6GB  quick SFT
pretrain_t2t.jsonl        # 10GB   full pretrain
sft_t2t.jsonl             # 14GB   full SFT
dpo.jsonl                 # 53MB
rlaif.jsonl               # 24MB
agent_rl.jsonl            # 86MB
agent_rl_math.jsonl       # 18MB
```

Formats: pretrain `{"text": "..."}`; SFT `{"conversations": [...]}`; DPO `{"chosen": [...], "rejected": [...]}`.

**English mix** (this fork): see [docs/ROADMAP.md](../docs/ROADMAP.md) Phase 3.

| File | ~size | Role |
|------|-------|------|
| `pretrain_en_mini.jsonl` | 1.5–2.0 GB | English pretrain (`{"text": "..."}`) |
| `sft_en_mini.jsonl` | 0.8–1.2 GB | English SFT (`{"conversations": [...]}`) |
| `dpo_en.jsonl` | (later) | English DPO |

Generate the English mini files locally (not vendored in git):

```bash
python scripts/prepare_english_data.py \
  --out-dir dataset \
  --pretrain-gb 1.6 \
  --sft-rows 250000 \
  --seed 42
```

`--dry-run` prints the planned byte/row budgets per source without downloading or writing. Use `--no-dedup` to skip light MinHash deduplication.

**Pretrain mix (streaming):** ~75% `HuggingFaceFW/fineweb-edu` (`sample-10BT`), ~15% `roneneldan/TinyStories`, ~10% `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`). Optional code (`python-edu`) is skipped — the Hub subset is metadata-only.

**SFT:** `HuggingFaceTB/smoltalk` (`all`) → `user` / `assistant` / `system` turns only; tool-call traces dropped.
