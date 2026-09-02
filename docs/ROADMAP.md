# MiniMind English Fork — Quality Roadmap

This fork keeps MiniMind's ~64M dense recipe and native-PyTorch pipeline. The goal is a **better English model at the same size and roughly the same wall-clock**, not a bigger model.

Upstream: [jingyaogong/minimind](https://github.com/jingyaogong/minimind). This repo: English-first docs, CLI, eval, and (next) English data + training efficiency.

---

## 1. Current-state diagnosis

### Architecture (already modern — do not grow it)

| Knob | MiniMind-3 default | Notes |
|------|--------------------|--------|
| Params | ~64M dense | 8 × 768, GQA 8Q/4KV, `head_dim=96` |
| Vocab | 6400, tied embeddings | ~4.9M params in embed/lm_head. Keep this size. |
| Attention | QK-Norm + flash SDPA + RoPE 1e6 + YaRN | Already Qwen3-aligned |
| FFN | SwiGLU, `intermediate ≈ π × hidden` rounded to 64 | Fine |
| MoE | 4 experts, top-1, ~198M-A64M | Native PyTorch MoE is ~50% slower. **Not the English default.** |

MobileLLM is right that depth often beats width at ~100M, but MiniMind already chose 8×768 as a **train-speed** tradeoff. Do not add layers unless you drop width and accept slower steps.

### Tokenizer (main English bottleneck)

- BPE + ByteLevel, 6400 merges, trained on **Chinese-heavy SFT** (`train_tokenizer.py` reads `sft_t2t_mini.jsonl`).
- Chinese ≈ 1.5–1.7 chars/token (fragmented). English ≈ 4–5 chars/token (acceptable, but merges are not English-optimized).
- Chat template is already English Qwen-style (`<|im_start|>`, `<think>`, `<tool_call>`). Keep it.
- Retraining on English **pretrain** text (not SFT chat) at the same 6400 vocab is a quality win without adding parameters. It **breaks** compatibility with upstream `.pth` weights. That is acceptable for this fork.

### Data (not in the repo)

Original files live on ModelScope / Hugging Face (`gongjy/minimind_dataset`). Nothing is vendored here.

| File | Size | Role | Language |
|------|------|------|----------|
| `pretrain_t2t_mini.jsonl` | 1.2GB | Quick pretrain | Chinese-heavy mix |
| `pretrain_t2t.jsonl` | 10GB | Full pretrain | Chinese-heavy mix |
| `sft_t2t_mini.jsonl` | 1.6GB | Quick SFT + some tool-call | Chinese-heavy |
| `sft_t2t.jsonl` | 14GB | Full SFT | Chinese-heavy + qwen3 synthetic |
| `dpo.jsonl` | 53MB | DPO | En+Zh (LlamaFactory 20k) |
| `rlaif.jsonl` | 24MB | PPO/GRPO prompts | From SFT, last assistant blank |
| `agent_rl*.jsonl` | 86+18MB | Tool-use RL | Chinese-centric tools/math |

Schema to keep (do not invent new formats):

```json
{"text": "plain pretrain document"}
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"chosen": [...], "rejected": [...]}
```

### Training waste (biggest quality-per-hour leak)

`PretrainDataset` / `SFTDataset` tokenize **one document per row** and **pad to `max_seq_len`**. Short Chinese/English docs become mostly pad. The model still attends over pad (mask only hits the CE ignore index).

Defaults:

- Pretrain: `max_seq_len=340`, `batch=32`, `accum=8`, `lr=5e-4`, 2 epochs, **no warmup**. Cosine: `lr * (0.1 + 0.45 * (1 + cos(π t / T)))`.
- AdamW with no param groups (norms/bias get the same decay).
- SFT: `lr=1e-5`, `max_seq_len=768`, `batch=16`.
- Residual init is vanilla `post_init()` — no GPT-2 `1/sqrt(2N)` residual scaling.
- Eval prompts, tool-call tests, and the default RM (`internlm2-1.8b-reward`) are Chinese.

For a 64M model, **tokens seen / hour** and **data quality** dominate architecture tweaks.

---

## 2. Non-goals

- Do not raise the default param count above ~64M.
- Do not default to MoE.
- Do not wrap training in Hugging Face `Trainer` / TRL (keep native PyTorch).
- Do not jump vocab to 32k–128k (embeddings would eat the budget).
- Do not train long CoT / huge R1 traces on 64M as the main path.
- Do not replace the educational “from-scratch” code with a megatron stack.

---

## 3. Phased implementation

### Phase 0 — English-first repo (this PR)

**Goal:** The fork is usable in English without changing the model.

**Done / in this change:**

- `README.md` is English; original Chinese is `README_zh.md`.
- Trainer argparse, comments, and logs translated.
- Eval prompts, tool-call tests, dataset comments, LoRA comments, WebUI tools/system prompt in English (WebUI stays bilingual).

**Files:** `README*.md`, `trainer/*`, `scripts/*`, `eval_llm.py`, `dataset/lm_dataset.py`, `model/model_lora.py`.

**Effect:** Zero quality change. Required for maintaining the fork.

---

### Phase 1 — Same FLOPs, more real tokens (highest ROI)

**Goal:** Raise effective tokens/step without a longer 3090 run.

**Changes:**

1. **Document packing** in `dataset/lm_dataset.py`
   - Concatenate docs with `eos` until `max_seq_len`.
   - Labels: next-token on all real tokens; `-100` on pad and (optionally) on the first token after a document boundary.
   - Add `--pack` (default on for pretrain).
2. **Attention pad mask** — pass a real `attention_mask` so SDPA does not mix pad keys (today pretrain often omits this).
3. **Warmup** in `get_lr`: linear 2–5% of steps, then existing cosine to 0.1× lr.
4. **AdamW groups:** `weight_decay=0.1` on matmul weights; `0` on bias, RMSNorm, embeddings.
5. Log **tokens/sec** and **padding fraction** so packing gains are visible.

**Files:** `dataset/lm_dataset.py`, `trainer/trainer_utils.py`, `trainer/train_pretrain.py`, `trainer/train_full_sft.py`.

**Expected:** 1.5–3× more non-pad tokens per hour on mini data. Same wall-clock, better language model.

**Risk:** Packing changes the number of optimizer steps per epoch (fewer rows). Keep a **token budget** (e.g. 0.4–0.8B tokens for mini) rather than a fixed epoch count.

---

### Phase 2 — English tokenizer, still vocab=6400

**Goal:** Better English merges without growing embeddings.

**Changes:**

- Point `train_tokenizer.py` at a **pretrain English sample** (10k–100k FineWeb-Edu / TinyStories docs), not SFT chat.
- Keep `VOCAB_SIZE=6400` and the same special/tool/think tokens.
- Script: print chars/token on held-out English vs the current tokenizer. Ship only if English chars/token stays ≥ ~4.0 and decode is lossless.
- Save under `model/` and document that upstream Chinese `.pth` files will not load cleanly.

**Files:** `trainer/train_tokenizer.py`, `model/tokenizer.json`, `model/tokenizer_config.json`.

**Expected:** Fewer broken English wordpieces → slightly better SFT and fewer tokens per sentence (more content per 340–512 ctx).

**Risk:** Breaks weight sharing with jingyaogong models. Do this **before** any English pretrain, not after.

---

### Phase 3 — English datasets (same jsonl schema)

**Goal:** Mini-sized English mix that a 64M model can actually learn from. Quality over dump-the-web.

**Recommended mini mix (~1–2GB on disk, similar to upstream mini):**

| Split | Source | Why | Convert to |
|-------|--------|-----|------------|
| ~70% pretrain | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) sample (or `sample-10BT` then subsample) | Educational web; SmolLM’s main lever | `{"text"}` |
| ~15% pretrain | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Fluency for tiny models | `{"text"}` |
| ~10% pretrain | Cosmopedia-v2 or OpenWebMath **sample** | Synthetic textbooks / light math | `{"text"}` |
| ~5% pretrain | python-edu or Stack-Edu sample | Optional code; keep short | `{"text"}` |
| SFT | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) and/or Magpie-Pro / Tulu-3 **subset** | Instruction style at this scale | `conversations` |
| DPO | UltraFeedback English (or `HuggingFaceH4/ultrafeedback_binarized`) | Preference without a Chinese RM | `chosen`/`rejected` |

**Scripts to add:** `scripts/prepare_english_data.py`

- Download via `datasets`, length-filter (drop < 64 or > 8k chars for mini), light dedup (`datasketch` is already a dep), write jsonl into `./dataset/`.
- Targets: `pretrain_en_mini.jsonl` (~1.2GB), `sft_en_mini.jsonl` (~1.5GB), `dpo_en.jsonl`.
- Full run later: 8–10GB FineWeb-Edu, not 100B. A 64M model saturates; extra noisy web does not help.

**Do not** use the original Chinese mini files as the English default. Optional 5–10% Chinese later for bilingual, not v1.

---

### Phase 4 — Tiny-model recipe + English eval

**Goal:** Stabilize 64M training and measure English, not C-Eval.

**Changes (all cheap):**

- Residual / output init scale `1/sqrt(2 * n_layers)` on `o_proj` and `down_proj` (GPT-2 / Small-init).
- Optional **z-loss** `1e-4 * logZ²` on the CE logits (PaLM). Flag `--z_loss`, default off until measured.
- SFT lr **2e-5–5e-5** for 64M (1e-5 is conservative).
- Pretrain `max_seq_len=512` once packing exists (340 was a Chinese-char heuristic).
- Eval: replace C-Eval/C-MMLU as the default with **HellaSwag, ARC-Easy, PIQA, OpenBookQA**, plus a 20-prompt English generation suite in `eval_llm.py` (already started). Add a thin `lm-eval-harness` wrapper later; do not vendor a huge bench.

**Files:** `model/model_minimind.py` (`_init_weights`), `trainer/train_*.py`, `eval_llm.py`, new `scripts/eval_english.py`.

---

### Phase 5 — Preference / RL (optional, after a good SFT)

- **DPO** on English UltraFeedback, 1 epoch, lr ≤ 5e-8 (upstream already warns about forgetting).
- **Skip PPO** until there is a small English RM (do not keep `internlm2-1.8b-reward` as default).
- **GRPO / CISPO** only with **verifiable** rewards (GSM8K-lite, simple code exec). 64M will not win open-ended RLAIF.

---

## 4. Recommended English mini recipe (64M dense)

Single 3090-class GPU, same order of cost as upstream “~2 hours”:

```bash
# after Phase 1–3 land
cd trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_en_mini.jsonl \
  --hidden_size 768 --num_hidden_layers 8 \
  --max_seq_len 512 --batch_size 32 --accumulation_steps 8 \
  --learning_rate 5e-4 --epochs 1 \
  --from_weight none

python train_full_sft.py \
  --data_path ../dataset/sft_en_mini.jsonl \
  --from_weight pretrain \
  --max_seq_len 768 --batch_size 16 \
  --learning_rate 3e-5 --epochs 1
```

Token target, not “2 epochs of padded rows”: about **0.4–1.0B pretrain tokens** then **0.1–0.3B SFT tokens** for the mini path. Full path: 2–5B FineWeb-Edu + 0.5B SFT if you have the hours.

Defaults to keep: `bfloat16`, `grad_clip=1.0`, `use_moe=0`, flash attn, tied embeddings, vocab 6400.

---

## 5. What not to do

| Temptation | Why skip |
|------------|----------|
| Vocab 32k+ | Embeddings become 25M+ of a 64M model |
| 16 layers at 768 | Slower, more VRAM; not “fast training” |
| MoE as default | +50% train time in this codebase |
| Distill from 7B into 64M as the only pretrain | Useful as a **supplement**, not a replacement for clean English text |
| 10k-token CoT SFT | 64M cannot hold that style; wastes ctx |
| Train on the full Chinese `sft_t2t.jsonl` “for more data” | Wrong language; English quality will stall |
| muP / custom kernels first | Diminishing returns vs packing + data |

---

## 6. Suggested PR order after this one

1. **Packing + warmup + AdamW groups** (Phase 1) — no data download required to review the code.
2. **`prepare_english_data.py`** + tokenizer retrain (Phase 2–3).
3. **Init scale + English eval** (Phase 4).
4. DPO English (Phase 5) only if SFT generation is already coherent.

Each step should be measurable: padding%, tokens/sec, English chars/token, and a fixed 20-prompt qualitative file plus ARC-Easy / HellaSwag when compute allows.

Trained checkpoints for this fork are stored and overwritten on [meet447/minimind](https://huggingface.co/meet447/minimind). Chat UI: [meet447/minimind-chat](https://huggingface.co/spaces/meet447/minimind-chat). After a pretrain/SFT run, push with `HF_TOKEN` set and `python scripts/push_to_hub.py`. After Space app edits, `python scripts/push_space.py`.
