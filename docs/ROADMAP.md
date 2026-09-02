# MiniMind English Fork — Quality Roadmap

Upstream: [jingyaogong/minimind](https://github.com/jingyaogong/minimind). This fork is English-first.

**Product goal:** a **~128M dense** model that produces **sensible English** for basic use (short answers, simple summaries, easy explanations) on a **single T4 / 3090-class GPU**. Not SOTA. Not a 7B. Not MoE.

**Locked MiniMind-128 recipe** (measured):

| Knob | MiniMind-3 (shipped) | MiniMind-128 (this goal) |
|------|----------------------|---------------------------|
| Params | 63.91M | **122.91M** |
| Shape | 8 × 768 | **16 × 768** (same width, 2× depth) |
| GQA / head_dim | 8Q/4KV, 96 | unchanged |
| Vocab | 6400 tied | **6400 English BPE**, same specials |
| FFN | SwiGLU, intermediate 2432 | unchanged |
| MoE | off | **off** |

Why 16×768 and not 10×1024 (138M) or 12×896 (125M): at this size depth beats width (MobileLLM / SmolLM). Keeping 768 reuses the MiniMind-3 kernel shape, GQA, and T4 compile path. ~123M is the 128M target without inventing a new width.

**T4 budget (fp16, packing, `torch.compile`):** ~2× the 64M step cost. Expect roughly **4–6h pretrain + 5–8h SFT** for the English mini mix (one epoch each). That is “not heavy compute.”

**Done already:** English repo (Phase 0), packing/warmup/AdamW groups (Phase 1), T4 64M smoke train on Chinese mini data, Hub + Space (`meet447/minimind`, `meet447/minimind-chat`).

**Now:** English tokenizer + English mini jsonl + 128M init, then train from scratch (old 64M `.pth` will not load into 16 layers).

---

## 1. Current-state diagnosis

### Architecture

| Knob | MiniMind-3 default | Notes |
|------|--------------------|--------|
| Params | ~64M dense | 8 × 768, GQA 8Q/4KV, `head_dim=96` |
| Vocab | 6400, tied embeddings | ~4.9M params in embed/lm_head. Keep this size. |
| Attention | QK-Norm + flash SDPA + RoPE 1e6 + YaRN | Already Qwen3-aligned |
| FFN | SwiGLU, `intermediate ≈ π × hidden` rounded to 64 | Fine |
| MoE | 4 experts, top-1, ~198M-A64M | Native PyTorch MoE is ~50% slower. **Not the English default.** |

MiniMind-3 used 8×768 as a train-speed tradeoff. MiniMind-128 spends that 2× depth budget because the product goal is **sensible English**, not a 2-hour 64M tutorial run.

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

- Do not grow past ~128M for v1 (no 16×1024, no 30-layer SmolLM clone).
- Do not default to MoE.
- Do not wrap training in Hugging Face `Trainer` / TRL (keep native PyTorch).
- Do not jump vocab to 32k–128k (embeddings would eat the 128M budget).
- Do not train long CoT / huge R1 traces as the main path.
- Do not replace the educational “from-scratch” code with a megatron stack.
- Do not keep training the Chinese-mini 64M checkpoint as the English product.

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

**Goal:** Stabilize 128M training and measure English, not C-Eval.

**Changes (all cheap):**

- Residual / output init scale `1/sqrt(2 * n_layers)` on `o_proj` and `down_proj` (GPT-2 / Small-init). Required at 16 layers.
- Optional **z-loss** `1e-4 * logZ²` on the CE logits (PaLM). Flag `--z_loss`, default off until measured.
- SFT lr **2e-5–5e-5** (1e-5 is conservative).
- Pretrain `max_seq_len=512` (340 was a Chinese-char heuristic).
- Eval: **HellaSwag, ARC-Easy**, plus a 20-prompt English generation suite (summarize, explain, short Q&A). Add a thin `lm-eval-harness` wrapper later; do not vendor a huge bench.

**Files:** `model/model_minimind.py` (`_init_weights`), `trainer/train_*.py`, `eval_llm.py`, new `scripts/eval_english.py`.

---

### Phase 5 — Preference / RL (optional, after a good SFT)

- **DPO** on English UltraFeedback, 1 epoch, lr ≤ 5e-8 (upstream already warns about forgetting).
- **Skip PPO** until there is a small English RM (do not keep `internlm2-1.8b-reward` as default).
- **GRPO / CISPO** only with **verifiable** rewards (GSM8K-lite, simple code exec). 64M will not win open-ended RLAIF.

---

## 4. Recommended English mini recipe (MiniMind-128)

Tesla T4 16GB (fp16) or a 3090 (bf16). Packing + warmup from Phase 1 required.

```bash
# after tokenizer + jsonl land
cd trainer

python train_pretrain.py \
  --preset minimind-128 \
  --data_path ../dataset/pretrain_en_mini.jsonl \
  --dtype float16 \
  --fast 1 --pack 1 --use_compile 1 \
  --max_seq_len 512 --batch_size 32 --accumulation_steps 4 \
  --learning_rate 5e-4 --epochs 1 \
  --from_weight none

python train_full_sft.py \
  --preset minimind-128 \
  --data_path ../dataset/sft_en_mini.jsonl \
  --dtype float16 \
  --fast 1 --use_compile 1 \
  --from_weight pretrain \
  --max_seq_len 512 --batch_size 8 --accumulation_steps 2 \
  --learning_rate 3e-5 --epochs 1
```

Token target: about **0.4–0.8B pretrain tokens** then **0.1–0.2B SFT tokens**. If T4 OOM, drop batch first, not layers.

Defaults to keep: `use_moe=0`, flash attn, tied embeddings, vocab 6400, residual init `1/sqrt(2N)`.

---

## 5. What not to do

| Temptation | Why skip |
|------------|----------|
| Vocab 32k+ | Embeddings become 25M+ of a 128M model |
| 10×1024 or 12×1024 | Wider, more VRAM, worse small-model quality than 16×768 |
| MoE as default | +50% train time in this codebase |
| Distill from 7B as the only pretrain | Useful as a **supplement**, not a replacement for clean English text |
| 10k-token CoT SFT | 128M still cannot hold that style; wastes ctx |
| Train on the full Chinese `sft_t2t.jsonl` “for more data” | Wrong language; English quality will stall |
| muP / custom kernels first | Diminishing returns vs packing + English data |

---

## 6. Build order for MiniMind-128

1. **Phase 1 packing** — done (merged).
2. **English 6400 tokenizer + `prepare_english_data.py`** (this work) — must land **before** any 128M pretrain.
3. Residual `1/sqrt(2N)` init + `--preset minimind-128` (this work).
4. T4 pretrain + SFT from scratch → push `meet447/minimind` → reload Space.
5. 20-prompt English smoke + optional ARC-Easy / HellaSwag.
6. DPO English only if SFT already writes coherent sentences.

Each step should be measurable: padding%, tokens/sec, English chars/token, and a fixed 20-prompt qualitative file.
