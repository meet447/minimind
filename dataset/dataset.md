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

**English mix** (this fork): see [docs/ROADMAP.md](../docs/ROADMAP.md) Phase 3. Target names: `pretrain_en_mini.jsonl`, `sft_en_mini.jsonl`, `dpo_en.jsonl`.
