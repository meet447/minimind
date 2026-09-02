#!/usr/bin/env bash
# MiniMind-128 English T4 recipe (fp16, packed, compile).
# Run from repo root after:
#   python scripts/prepare_english_data.py --out-dir dataset --pretrain-gb 1.6 --sft-rows 250000
#   python trainer/train_tokenizer.py --data_path dataset/pretrain_en_mini.jsonl --install
set -euo pipefail
cd "$(dirname "$0")/../trainer"

PRETRAIN_DATA="${PRETRAIN_DATA:-../dataset/pretrain_en_mini.jsonl}"
SFT_DATA="${SFT_DATA:-../dataset/sft_en_mini.jsonl}"
DTYPE="${DTYPE:-float16}"

python train_pretrain.py \
  --preset minimind-128 \
  --data_path "$PRETRAIN_DATA" \
  --dtype "$DTYPE" \
  --device cuda:0 \
  --fast 1 --pack 1 --use_compile 1 \
  --max_seq_len 512 --batch_size 32 --accumulation_steps 4 \
  --num_workers 4 \
  --learning_rate 5e-4 --epochs 1 \
  --from_weight none \
  --log_interval 10 --save_interval 2000

python train_full_sft.py \
  --preset minimind-128 \
  --data_path "$SFT_DATA" \
  --dtype "$DTYPE" \
  --device cuda:0 \
  --fast 1 --use_compile 1 \
  --max_seq_len 512 --batch_size 8 --accumulation_steps 2 \
  --num_workers 4 \
  --learning_rate 3e-5 --epochs 1 \
  --from_weight pretrain \
  --log_interval 10 --save_interval 2000

echo "Weights: ../out/pretrain_768_L16.pth and ../out/full_sft_768_L16.pth"
