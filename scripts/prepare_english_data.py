#!/usr/bin/env python3
"""Stream public HuggingFace datasets into MiniMind English jsonl files."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from datasketch import MinHash
from datasets import IterableDataset, load_dataset

PRETRAIN_OUT = "pretrain_en_mini.jsonl"
SFT_OUT = "sft_en_mini.jsonl"

MIN_PRETRAIN_CHARS = 64
MAX_PRETRAIN_CHARS = 8000

CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)
ALLOWED_ROLES = {"user", "assistant", "system"}
TOOL_MARKERS = ("<tool_call>", "<tools>", "you have access to the following tools")


def has_tool_trace(message: dict) -> bool:
    if message.get("tool_calls") or message.get("tools"):
        return True
    content = str(message.get("content", ""))
    lower = content.lower()
    if any(marker in lower for marker in TOOL_MARKERS):
        return True
    return False


@dataclass
class PretrainSource:
    name: str
    fraction: float
    loader: Callable[[int], IterableDataset]
    text_field: str = "text"


@dataclass
class Stats:
    seen: int = 0
    kept: int = 0
    skipped_empty: int = 0
    skipped_length: int = 0
    skipped_cjk: int = 0
    skipped_dedup: int = 0
    bytes_written: int = 0


class Deduper:
    """Light near-duplicate filter using datasketch MinHash digests."""

    def __init__(self, enabled: bool = True, num_perm: int = 32):
        self.enabled = enabled
        self.num_perm = num_perm
        self._seen: set[bytes] = set()

    def is_duplicate(self, text: str) -> bool:
        if not self.enabled:
            return False
        digest = self._digest(text)
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False

    def _digest(self, text: str) -> bytes:
        normalized = " ".join(text.split()).lower()
        mh = MinHash(num_perm=self.num_perm)
        if len(normalized) <= 3:
            mh.update(normalized.encode("utf-8"))
            return bytes(mh.digest())
        for i in range(0, len(normalized) - 2, 3):
            mh.update(normalized[i : i + 3].encode("utf-8"))
        return bytes(mh.digest())


def is_mostly_cjk(text: str, threshold: float = 0.20) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    cjk_count = sum(1 for ch in letters if CJK_RE.match(ch))
    return (cjk_count / len(letters)) > threshold


def passes_pretrain_text(text: str) -> tuple[bool, str]:
    text = text.strip()
    if not text:
        return False, "empty"
    if len(text) < MIN_PRETRAIN_CHARS or len(text) > MAX_PRETRAIN_CHARS:
        return False, "length"
    if is_mostly_cjk(text):
        return False, "cjk"
    return True, ""


def normalize_messages(messages: list[dict]) -> Optional[list[dict]]:
    if any(has_tool_trace(message) for message in messages):
        return None

    cleaned: list[dict] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        if role not in ALLOWED_ROLES:
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    roles = {turn["role"] for turn in cleaned}
    if "user" not in roles or "assistant" not in roles:
        return None
    return cleaned


def load_fineweb_edu(seed: int) -> IterableDataset:
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=seed, buffer_size=10_000)


def load_tinystories(seed: int) -> IterableDataset:
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    return ds.shuffle(seed=seed, buffer_size=10_000)


def load_cosmopedia(seed: int) -> IterableDataset:
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=seed, buffer_size=10_000)


def pretrain_sources(include_code: bool = False) -> list[PretrainSource]:
    fractions = {
        "fineweb-edu": 0.70,
        "tinystories": 0.15,
        "cosmopedia": 0.10,
        "code": 0.05,
    }
    if not include_code:
        fractions["fineweb-edu"] += fractions.pop("code")

    return [
        PretrainSource("fineweb-edu", fractions["fineweb-edu"], load_fineweb_edu),
        PretrainSource("tinystories", fractions["tinystories"], load_tinystories),
        PretrainSource("cosmopedia-v2", fractions["cosmopedia"], load_cosmopedia),
    ]


def write_pretrain(
    out_path: Path,
    target_bytes: int,
    seed: int,
    dedup: bool,
    dry_run: bool,
) -> dict[str, Stats]:
    deduper = Deduper(enabled=dedup)
    source_stats: dict[str, Stats] = {}
    sources = pretrain_sources(include_code=False)

    if dry_run:
        for source in sources:
            budget = int(target_bytes * source.fraction)
            source_stats[source.name] = Stats(bytes_written=budget)
        return source_stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for source in sources:
            stats = Stats()
            source_stats[source.name] = stats
            budget = int(target_bytes * source.fraction)
            stream = source.loader(seed)

            for row in stream:
                if stats.bytes_written >= budget:
                    break

                stats.seen += 1
                text = str(row.get(source.text_field, ""))
                ok, reason = passes_pretrain_text(text)
                if not ok:
                    if reason == "empty":
                        stats.skipped_empty += 1
                    elif reason == "length":
                        stats.skipped_length += 1
                    elif reason == "cjk":
                        stats.skipped_cjk += 1
                    continue

                text = text.strip()
                if deduper.is_duplicate(text):
                    stats.skipped_dedup += 1
                    continue

                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                line_bytes = len(line.encode("utf-8"))
                if stats.bytes_written + line_bytes > budget and stats.kept > 0:
                    break

                out_f.write(line)
                stats.kept += 1
                stats.bytes_written += line_bytes

    return source_stats


def load_smoltalk(seed: int) -> IterableDataset:
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    return ds.shuffle(seed=seed, buffer_size=10_000)


def write_sft(
    out_path: Path,
    target_rows: int,
    seed: int,
    dedup: bool,
    dry_run: bool,
) -> Stats:
    stats = Stats()
    deduper = Deduper(enabled=dedup)

    if dry_run:
        stats.bytes_written = target_rows
        return stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream = load_smoltalk(seed)

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in stream:
            if stats.kept >= target_rows:
                break

            stats.seen += 1
            conversations = normalize_messages(row.get("messages", []))
            if conversations is None:
                stats.skipped_empty += 1
                continue

            joined = "\n".join(
                f"{turn['role']}: {turn['content']}" for turn in conversations
            )
            if is_mostly_cjk(joined):
                stats.skipped_cjk += 1
                continue

            dedup_key = joined
            if deduper.is_duplicate(dedup_key):
                stats.skipped_dedup += 1
                continue

            line = json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n"
            out_f.write(line)
            stats.kept += 1
            stats.bytes_written += len(line.encode("utf-8"))

    return stats


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / 1_000_000_000


def print_plan(
    out_dir: Path,
    pretrain_gb: float,
    sft_rows: int,
    seed: int,
    dedup: bool,
    dry_run: bool,
) -> None:
    target_bytes = int(pretrain_gb * 1_000_000_000)
    print("MiniMind English data preparation")
    print(f"  out_dir:      {out_dir}")
    print(f"  pretrain_gb:  {pretrain_gb:.2f} (~{target_bytes:,} bytes)")
    print(f"  sft_rows:     {sft_rows:,}")
    print(f"  seed:         {seed}")
    print(f"  dedup:        {dedup}")
    print(f"  dry_run:      {dry_run}")
    print()
    print("Outputs:")
    print(f"  {out_dir / PRETRAIN_OUT}")
    print(f"  {out_dir / SFT_OUT}")
    print()
    print("Pretrain mix (streaming; code subset skipped — metadata-only on Hub):")
    for source in pretrain_sources(include_code=False):
        budget = int(target_bytes * source.fraction)
        print(
            f"  {source.fraction * 100:4.1f}%  {source.name:<16} "
            f"~{bytes_to_gb(budget):.3f} GB"
        )
    print()
    print("SFT source:")
    print("  100.0%  HuggingFaceTB/smoltalk (config=all) -> conversations")


def print_summary(
    pretrain_stats: dict[str, Stats],
    sft_stats: Stats,
    out_dir: Path,
) -> None:
    print()
    print("Pretrain summary:")
    total_bytes = 0
    total_rows = 0
    for name, stats in pretrain_stats.items():
        total_bytes += stats.bytes_written
        total_rows += stats.kept
        print(
            f"  {name}: kept={stats.kept:,} bytes={stats.bytes_written:,} "
            f"(seen={stats.seen:,}, len={stats.skipped_length:,}, "
            f"cjk={stats.skipped_cjk:,}, dedup={stats.skipped_dedup:,})"
        )
    print(
        f"  total: {total_rows:,} rows, {bytes_to_gb(total_bytes):.3f} GB "
        f"-> {out_dir / PRETRAIN_OUT}"
    )

    print()
    print("SFT summary:")
    print(
        f"  kept={sft_stats.kept:,} rows, {bytes_to_gb(sft_stats.bytes_written):.3f} GB "
        f"(seen={sft_stats.seen:,}, invalid={sft_stats.skipped_empty:,}, "
        f"cjk={sft_stats.skipped_cjk:,}, dedup={sft_stats.skipped_dedup:,}) "
        f"-> {out_dir / SFT_OUT}"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream English HF datasets into MiniMind jsonl files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory for output jsonl files (default: dataset)",
    )
    parser.add_argument(
        "--pretrain-gb",
        type=float,
        default=1.6,
        help="Target on-disk size for pretrain jsonl in decimal GB (default: 1.6)",
    )
    parser.add_argument(
        "--sft-rows",
        type=int,
        default=250_000,
        help="Target number of SFT conversations (default: 250000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for streaming datasets (default: 42)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable light MinHash deduplication",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts and exit without writing files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    if args.pretrain_gb <= 0:
        print("error: --pretrain-gb must be positive", file=sys.stderr)
        return 1
    if args.sft_rows <= 0:
        print("error: --sft-rows must be positive", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    dedup = not args.no_dedup
    print_plan(out_dir, args.pretrain_gb, args.sft_rows, args.seed, dedup, args.dry_run)

    target_bytes = int(args.pretrain_gb * 1_000_000_000)
    pretrain_stats = write_pretrain(
        out_dir / PRETRAIN_OUT,
        target_bytes=target_bytes,
        seed=args.seed,
        dedup=dedup,
        dry_run=args.dry_run,
    )
    sft_stats = write_sft(
        out_dir / SFT_OUT,
        target_rows=args.sft_rows,
        seed=args.seed,
        dedup=dedup,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print()
        print("Dry run only — no files written.")
        return 0

    print_summary(pretrain_stats, sft_stats, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
