#!/usr/bin/env python3
"""Create or update the MiniMind Gradio chat Space.

Reads HF_TOKEN from the environment. Never writes the token to disk.

Usage:
  export HF_TOKEN=hf_...
  python scripts/push_space.py --space-id meet447/minimind-chat

Free personal accounts cannot create Gradio CPU Spaces. This script requests
ZeroGPU (zero-a10g), which is the free Gradio hosting path (up to 2 Spaces).
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "spaces" / "chat"


def main():
    parser = argparse.ArgumentParser(description="Upload the MiniMind chat Space")
    parser.add_argument("--space-id", default="meet447/minimind-chat")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Update MiniMind chat Space")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the environment. Do not put it in git.")
    if not SPACE_DIR.is_dir():
        raise SystemExit(f"Missing Space app: {SPACE_DIR}")

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message=args.commit_message,
        ignore_patterns=[".git*", "__pycache__*", "*.pyc"],
    )
    print(f"Uploaded to https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
