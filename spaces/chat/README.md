---
title: MiniMind Chat
emoji: 🧠
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat with the latest meet447/minimind weights
python_version: "3.12"
startup_duration_timeout: 30m
---

# MiniMind Chat

Public Gradio Space for the English-first MiniMind fork.

- Model repo: [meet447/minimind](https://huggingface.co/meet447/minimind)
- Training code: [meet447/minimind](https://github.com/meet447/minimind)

The app downloads `main` from the model repo on startup. After you push new
weights with `python scripts/push_to_hub.py`, click **Reload latest weights**
in the Space (or restart it) to pick them up.

This Space runs on **ZeroGPU** (the free Gradio path). The current ~64M
checkpoint is the T4 packed pretrain + SFT run on the upstream Chinese-heavy
mini datasets.
