#!/usr/bin/env python3
"""Hugging Face Space: chat with the latest meet447/minimind weights."""
import os
import re
import spaces  # must import before torch

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import gradio as gr

MODEL_ID = os.environ.get("MODEL_ID", "meet447/minimind")

_state = {"model": None, "tokenizer": None, "revision": "unknown"}


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def load_weights(force: bool = False) -> str:
    local_dir = snapshot_download(MODEL_ID, force_download=force)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["revision"] = os.path.basename(os.path.realpath(local_dir))
    return f"Loaded {MODEL_ID} ({_state['revision']})"


def _history_to_messages(history) -> list[dict]:
    messages = []
    if not history:
        return messages
    if isinstance(history[0], dict):
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages
    for turn in history:
        if not turn:
            continue
        user, assistant = (turn + [None, None])[:2]
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    return messages


def _apply_template(tokenizer, messages, open_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, open_thinking=bool(open_thinking), **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@spaces.GPU(duration=60)
def chat(message: str, history, temperature: float, max_new_tokens: float, open_thinking: bool) -> str:
    """Reply with the latest MiniMind checkpoint from meet447/minimind."""
    if _state["model"] is None:
        load_weights(force=False)
    model, tokenizer = _state["model"], _state["tokenizer"]
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})
    prompt = _apply_template(tokenizer, messages, open_thinking)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=float(temperature) > 0,
            temperature=max(float(temperature), 1e-5),
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text if open_thinking else _strip_think(text) or text


load_weights(force=False)

with gr.Blocks(title="MiniMind Chat") as demo:
    gr.Markdown(
        f"""# MiniMind Chat
Talk to the latest weights from [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}).
This Space reloads `main` on startup. After a new `push_to_hub.py` upload, click **Reload latest weights** (or restart the Space).

This is a **~64M** model on ZeroGPU. The current T4 checkpoint was trained on the upstream Chinese-heavy mini sets, so English replies can be weak until the English data/tokenizer work lands.
"""
    )
    status = gr.Textbox(value="Ready", label="Status", interactive=False)
    with gr.Row():
        temperature = gr.Slider(0.0, 1.2, value=0.85, step=0.05, label="Temperature")
        max_new_tokens = gr.Slider(32, 512, value=160, step=8, label="Max new tokens")
        open_thinking = gr.Checkbox(value=False, label="Show <think> tokens")
    reload_btn = gr.Button("Reload latest weights")
    reload_btn.click(lambda: load_weights(force=True), outputs=status)
    gr.ChatInterface(
        fn=chat,
        type="messages",
        additional_inputs=[temperature, max_new_tokens, open_thinking],
        examples=[
            ["Why is the sky blue?"],
            ["Write a Python function to compute Fibonacci numbers."],
            ["Explain photosynthesis in two sentences."],
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(mcp_server=True)
