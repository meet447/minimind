# This fork DOES retrain the tokenizer on English pretrain text. That breaks
# compatibility with upstream MiniMind .pth checkpoints; use only with weights
# trained on this vocabulary.
import argparse
import json
import os
import shutil
import sys
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, '../dataset/pretrain_en_mini.jsonl')
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, '../model_learn_tokenizer')
DEFAULT_MODEL_CONFIG = os.path.join(SCRIPT_DIR, '../model/tokenizer_config.json')
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, '../model')
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36
MIN_TEXT_CHARS = 32
ENGLISH_CHARS_PER_TOKEN_MIN = 4.0

SPECIAL_TOKENS_LIST = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>",
    "<tts_text_eod>", "<tts_text_bos_single>",
]

ADDITIONAL_TOKENS_LIST = [
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<think>", "</think>",
]

ENGLISH_EVAL_TEXTS = [
    (
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast "
        "amounts of text data to understand and generate human-like language. These models use deep "
        "learning techniques, specifically transformers, to process and predict the next word in a "
        "sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in "
        "coding, translation, and creative writing."
    ),
    (
        "The development of sustainable energy is crucial for the future of our planet. As climate "
        "change continues to impact global weather patterns, transitioning from fossil fuels to "
        "renewable sources like solar, wind, and hydroelectric power has become an urgent priority. "
        "Innovations in battery storage technology and smart grid management are essential to ensure "
        "a reliable energy supply."
    ),
    (
        "Python is a high-level programming language known for its clean syntax and rich ecosystem. "
        "It is widely used in data science, machine learning, and web development. Developers can "
        "build complex applications quickly with libraries such as NumPy, Pandas, and PyTorch. "
        "Whether you are a beginner or an expert, Python offers something for everyone."
    ),
    (
        "Interstellar travel refers to journeys between stars or across galaxies. Chemical rockets "
        "are poorly suited to such distances, so scientists study ion drives, nuclear thermal rockets, "
        "and other advanced propulsion concepts. Although human exploration has so far reached only "
        "the Moon, progress in materials science and energy may one day enable missions to Mars and "
        "beyond the solar system."
    ),
]


def _extract_text(record):
    text = record.get('text')
    if isinstance(text, str) and text.strip():
        return text.strip()
    contents = [
        item.get('content')
        for item in record.get('conversations', [])
        if isinstance(item, dict) and item.get('content')
    ]
    if contents:
        return "\n".join(contents).strip()
    return None


def get_texts(data_path, max_docs):
    count = 0
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if count >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _extract_text(data)
            if not text or len(text) < MIN_TEXT_CHARS:
                continue
            yield text
            count += 1


def _load_model_tokenizer_config():
    with open(DEFAULT_MODEL_CONFIG, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_all_special_tokens(special_tokens_num=SPECIAL_TOKENS_NUM):
    num_buffer = special_tokens_num - len(SPECIAL_TOKENS_LIST + ADDITIONAL_TOKENS_LIST)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    return SPECIAL_TOKENS_LIST + ADDITIONAL_TOKENS_LIST + buffer_tokens


def train_tokenizer(data_path, tokenizer_dir, vocab_size, max_docs, special_tokens_num=SPECIAL_TOKENS_NUM):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    all_special_tokens = _build_all_special_tokens(special_tokens_num)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens,
    )
    texts = get_texts(data_path, max_docs)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL_TOKENS_LIST)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in SPECIAL_TOKENS_LIST:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    added_tokens_decoder = {}
    for token in all_special_tokens:
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in SPECIAL_TOKENS_LIST,
        }

    model_config = _load_model_tokenizer_config()
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [
            t for t in SPECIAL_TOKENS_LIST if t != "<|endoftext|>"
        ],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": model_config["chat_template"],
        "tokenizer_class": "PreTrainedTokenizerFast",
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    messages = [
        {"role": "system", "content": "You are an excellent chatbot that always gives me correct responses!"},
        {"role": "user", "content": "Where are you from?"},
        {"role": "assistant", "content": "I am from the Moon"},
        {"role": "user", "content": "Where are you really from?"},
        {"role": "assistant", "content": "I am from Earth"},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    chat_ids = tokenizer(chat_prompt)["input_ids"]
    chat_decoded = tokenizer.decode(chat_ids, skip_special_tokens=False)
    chat_lossless = chat_decoded == chat_prompt

    print('-' * 100)
    print('Lossless decode check (chat template):', chat_lossless)
    print('Tokenizer vocab size:', len(tokenizer))
    print('Chat template encode length:', len(chat_ids))
    print('-' * 100)
    print('English chars/token:')

    english_ratios = []
    english_lossless = True
    for i, text in enumerate(ENGLISH_EVAL_TEXTS):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        lossless = decoded == text
        english_lossless = english_lossless and lossless
        ratio = len(text) / len(encoded)
        english_ratios.append(ratio)
        print(
            f"Sample {i + 1} | Chars: {len(text):4} | Tokens: {len(encoded):3} | "
            f"Chars/token: {ratio:.2f} | Lossless: {lossless}"
        )

    mean_english = sum(english_ratios) / len(english_ratios)
    decode_lossless = chat_lossless and english_lossless
    eval_passed = decode_lossless and mean_english >= ENGLISH_CHARS_PER_TOKEN_MIN

    print('-' * 100)
    print(f"Mean English chars/token: {mean_english:.2f}")
    print(f"Decode lossless: {decode_lossless}")
    print(f"Eval passed: {eval_passed}")

    return eval_passed, mean_english, decode_lossless


def install_tokenizer(tokenizer_dir, model_dir=DEFAULT_MODEL_DIR):
    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tokenizer_dir, name)
        dst = os.path.join(model_dir, name)
        shutil.copy2(src, dst)
        print(f"Installed {src} -> {dst}")


def resolve_data_path(data_path):
    data_path = os.path.abspath(data_path)
    if os.path.isfile(data_path):
        return data_path
    fallback = os.path.join(SCRIPT_DIR, '../dataset/sft_t2t_mini.jsonl')
    if os.path.isfile(fallback):
        print(f"Data path not found: {data_path}")
        print(f"Falling back to: {fallback}")
        return os.path.abspath(fallback)
    print(f"Error: data file not found: {data_path}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 6400-token English BPE tokenizer for MiniMind."
    )
    parser.add_argument(
        '--data_path',
        default=DEFAULT_DATA_PATH,
        help='Pretrain jsonl with {"text": "..."} records (SFT jsonl accepted as fallback)',
    )
    parser.add_argument(
        '--out_dir',
        default=DEFAULT_OUT_DIR,
        help='Output directory for learned tokenizer artifacts',
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=VOCAB_SIZE,
        help='BPE vocabulary size (default: 6400)',
    )
    parser.add_argument(
        '--max_docs',
        type=int,
        default=80000,
        help='Maximum number of training documents to use',
    )
    parser.add_argument(
        '--install',
        action='store_true',
        help='Copy tokenizer.json and tokenizer_config.json into ../model/ after eval passes',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    out_dir = os.path.abspath(args.out_dir)

    train_tokenizer(data_path, out_dir, args.vocab_size, args.max_docs)
    passed, mean_english, decode_lossless = eval_tokenizer(out_dir)

    if not passed:
        if not decode_lossless:
            print("Eval failed: decode is not lossless.", file=sys.stderr)
        if mean_english < ENGLISH_CHARS_PER_TOKEN_MIN:
            print(
                f"Eval failed: mean English chars/token {mean_english:.2f} < "
                f"{ENGLISH_CHARS_PER_TOKEN_MIN}.",
                file=sys.stderr,
            )
        sys.exit(1)

    if args.install:
        install_tokenizer(out_dir)
