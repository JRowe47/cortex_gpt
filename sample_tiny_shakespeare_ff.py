import os
import argparse
import pickle
import torch

# Try importing GPT and GPTConfig from model.py or cortex_model.py
GPT = None
GPTConfig = None
try:
    from model import GPT, GPTConfig
except Exception:
    try:
        from cortex_model import GPT, GPTConfig
    except Exception as e:
        raise RuntimeError(
            "Could not import GPT/GPTConfig from model.py or cortex_model.py"
        ) from e


def load_tokenizer(checkpoint_path: str):
    """Load char-level stoi/itos if meta.pkl exists next to checkpoint."""
    meta_path = os.path.join(os.path.dirname(checkpoint_path), "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta.get("stoi"), meta.get("itos")
    return None, None


def make_encode_decode(stoi, itos, vocab_size_from_model: int):
    """Return encode/decode functions for either char or byte level tokens."""
    if stoi is not None and itos is not None:
        stoi_map = dict(stoi)
        itos_list = list(itos)

        def encode(s: str):
            return [stoi_map[c] for c in s]

        def decode(ids):
            return "".join(itos_list[i] for i in ids)

        return encode, decode

    if vocab_size_from_model == 256:
        def encode(s: str):
            return list(s.encode("utf-8"))

        def decode(ids):
            return bytes(int(i) for i in ids).decode("utf-8", errors="ignore")

        return encode, decode

    raise RuntimeError(
        "No tokenizer info found and vocab_size != 256; please provide meta.pkl"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate text from FF Tiny Shakespeare model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "out_ff", "ff_final.pt"),
        help="path to ff_final.pt checkpoint",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_args = checkpoint.get("model_args", checkpoint.get("config", {}))
    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)
    model.load_state_dict(checkpoint["model"])  # load weights
    model.to(device)
    model.eval()

    stoi, itos = load_tokenizer(ckpt_path)
    encode, decode = make_encode_decode(stoi, itos, gpt_conf.vocab_size)
    test_str = "Round trip OK"
    assert decode(encode(test_str)) == test_str

    prompt = "The thing about gooses is "
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=500, temperature=0.8, top_k=200
        )
    text = decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

