import argparse
import torch
from model import GPT, GPTConfig


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config") or ckpt.get("model_args")
    if config is None:
        raise ValueError("Checkpoint is missing model config")
    gpt_conf = GPTConfig(**config)
    model = GPT(gpt_conf)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def decode_bytes(tokens) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser(description="Generate text from FF-trained Tiny Shakespeare model")
    p.add_argument("--ckpt", type=str, default="out_ff/ff_final.pt")
    p.add_argument("--prompt", type=str, default="The thing about gooses is ")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed)

    model = load_model(args.ckpt, device)

    prompt_bytes = args.prompt.encode("utf-8")
    idx = torch.tensor(list(prompt_bytes), dtype=torch.long, device=device)[None, :]

    with torch.no_grad():
        out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    generated = out[0, idx.size(1):].tolist()
    text = decode_bytes(generated)
    print(text)


if __name__ == "__main__":
    main()
