import torch
import os

# Try importing GPT and GPTConfig from model.py or cortex_model.py
GPT = None
GPTConfig = None
try:
    from model import GPT, GPTConfig
except Exception:
    try:
        from cortex_model import GPT, GPTConfig
    except Exception as e:
        raise RuntimeError("Could not import GPT/GPTConfig from model.py or cortex_model.py") from e

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(os.path.dirname(__file__), "ff_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    gpt_conf = GPTConfig(**config)
    model = GPT(gpt_conf)
    model.load_state_dict(checkpoint["model"])  # load weights
    model.to(device)
    model.eval()

    prompt = "The thing about gooses is "
    input_ids = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long, device=device)[None, :]

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=500, temperature=1.0)
    text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
    print(text)

if __name__ == "__main__":
    main()
