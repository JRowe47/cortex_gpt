import argparse
import torch

from ts_ff_generate import (
    load_model,
    ff_generate,
    decode_bytes,
    get_blocks,
)


def main():
    p = argparse.ArgumentParser(
        description="Sample text from an FF-trained Tiny Shakespeare model",
    )
    p.add_argument("--ckpt", type=str, default="ff_final.pt", help="checkpoint path")
    p.add_argument("--prompt", type=str, default="The thing about gooses is ")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--ff_scan", action="store_true", help="use forward-forward scanning instead of model.generate")
    p.add_argument(
        "--scan_candidates",
        type=int,
        default=64,
        help="candidates per step for FF scan (0 or >= vocab to evaluate all tokens)",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    use_cpu = getattr(args, "cpu", False)
    if not use_cpu and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU")
        use_cpu = True
    device = torch.device("cuda" if not use_cpu else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.ckpt, device)

    idx = torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.long, device=device)[None, :]

    with torch.no_grad():
        if args.ff_scan:
            blocks = get_blocks(model)
            out = ff_generate(
                model,
                blocks,
                idx,
                args.max_new_tokens,
                model.config.block_size,
                device,
                num_candidates=args.scan_candidates,
            )
        else:
            out = model.generate(idx, args.max_new_tokens)

    print(decode_bytes(out[0].tolist()))


if __name__ == "__main__":
    main()
