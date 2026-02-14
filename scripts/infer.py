import argparse

import torch

from config import load_configs
from inference.engine import Engine
from model.gpt import GPT


class DummyTokenizer:
    def encode(self, text: str):
        return [ord(c) % 255 for c in text]

    def decode(self, ids):
        return "".join(chr(i % 255) for i in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--mode", type=str, default="default", choices=["default", "mcts", "beam", "refine"])
    args = parser.parse_args()

    model_cfg, _, infer_cfg = load_configs("config/model.yaml", "config/train.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(model_cfg).to(device).eval()

    tok = DummyTokenizer()
    engine = Engine(model, tok, cfg=infer_cfg.__dict__)
    out = engine.generate([args.prompt], max_new=args.max_new, reasoning_mode=None if args.mode == "default" else args.mode)
    print(out[0])


if __name__ == "__main__":
    main()
