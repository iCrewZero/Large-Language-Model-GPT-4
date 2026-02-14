import torch

from config import load_configs
from model.gpt import GPT
from training.optimizer import build_optimizer
from training.prm import ProcessRewardModel
from training.train import train_continuous


def fake_stream(vocab_size: int, n_samples: int = 32, max_len: int = 256):
    for _ in range(n_samples):
        length = torch.randint(32, max_len, (1,)).item()
        seq = torch.randint(0, vocab_size, (length,))
        yield {"input_ids": seq[:-1], "labels": seq}


def main():
    model_cfg, train_cfg, _ = load_configs("config/model.yaml", "config/train.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, lr=train_cfg.lr, wd=train_cfg.weight_decay)
    prm = ProcessRewardModel(model_cfg.dim).to(device) if train_cfg.enable_prm else None

    metrics = train_continuous(
        model,
        optimizer,
        stream=fake_stream(model_cfg.vocab_size),
        max_tokens_per_step=train_cfg.max_tokens_per_step,
        total_steps=100,
        curriculum_start_len=train_cfg.curriculum_start_len,
        curriculum_end_len=train_cfg.curriculum_end_len,
        prm=prm,
        group_size=train_cfg.group_size,
        grad_noise_std=train_cfg.grad_noise_std,
    )
    print(f"train steps: {len(metrics)}")


if __name__ == "__main__":
    main()
