import torch

from config import load_configs
from model.gpt import GPT
from training.optimizer import build_optimizer
from training.precision import PrecisionManager
from training.prm import ProcessRewardModel
from training.scheduler import WarmupCosineLRScheduler
from training.train import train_continuous


def fake_stream(vocab_size: int, n_samples: int = 32, max_len: int = 256):
    for _ in range(n_samples):
        length = torch.randint(32, max_len, (1,)).item()
        seq = torch.randint(0, vocab_size, (length,))
        yield {"input_ids": seq, "labels": seq.clone()}


def main():
    model_cfg, train_cfg, _ = load_configs("config/model.yaml", "config/train.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(
        model,
        lr=train_cfg.lr,
        wd=train_cfg.weight_decay,
        lr_embed_mult=train_cfg.lr_embed_mult,
        lr_head_mult=train_cfg.lr_head_mult,
        lr_moe_mult=train_cfg.lr_moe_mult,
    )
    scheduler = WarmupCosineLRScheduler(
        optimizer,
        warmup_steps=train_cfg.warmup_steps,
        max_steps=train_cfg.max_steps,
        min_lr=train_cfg.min_lr,
    )
    precision = PrecisionManager(enable_fp8=train_cfg.enable_fp8, grad_scale_init=train_cfg.fp8_grad_scale_init)
    prm = ProcessRewardModel(model_cfg.dim).to(device) if train_cfg.enable_prm else None

    metrics = train_continuous(
        model,
        optimizer,
        stream=fake_stream(model_cfg.vocab_size),
        max_tokens_per_step=train_cfg.max_tokens_per_step,
        total_steps=100,
        curriculum_start_len=train_cfg.curriculum_start_len,
        curriculum_end_len=train_cfg.curriculum_end_len,
        scheduler=scheduler,
        prm=prm,
        precision=precision,
        group_size=train_cfg.group_size,
        grad_noise_std=train_cfg.grad_noise_std,
        grad_clip_norm=train_cfg.grad_clip_norm,
        loss_ema_momentum=train_cfg.loss_ema_momentum,
        router_reg_weight=train_cfg.router_reg_weight,
        activation_sparsity_weight=train_cfg.activation_sparsity_weight,
        token_entropy_weight=train_cfg.token_entropy_weight,
    )
    print(f"train steps: {len(metrics)}")


if __name__ == "__main__":
    main()
