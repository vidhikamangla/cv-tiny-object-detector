"""Linear warmup then cosine decay, stepped per iteration."""
import math


def lr_at(step, total_steps, base_lr, warmup_steps=500, final_frac=0.02):
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * t))
    return base_lr * (final_frac + (1 - final_frac) * cos)
