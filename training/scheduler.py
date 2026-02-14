import math


def cosine_schedule(step, total):
    return 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))


class CurriculumLengthScheduler:
    def __init__(self, start_len: int, end_len: int, total_steps: int):
        self.start_len = start_len
        self.end_len = end_len
        self.total_steps = max(total_steps, 1)

    def __call__(self, step: int) -> int:
        alpha = min(1.0, step / self.total_steps)
        return int(self.start_len + alpha * (self.end_len - self.start_len))
