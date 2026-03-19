import torch

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(beta, 1e-4, 0.999)