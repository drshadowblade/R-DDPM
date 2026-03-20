import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from RDDPM import RDDPM, RDDIM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
torch.set_default_device(device)

B, C, H, W = 1, 1, 32, 32
base_dim = 64
n_visits = 8
T = 200

def make_bullet_frame(pos_x, H=32, W=32):
    """White rectangle moving across a black canvas."""
    frame = torch.zeros(1, 1, H, W)
    x = int(pos_x * W)
    frame[0, 0, H//2-2:H//2+2, max(0,x-3):min(W,x+3)] = 1.0
    return frame

positions = torch.linspace(0.1, 0.9, n_visits)
x0_seq = torch.stack([make_bullet_frame(pos) for pos in positions], dim=0)
lt_seq = torch.arange(n_visits, dtype=torch.long)

model = RDDIM(
    input_size=(H, W),
    n_channels=C,
    base_dim=base_dim,
    T=T,
    beta_schedule="Linear"
)

print("Training...")
for step in range(2000):
    loss = model.train_step(x0_seq, lt_seq)
    if step % 50 == 0:
        print(f"  Step {step}: loss={loss:.4f}")

print("Sampling...")
with torch.no_grad():
    outputs = model.sample(shape=(B, C, H, W), T=T, lt_seq=lt_seq)

fig, axes = plt.subplots(2, n_visits, figsize=(n_visits * 2, 4))
for i in range(n_visits):
    axes[0, i].imshow(x0_seq[i][0, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f"GT v{i}")
    axes[0, i].axis('off')

    axes[1, i].imshow(outputs[i][0, 0].cpu(), cmap='gray', vmin=-1, vmax=1)
    axes[1, i].set_title(f"Gen v{i}")
    axes[1, i].axis('off')

axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
axes[1, 0].set_ylabel("Generated", fontsize=9)
plt.tight_layout()
plt.savefig('bullet_comparison.png', dpi=150)
print("Saved bullet_comparison.png")

fig2, ax = plt.subplots()
frames = []
for out in outputs:
    img = out[0, 0].cpu().numpy()
    im = ax.imshow(img, animated=True, cmap='gray', vmin=-1, vmax=1)
    frames.append([im])

ani = animation.ArtistAnimation(fig2, frames, interval=300, blit=True)
ani.save('bullet_generated.gif', writer='pillow')
print("Saved bullet_generated.gif")