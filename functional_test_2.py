import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from RDDPM import RDDPM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

B, C, H, W = 1, 1, 64, 64
base_dim = 64
n_visits = 12
T = 500
TRAIN_STEPS = 4000
N_INPUT = 4
total_start = time.perf_counter()

def make_circle_frame(pos_x, radius, H=64, W=64):
    frame = torch.zeros(1, 1, H, W)
    cx = int(pos_x * W)
    cy = H // 2
    r = max(1, int(radius))
    for i in range(H):
        for j in range(W):
            if (i - cy) ** 2 + (j - cx) ** 2 <= r ** 2:
                frame[0, 0, i, j] = 1.0
    frame = frame * 2 - 1
    return frame


def make_sequence(n_visits, H=64, W=64, grow_then_shrink=True):
    positions = torch.linspace(0.15, 0.85, n_visits)
    if grow_then_shrink:
        radii = [2 + 5 * np.sin(np.pi * i / (n_visits - 1)) for i in range(n_visits)]
    else:
        radii = [7 - 5 * (i / (n_visits - 1)) for i in range(n_visits)]
    frames = [make_circle_frame(positions[i].item(), radii[i], H, W) for i in range(n_visits)]
    return frames, positions, radii


print("Building synthetic sequences...")
seq_build_start = time.perf_counter()
seq_frames, seq_pos, seq_radii = make_sequence(n_visits, grow_then_shrink=True, H=H, W=W)
seq_build_elapsed = time.perf_counter() - seq_build_start
print(f"Built sequences in {seq_build_elapsed:.3f}s")
lt_seq = torch.arange(n_visits, dtype=torch.long)

model = RDDPM(
    input_size=(H, W),
    n_channels=C,
    base_dim=base_dim,
    gru_n_layers=6,
    n_res_blocks=3,
    T=T,
    beta_schedule="Linear"
)
model = model.to(device)
print("Model instantiated.")

print(f"Training for {TRAIN_STEPS} steps on the single grow-then-shrink sequence...")
train_start = time.perf_counter()
for step in range(TRAIN_STEPS):
    loss = model.train_step(seq_frames, lt_seq, device=device)
    if step % 50 == 0:
        print(f"  Step {step:3d}: loss = {loss:.4f}")
train_elapsed = time.perf_counter() - train_start
print(f"Training finished in {train_elapsed:.3f}s (avg {train_elapsed/max(1,TRAIN_STEPS):.3f}s/step)")

print("Sampling future visits with warm-start context...")
with torch.no_grad():
    sample_start = time.perf_counter()
    pre_images = [f.to(device) for f in seq_frames[:N_INPUT]]
    pre_times = list(range(N_INPUT))
    lt_future = torch.arange(N_INPUT, n_visits, dtype=torch.long)
    gen_future = model.sample(
        shape=(B, C, H, W),
        T=T,
        lt_seq=lt_future,
        pre_images=pre_images,
        pre_times=pre_times,
        device=device,
    )
    gen = [f.detach().cpu() for f in seq_frames[:N_INPUT]] + gen_future
    sample_elapsed = time.perf_counter() - sample_start
    print(f"Sampling sequence took {sample_elapsed:.3f}s")
    gen_stats = [
        (i, float(g.min()), float(g.max()), float(g.mean()))
        for i, g in enumerate(gen)
    ]
    print("Generated frame stats (idx, min, max, mean):")
    for item in gen_stats:
        print(f"  {item[0]:2d}: min={item[1]: .3f}, max={item[2]: .3f}, mean={item[3]: .3f}")

fig, axes = plt.subplots(2, n_visits, figsize=(n_visits * 2, 4))
row_labels = ["GT grow-then-shrink", "Gen grow-then-shrink"]
rows = [seq_frames, gen]

for row_idx, (label, frames) in enumerate(zip(row_labels, rows)):
    for col in range(n_visits):
        img = torch.clamp(frames[col][0, 0].detach().cpu().float(), -1, 1).numpy()
        axes[row_idx, col].imshow(img, cmap='gray', vmin=-1, vmax=1)
        axes[row_idx, col].axis('off')
        if col == 0:
            axes[row_idx, col].set_ylabel(label, fontsize=7)
        if row_idx == 0:
            axes[row_idx, col].set_title(f"v{col}", fontsize=7)

plt.suptitle("R-DDPM Longitudinal Test: Single Growing/Shrinking Circle", fontsize=11)
plt.tight_layout()
plt.savefig('functional_test_grid.png', dpi=150)
grid_elapsed = time.perf_counter() - (sample_start + sample_elapsed)
print("\nSaved functional_test_grid.png")
print(f"Grid generation + save took {grid_elapsed:.3f}s (approx)")

def save_gif(gt_frames, gen_frames, filename, title):
    gif_start = time.perf_counter()
    fig2, axes2 = plt.subplots(1, 2, figsize=(5, 3))
    axes2[0].set_title("Ground Truth", fontsize=9)
    axes2[1].set_title("Generated", fontsize=9)
    fig2.suptitle(title, fontsize=9)

    anim_frames = []
    for gt, gen in zip(gt_frames, gen_frames):
        gt_img = torch.clamp(gt[0, 0].detach().cpu().float(), -1, 1).numpy()
        gen_img = torch.clamp(gen[0, 0].detach().cpu().float(), -1, 1).numpy()
        im1 = axes2[0].imshow(gt_img, animated=True, cmap='gray', vmin=-1, vmax=1)
        im2 = axes2[1].imshow(gen_img, animated=True, cmap='gray', vmin=-1, vmax=1)
        axes2[0].axis('off')
        axes2[1].axis('off')
        anim_frames.append([im1, im2])

    ani = animation.ArtistAnimation(fig2, anim_frames, interval=400, blit=True)
    ani.save(filename, writer='pillow')
    plt.close(fig2)
    gif_elapsed = time.perf_counter() - gif_start
    print(f"Saved {filename} in {gif_elapsed:.3f}s")

gif_start = time.perf_counter()
save_gif(seq_frames, gen, 'gen_grow_then_shrink.gif', 'Single Sequence: Circle Grows Then Shrinks')
gif_elapsed = time.perf_counter() - gif_start

total_elapsed = time.perf_counter() - total_start
print(f"\nGIF: {gif_elapsed:.3f}s")
print(f"Total script time: {total_elapsed:.3f}s")