import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from RDDPM import RDDIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

B, C, H, W = 1, 1, 64, 64
base_dim = 64
n_visits = 12
T = 1000
TRAIN_STEPS = 5000
total_start = time.perf_counter()

def make_circle_frame(pos_x, radius, H=32, W=32):
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


def make_sequence(n_visits, H=32, W=32, grow_then_shrink=True):
    positions = torch.linspace(0.15, 0.85, n_visits)
    if grow_then_shrink:
        radii = [2 + 5 * np.sin(np.pi * i / (n_visits - 1)) for i in range(n_visits)]
    else:
        radii = [7 - 5 * (i / (n_visits - 1)) for i in range(n_visits)]
    frames = [make_circle_frame(positions[i].item(), radii[i], H, W) for i in range(n_visits)]
    return frames, positions, radii


print("Building synthetic sequences...")
seq_build_start = time.perf_counter()
seq1_frames, seq1_pos, seq1_radii = make_sequence(n_visits, grow_then_shrink=True, H=H, W=W)
seq2_frames, seq2_pos, seq2_radii = make_sequence(n_visits, grow_then_shrink=False, H=H, W=W)
seq_build_elapsed = time.perf_counter() - seq_build_start
print(f"Built sequences in {seq_build_elapsed:.3f}s")
lt_seq = torch.arange(n_visits, dtype=torch.long)

model = RDDIM(
    input_size=(H, W),
    n_channels=C,
    base_dim=base_dim,
    gru_n_layers=6,
    n_res_blocks=3,
    T=T,
    eta=1.0,
    beta_schedule="Linear"
)
print("Model instantiated.")

print(f"Training for {TRAIN_STEPS} steps (alternating sequences)...")
train_start = time.perf_counter()
for step in range(TRAIN_STEPS):
    seq = seq1_frames if step % 2 == 0 else seq2_frames
    loss = model.train_step(seq, lt_seq)
    if step % 50 == 0:
        print(f"  Step {step:3d}: loss = {loss:.4f}")
train_elapsed = time.perf_counter() - train_start
print(f"Training finished in {train_elapsed:.3f}s (avg {train_elapsed/max(1,TRAIN_STEPS):.3f}s/step)")

print("Sampling both sequences...")
with torch.no_grad():
    sample1_start = time.perf_counter()
    gen1 = model.sample(shape=(B, C, H, W), T=T, lt_seq=lt_seq)
    sample1_elapsed = time.perf_counter() - sample1_start
    print(f"Sampling seq1 took {sample1_elapsed:.3f}s")

    sample2_start = time.perf_counter()
    gen2 = model.sample(shape=(B, C, H, W), T=T, lt_seq=lt_seq)
    sample2_elapsed = time.perf_counter() - sample2_start
    print(f"Sampling seq2 took {sample2_elapsed:.3f}s")

fig, axes = plt.subplots(4, n_visits, figsize=(n_visits * 2, 8))
row_labels = ["GT seq1 (grow)", "Gen seq1", "GT seq2 (shrink)", "Gen seq2"]
gt_seqs = [seq1_frames, seq1_frames, seq2_frames, seq2_frames]
gen_seqs = [seq1_frames, gen1, seq2_frames, gen2]

for row_idx, (label, gt, gen) in enumerate(zip(row_labels, gt_seqs, gen_seqs)):
    use = gt if "GT" in label else gen
    for col in range(n_visits):
        img = use[col][0, 0].cpu().numpy()
        axes[row_idx, col].imshow(img, cmap='gray', vmin=-1, vmax=1)
        axes[row_idx, col].axis('off')
        if col == 0:
            axes[row_idx, col].set_ylabel(label, fontsize=7)
        if row_idx == 0:
            axes[row_idx, col].set_title(f"v{col}", fontsize=7)

plt.suptitle("R-DDPM Longitudinal Test: Growing/Shrinking Circle", fontsize=11)
plt.tight_layout()
plt.savefig('functional_test_grid.png', dpi=150)
grid_elapsed = time.perf_counter() - (sample2_start + sample2_elapsed)
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
        gt_img = gt[0, 0].cpu().numpy()
        gen_img = gen[0, 0].cpu().numpy()
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

gif1_start = time.perf_counter()
save_gif(seq1_frames, gen1, 'gen_seq1_grow.gif',    'Seq 1: Circle Grows as it Moves')
gif1_elapsed = time.perf_counter() - gif1_start

gif2_start = time.perf_counter()
save_gif(seq2_frames, gen2, 'gen_seq2_shrink.gif',  'Seq 2: Circle Shrinks as it Moves')
gif2_elapsed = time.perf_counter() - gif2_start

total_elapsed = time.perf_counter() - total_start
print(f"\nGIFs: gif1 {gif1_elapsed:.3f}s, gif2 {gif2_elapsed:.3f}s")
print(f"Total script time: {total_elapsed:.3f}s")