# R-DDPM

R-DDPM (Recurrent Denoising Diffusion Probabilistic Model) is a compact research-oriented implementation combining a U-Net / diffusion core with a convolutional GRU to model longitudinal (sequence) data.

This README documents the main classes implemented across the repository and describes the two included functional tests.

Repository layout

- [classes.py](classes.py) — core building blocks for a DDPM (U-Net, ResBlocks, position embeddings) plus a small MNIST example at the bottom of the file.
- [RDDPM.py](RDDPM.py) — recurrent DDPM implementation that extends the UNet with a ConvGRU and long-term visit embedding (sequence-aware diffusion model).
- [convgru/convgru.py](convgru/convgru.py) — convolutional GRU cell and a small multi-layer ConvGRU wrapper used by `RUnet`.
- [functional_test_1.py](functional_test_1.py) — synthetic moving-bullet longitudinal test (32×32).
- [functional_test_2.py](functional_test_2.py) — synthetic growing/shrinking-circle longitudinal test (64×64).

Detailed file & class reference

1) classes.py

- `SinPositionalEmbedding(dim)`
    - Purpose: create sinusoidal positional embeddings for scalar time/visit indices.
    - Call: `forward(t)` where `t` is a 1D tensor of shape `(batch,)`. Returns a `(batch, dim)` embedding (sin/cos concatenation).

- `ResBlock(in_channels, out_channels, time_dim, kernel_size=3, padding=1, n_groups=32)`
    - Residual convolutional block with GroupNorm and time-dependent affine modulation.
    - Call: `forward(x, time_emb)` where `time_emb` is the time embedding produced by `SinPositionalEmbedding` + MLP. The module internally computes `scale, shift = time_mlp(time_emb).chunk(2, dim=-1)` for FiLM-style modulation.

- `ResBlockStack(in_channels, out_channels, time_dim, n_blocks=1)`
    - Stacks multiple `ResBlock`s to increase depth within a UNet stage while preserving spatial resolution.

- `UNet(n_channels, base_dim, n_heads=8, n_res_blocks=1)`
    - U-Net backbone built from `ResBlockStack` stages. Provides:
        - `encode(x, time_emb)` → `(d1, d2, d3)` intermediate feature maps
        - `spatial_attention(x)` → applies a MultiHead attention over spatial tokens
        - `decode(d1, d2, attn_out, time_emb)` → decode to output image
        - `forward(x, time_step)` → main entrypoint used by DDPM.

- `DDPM(n_channels, base_dim, T, ...)` — a non-recurrent diffusion wrapper around `UNet`.
    - Key methods:
        - `q_sample(x0, t)` — add noise to `x0` at time `t` (returns `(xt, noise)`).
        - `p_sample(xt, t)` — single reverse step using the UNet prediction.
        - `sample(shape, T)` — full sampling loop.
        - `train_step(x0)` — single training step that computes MSE between predicted noise and the true noise.
    - Note: `classes.py` contains a `__main__` example that downloads MNIST via `torchvision` and runs a short training loop, saving `samples.png`.

2) RDDPM.py

- `RUnet(UNet)`
    - Extends `UNet` by adding a `ConvGRU` block that processes the high-level spatial features and a separate long-term MLP embedding for visit/sequence labels.
    - Forward signature: `forward(x, dt, lt, h_prev=None)` → `(out, h)` where `h` is the updated recurrent hidden state (list of layer tensors).

- `RDDPM(nn.Module)`
    - Recurrent DDPM wrapper that carries the diffusion schedule and optimizer.
    - Key methods:
        - `q_sample(x0, t)` — produce noisy `xt` and the noise used.
        - `p_sample(xt, dt, lt, h_prev=None)` — single reverse step conditioned on current timestep `dt` and visit label `lt`. Carries and returns recurrent hidden state.
        - `sample(shape, T, lt_seq)` — sample a longitudinal sequence: for each visit label in `lt_seq` the model starts from noise, denoises through timesteps, and carries the GRU state between visits. Returns a list of generated frames (one per visit).
        - `train_step(x0_seq, lt_seq)` — trains on a sequence of frames (list or tensor sequence) and returns the total MSE loss; detaches recurrent states to prevent gradient accumulation across sequence boundaries.
        - `RDDIM` subclass: For DDIM-style sampling, `RDDIM` is provided (it subclasses `RDDPM`) and implements `p_sample_ddim(xt, dt, lt, h_prev=None)` and `sample(shape, T, lt_seq, device=None)`. The constructor accepts an `eta` parameter (default `0.0`) for deterministic (`eta=0.0`) or stochastic (`eta>0`) sampling. `RDDIM.sample` keeps the same `T` / `lt_seq` signature as `RDDPM.sample` for compatibility.

3) convgru/convgru.py

- `ConvGRUCell(input_size, hidden_size, kernel_size)`
    - A convolutional GRU cell using `Conv2d` gates. `forward(input_, prev_state)` returns `new_state` with the same spatial dimensions.

- `ConvGRU(input_size, hidden_sizes, kernel_sizes, n_layers)`
    - Stacks multiple `ConvGRUCell`s into a multi-layer recurrent block. `forward(x, hidden=None)` returns a list `upd_hidden` of per-layer hidden states (shape: `[layer, batch, channels, H, W]` preserved as a Python list of 4D tensors).

Functional tests (what they do)

- `functional_test_1.py` — Moving bullet test
    - Purpose: minimal longitudinal sanity check of the recurrent pathway.
    - Synthetic data: short sequence (default n_visits=8) of a white rectangular "bullet" sweeping horizontally across a black 32×32 canvas.
    - Model: instantiates `RDDPM` with `T=200`, `base_dim=64` and trains for a small number of steps (default loop runs 2000 steps in the script).
    - Outputs saved to working directory:
        - `bullet_comparison.png` — grid comparing GT vs generated frames.
        - `bullet_generated.gif` — animated generated sequence.
    - Command: `python functional_test_1.py` (creates outputs in the current directory).

- `functional_test_2.py` — Growing / Shrinking circle test (longer sequence)
    - Purpose: exercise the recurrent hidden state across a longer sequence and compare generation for two alternating dynamics (growing then shrinking radius).
    - Synthetic data: two sequences of moving circles (default n_visits=12) on 64×64 frames.
    - Model: `RDDPM` with longer GRU stack (`gru_n_layers=6`) and deeper ResBlock stacks (`n_res_blocks=3`). Default script trains for `TRAIN_STEPS=5000`.
    - Outputs saved:
        - `functional_test_grid.png` — 4×N grid showing GT and generated sequences.
        - `gen_seq1_grow.gif` and `gen_seq2_shrink.gif` — animated comparisons.
    - Command: `python functional_test_2.py`.
    - Note: `functional_test_2.py` has been instrumented to print per-section timings when run (sequence build, training, sampling, image/GIF saving, and total runtime).

Quick start

Install minimal dependencies (CPU or GPU as appropriate):

```bash
python -m pip install "torch" torchvision matplotlib pillow
```

Run the functional tests (they use synthetic data and do not require MNIST files):

```bash
python functional_test_1.py
python functional_test_2.py
```