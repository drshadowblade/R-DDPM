import torch.nn as nn
import torch

class SinPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinPositionalEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, kernel_size=3, padding=1, n_groups=32):
        super(ResBlock, self).__init__()
        gn1_groups = n_groups if in_channels % n_groups == 0 else 1
        gn2_groups = n_groups if out_channels % n_groups == 0 else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(gn1_groups, in_channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(gn2_groups, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels * 2)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.GroupNorm(gn2_groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.gn1(x)
        h = self.silu(h)
        h = self.conv1(h)
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h =self.gn2(h)
        h = h * (1 + scale) + shift
        h = self.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)
    
class ResBlockStack(nn.Module):
    """
    Stack of ResBlocks that forwards the time embedding to each block.
    Used to increase depth within each UNet stage without changing spatial scales.
    """
    def __init__(self, in_channels, out_channels, time_dim, n_blocks=1):
        super(ResBlockStack, self).__init__()
        self.blocks = nn.ModuleList()
        # first block may change channel dimension
        self.blocks.append(ResBlock(in_channels, out_channels, time_dim))
        for _ in range(n_blocks - 1):
            self.blocks.append(ResBlock(out_channels, out_channels, time_dim))

    def forward(self, x, time_emb):
        h = x
        for b in self.blocks:
            h = b(h, time_emb)
        return h
class UNet(nn.Module):
    def __init__(self, n_channels, base_dim, n_heads=8, n_res_blocks=1):
        super(UNet, self).__init__()
        self.base_dim = base_dim
        assert base_dim % n_heads == 0, "base_dim must be divisible by n_heads"
        self.time_mlp = nn.Sequential(
            SinPositionalEmbedding(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim)
        )
        # allow stacking multiple ResBlocks per stage to increase depth
        self.down1 = ResBlockStack(n_channels, base_dim, base_dim, n_blocks=n_res_blocks)
        self.down2 = ResBlockStack(base_dim, base_dim * 2, base_dim, n_blocks=n_res_blocks)
        self.down3 = ResBlockStack(base_dim * 2, base_dim * 4, base_dim, n_blocks=n_res_blocks)
        self.up1 = ResBlockStack(base_dim * 4 + base_dim * 2, base_dim * 2, base_dim, n_blocks=n_res_blocks)
        self.up2 = ResBlockStack(base_dim * 2 + base_dim, base_dim, base_dim, n_blocks=n_res_blocks)
        self.up3 = ResBlockStack(base_dim, base_dim, base_dim, n_blocks=n_res_blocks)
        self.pool = nn.MaxPool2d(2)
        self.upsample1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 4, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(base_dim * 2, base_dim * 2, kernel_size=2, stride=2)
        self.attention = nn.MultiheadAttention(base_dim * 4, num_heads=n_heads)
        self.attention_layernorm = nn.GroupNorm(1, base_dim * 4)
        self.out = nn.Conv2d(base_dim, n_channels, kernel_size=1)

    def encode(self, x, time_emb):
        d1 = self.down1(x, time_emb)
        d2 = self.down2(self.pool(d1), time_emb)
        d3 = self.down3(self.pool(d2), time_emb)
        return d1, d2, d3
    
    def spatial_attention(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(2, 0, 1)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(1, 2, 0).view(b, c, h, w)
        return self.attention_layernorm(attn_out + x)
    
    def decode(self, d1, d2, attn_out, time_emb):
        attn_out = self.upsample1(attn_out)
        u1 = self.up1(torch.cat([attn_out, d2], dim=1), time_emb)
        u1 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u1, d1], dim=1), time_emb)
        u3 = self.up3(u2, time_emb)
        out = self.out(u3)
        return out

    def forward(self, x, time_step):
        time_emb = self.time_mlp(time_step.float())
        d1, d2, d3 = self.encode(x, time_emb)
        attn_out = self.spatial_attention(d3)
        return self.decode(d1, d2, attn_out, time_emb)

class DDPM(nn.Module):
    def __init__(self, n_channels, base_dim, T, beta_start=1e-4, beta_end=0.02, device=None):
        super(DDPM, self).__init__()
        self.model = UNet(n_channels, base_dim)
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T, )
        self.alpha = 1.0 - self.beta
        self.sqrt_alpha_bar = torch.sqrt(torch.cumprod(self.alpha, dim=0))
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - torch.cumprod(self.alpha, dim=0))
        self.sqrt_beta = torch.sqrt(self.beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        return self.sqrt_alpha_bar[t].view(-1, 1, 1, 1) * x0 + self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1) * noise, noise
    def p_sample(self, xt, t):
        xt = xt
        if not isinstance(t, torch.Tensor):
            t = torch.full((xt.size(0),), t, dtype=torch.long)
        else:
            t = t
        pred_noise = self.model(xt, t)
        mean = (
            1 / torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1) *
            (xt - (self.beta[t] / self.sqrt_one_minus_alpha_bar[t]).view(-1, 1, 1, 1) * pred_noise)
        )
        noise = torch.randn_like(xt)
        mask = (t > 0).float().view(-1, 1, 1, 1)
        return mean + mask * self.sqrt_beta[t].view(-1, 1, 1, 1) * noise
    def sample(self, shape, T):
        xt = torch.randn(shape, )
        for t in reversed(range(T)):
            xt = self.p_sample(xt, t)
        return xt
    def train_step(self, x0):
        x0 = x0
        t = torch.randint(0, self.T, (x0.size(0),), )
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.model(xt, t)
        loss = nn.MSELoss()(pred_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    transform = Compose([
    ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(root='data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = DDPM(n_channels=1, base_dim=64, T=500)
    for epoch in range(5):
        for x, _ in dataloader:
            loss = model.train_step(x)
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with torch.no_grad():
        samples = model.sample((16, 1, 28, 28), T=500)
        samples = (samples + 1) / 2
    samples = samples.detach().cpu()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, 0], cmap='gray')
        ax.axis('off')
    plt.savefig('samples.png')