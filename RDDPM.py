from classes import UNet, SinPositionalEmbedding
from convgru.convgru import ConvGRU
import torch
from torch import nn
from utils import cosine_beta_schedule

class RUnet(UNet):
    def __init__(self, input_size, n_channels, base_dim, gru_n_layers = 2, n_heads=8, n_res_blocks=1):
        super(RUnet, self).__init__(n_channels, base_dim, n_heads, n_res_blocks)   
        self.gru = ConvGRU(
            input_size=base_dim * 4,
            hidden_sizes=base_dim * 4,
            kernel_sizes=3,
            n_layers=gru_n_layers,
        )
        self.long_mlp = nn.Sequential(
            SinPositionalEmbedding(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim)
        )
        self.emb_proj = nn.Linear(base_dim * 2, base_dim)
    def forward(self, x, dt, lt, h_prev = None):
        time_emb = self.time_mlp(dt.float())
        long_emb = self.long_mlp(lt.float())
        emb = torch.cat([time_emb, long_emb], dim=-1)
        emb = self.emb_proj(emb)
        d1, d2, d3 = self.encode(x, emb)
        attn_out = self.spatial_attention(d3)
        upd_hidden = self.gru(attn_out, h_prev)
        h = upd_hidden
        h_last = h[-1] if isinstance(h, list) else h
        drop = attn_out + h_last
        out = self.decode(d1, d2, drop, emb)
        return out, h
    
class RDDPM(nn.Module):
    def __init__(self, input_size, n_channels, base_dim, gru_n_layers = 4, n_heads=8, n_res_blocks=1, beta_start =1e-4, beta_end=0.02, T=1000, beta_schedule='cosine'):
        super().__init__()
        self.T = T
        self.model = RUnet(input_size, n_channels, base_dim, gru_n_layers, n_heads, n_res_blocks)
        if beta_schedule == 'cosine':
            self.beta = cosine_beta_schedule(T, s=0.008)
        else:
            self.beta =  torch.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.sqrt_alpha_bar = torch.sqrt(torch.cumprod(self.alpha, dim=0))
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - torch.cumprod(self.alpha, dim=0))
        self.sqrt_beta = torch.sqrt(self.beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bar
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar
        return sqrt_alpha_bar[t].view(-1, 1, 1, 1) * x0 + sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1) * noise, noise
    
    def p_sample(self, xt, dt, lt, h_prev=None):
        if not isinstance(dt, torch.Tensor):
            dt = torch.full((xt.size(0),), dt, dtype=torch.long)
        else:
            dt = dt
        if not isinstance(lt, torch.Tensor):
            lt = torch.full((xt.size(0),), lt, dtype=torch.long)
        else:
            lt = lt
        pred_noise, h = self.model(xt, dt, lt, h_prev)
        mean = (
            1 / torch.sqrt(self.alpha[dt]).view(-1, 1, 1, 1) *
            (xt - (self.beta[dt] / self.sqrt_one_minus_alpha_bar[dt]).view(-1, 1, 1, 1) * pred_noise)
        )
        noise = torch.randn_like(xt)
        mask = (dt > 0).float().view(-1, 1, 1, 1)
        return mean + mask * self.sqrt_beta[dt].view(-1, 1, 1, 1) * noise, h
    
    def sample(self, shape, T, lt_seq):
        h = None
        outputs = []
        for lt in lt_seq:
            xt = torch.randn(shape)
            lt = lt.expand(shape[0])
            for t in reversed(range(T)):
                xt, h_new = self.p_sample(xt, t, lt, h_prev=h)
            h = h_new
            outputs.append(xt)
        return outputs
    def train_step(self, x0_seq, lt_seq):
        h = None
        total_loss = 0
        for x0, lt in zip(x0_seq, lt_seq):
            x0 = x0
            dt = torch.randint(0, self.T, (x0.size(0),))
            lt = lt.expand(x0.size(0))
            xt, noise = self.q_sample(x0, dt)
            pred_noise, h = self.model(xt, dt, lt, h)
            if h is not None:
                if isinstance(h, list):
                    h = [hi.detach() for hi in h]
                else:
                    h = h.detach()
            total_loss += nn.MSELoss()(pred_noise, noise)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()