from classes import UNet, SinPositionalEmbedding
from convgru.convgru import ConvGRU
import torch
from torch import nn
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class RUnet(UNet):
    def __init__(self, input_size, n_channels, base_dim, gru_n_layers = 3, n_heads=8, n_res_blocks=1):
        super(RUnet, self).__init__(n_channels, base_dim, n_heads, n_res_blocks)   
        self.gru_n_layers = gru_n_layers
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
        self.film_embeds = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 2)
        )
        self.film_memory = nn.Conv2d(base_dim* 4, base_dim*8, 1)

    def get_hidden_state(self, x, t, h_prev=None):
        if not torch.is_tensor(t):
            t = torch.full((x.size(0),), t, device=x.device, dtype=torch.float32)
        else:
            t = t.to(x.device).float()
        time_emb = self.time_mlp(t)
        _, _, d3 = self.encode(x, time_emb)
        attn_out = self.spatial_attention(d3)
        upd_hidden = self.gru(attn_out, h_prev)
        return upd_hidden

    def forward(self, x, dt, lt, h_prev = None):
        device = x.device
        dt = dt.to(device)
        lt = lt.to(device)
        time_emb = self.time_mlp(dt.float())
        long_emb = self.long_mlp(lt.float())
        scale, shift = self.film_embeds(long_emb).chunk(2, 1)
        emb = time_emb * (1 + scale) + shift
        d1, d2, d3 = self.encode(x, emb)
        attn_out = self.spatial_attention(d3)
        upd_hidden = self.gru(attn_out, h_prev)
        h = upd_hidden
        h_last = h[-1] if isinstance(h, list) else h
        scale, shift = self.film_memory(h_last).chunk(2, 1)
        drop = attn_out * (1 + scale) + shift
        out = self.decode(d1, d2, drop, emb)
        return out, h
    
class RDDPM(nn.Module):
    def __init__(self, input_size, n_channels, base_dim, gru_n_layers = 3, n_heads=8, n_res_blocks=1, beta_start =1e-4, beta_end=0.02, T=1000, beta_schedule='cosine'):
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

    def q_sample(self, x0, t, device=None):
        if device is None:
            device = x0.device
        noise = torch.randn_like(x0, device=device)
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        t = t.to(device)
        return sqrt_alpha_bar[t].view(-1, 1, 1, 1) * x0 + sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1) * noise, noise
    
    def p_sample(self, xt, dt, lt, h_prev=None, device=None):
        if device is None:
            device = xt.device
        if not isinstance(dt, torch.Tensor):
            dt = torch.full((xt.size(0),), dt, dtype=torch.long, device=device)
        else:
            dt = dt.to(device)
        if not isinstance(lt, torch.Tensor):
            lt = torch.full((xt.size(0),), lt, dtype=torch.long, device=device)
        else:
            lt = lt.to(device)
        pred_noise, h = self.model(xt, dt, lt, h_prev)
        # Move tensors to the correct device before indexing
        alpha = self.alpha.to(device)[dt]
        beta = self.beta.to(device)[dt]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)[dt]
        sqrt_beta = self.sqrt_beta.to(device)[dt]
        mean = (
            1 / torch.sqrt(alpha).view(-1, 1, 1, 1) *
            (xt - (beta / sqrt_one_minus_alpha_bar).view(-1, 1, 1, 1) * pred_noise)
        )
        noise = torch.randn_like(xt, device=device)
        mask = (dt > 0).float().view(-1, 1, 1, 1)
        return mean + mask * sqrt_beta.view(-1, 1, 1, 1) * noise, h
    
    def sample(self, shape, T, lt_seq, pre_images=None, pre_times=None, device=None):
        if device is None:
            device = next(self.model.parameters()).device
        h = None
        if pre_images is not None:
            if pre_times is None:
                pre_times = list(range(len(pre_images)))
            for img, t in zip(pre_images, pre_times):
                h = self.model.get_hidden_state(img.to(device), t, h_prev=h)
        outputs = []
        for lt in lt_seq:
            xt = torch.randn(shape, device=device)
            lt = lt.expand(shape[0]).to(device)
            for t in reversed(range(T)):
                xt, h_new = self.p_sample(xt, t, lt, h_prev=h, device=device)
            h = h_new
            # Move each output to CPU immediately to free CUDA memory
            outputs.append(xt.detach().cpu())
            del xt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs
    def train_step(self, x0_seq, lt_seq, device=None):
        h = None
        total_loss = 0
        if device is None:
            device = next(self.model.parameters()).device
        for x0, lt in zip(x0_seq, lt_seq):
            x0 = x0.to(device)
            dt = torch.randint(0, self.T, (x0.size(0),), device=device)
            lt = lt.expand(x0.size(0)).to(device)
            xt, noise = self.q_sample(x0, dt, device=device)
            if h is not None:
                if isinstance(h, list):
                    h = [hi.to(device) for hi in h]
                else:
                    h = h.to(device)
            pred_noise, h = self.model(xt, dt, lt, h)
            if h is not None:
                if isinstance(h, list):
                    h = [hi.detach() for hi in h]
                else:
                    h = h.detach()
            total_loss += nn.MSELoss()(pred_noise, noise)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()


class RDDIM(RDDPM):
    def __init__(self, input_size, n_channels, base_dim, gru_n_layers = 4, n_heads=8, n_res_blocks=1, beta_start =1e-4, beta_end=0.02, T=1000, beta_schedule='cosine', eta=0.0):
        super().__init__(input_size, n_channels, base_dim, gru_n_layers, n_heads, n_res_blocks, beta_start, beta_end, T, beta_schedule)
        self.eta = eta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def p_sample(self, xt, dt, lt, h_prev=None, device=None):

        if device is None:
            device = xt.device
        if not isinstance(dt, torch.Tensor):
            dt = torch.full((xt.size(0),), dt, dtype=torch.long)
        else:
            dt = dt.long()
        dt = dt.to(self.sqrt_alpha_bar.device)  # Ensure dt is on the same device as sqrt_alpha_bar
        if not isinstance(lt, torch.Tensor):
            lt = torch.full((xt.size(0),), lt, dtype=torch.long, device=device)
        else:
            lt = lt.long().to(device)

        pred_noise, h = self.model(xt, dt, lt, h_prev)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[dt].to(device).view(-1,1,1,1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[dt].to(device).view(-1,1,1,1)

        x0_pred = (xt - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t + 1e-8)

        prev_idx = (dt - 1).clamp(min=0)
        sqrt_alpha_bar_prev = self.sqrt_alpha_bar[prev_idx].to(device).view(-1,1,1,1)
        sqrt_one_minus_alpha_bar_prev = self.sqrt_one_minus_alpha_bar[prev_idx].to(device).view(-1,1,1,1)
        alpha_bar_t = self.alpha_bar[dt].to(device).view(-1,1,1,1)
        alpha_bar_prev = self.alpha_bar[prev_idx].to(device).view(-1,1,1,1)

        eta = float(self.eta)
        if eta == 0.0:
            x_prev_det = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * pred_noise
            mask = (dt > 0).float().view(-1,1,1,1).to(device)
            x_prev_det = x_prev_det.to(device)
            x0_pred = x0_pred.to(device)
            x_prev = mask * x_prev_det + (1 - mask) * x0_pred
            return x_prev, h
        else:
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(torch.clamp(1 - alpha_bar_t / (alpha_bar_prev + 1e-8), min=0.0))
            noise = torch.randn_like(xt, device=device)
            nonzero_term = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * pred_noise
            x_prev = sqrt_alpha_bar_prev * x0_pred + nonzero_term + sigma * noise
            mask = (dt > 0).float().view(-1,1,1,1)
            x_prev = mask * x_prev + (1 - mask) * x0_pred
            return x_prev, h

    def sample(self, shape, T, lt_seq, pre_images=None, device=None, pre_times=None):
        if device is None:
            device = next(self.model.parameters()).device
        h = None
        if pre_images is not None:
            if pre_times is None:
                pre_times = list(range(len(pre_images)))
            for img, t in zip(pre_images, pre_times):
                h = self.model.get_hidden_state(img.to(device), t, h_prev=h)
        outputs = []
        for lt in lt_seq:
            xt = torch.randn(shape, device=device)
            lt = lt.expand(shape[0]).to(device)
            for t in reversed(range(T)):
                xt, h_new = self.p_sample(xt, t, lt, h_prev=h, device=device)
            h = h_new
            outputs.append(xt.detach().cpu())
            del xt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs