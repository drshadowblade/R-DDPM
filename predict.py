from RDDPM import RDDIM
import torch
from utils import sample_and_save

LT_SEQ = torch.tensor(range(2))
sequences = ['FLAIR', 'POST', 'PRE', 'T2']
BASE_DIM = 128
GRU_LAYERS = 1
RES_BLOCKS = 2
T = 1000
ETA = 0.0
BETA_SCHEDULE = 'linear'
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = "cuda"
H = W = 128
ckpt = torch.load('checkpoints/latest.pth')
model = RDDIM(
    input_size=(H, W),
    n_channels=len(sequences),
    base_dim=BASE_DIM,
    gru_n_layers=GRU_LAYERS,
    n_res_blocks=RES_BLOCKS,
    T=T,
    eta=ETA,
    beta_schedule=BETA_SCHEDULE,
    )
model.load_state_dict(ckpt['model_state'])
model = torch.compile(model).to(DEVICE)
model.eval()

gt_frames = []

with torch.no_grad():
    for i in range(1):
        sample_and_save(model, gt_frames, LT_SEQ, (H, W), sequences, "output", T)

