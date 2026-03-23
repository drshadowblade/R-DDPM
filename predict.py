import torch
from pathlib import Path
import pandas as pd
from utils import load_session_tensor, sample_and_save
from RDDPM import RDDPM, RDDIM

# ==== CONSTANTS (replace args) ====
CHECKPOINT = 'checkpoints/latest.pth'  # Path to model checkpoint
DATA_PATH = 'training_data/'  # Path to training data
PATIENT_ID = 'P001'  # Patient ID
N_INPUT = 3  # Number of initial images to use
OUT_DIR = 'output_predict/'  # Output directory
T_STEPS = 1000  # Diffusion steps
# ==== END CONSTANTS ====

# Load session table
csv_path = Path(DATA_PATH) / 'session_table.csv'
session_table = pd.read_csv(csv_path)
session_table = session_table[session_table['FLAIR'].str.contains(PATIENT_ID)]
session_table = session_table.sort_values('session_date').reset_index(drop=True)

data_root = Path(DATA_PATH)

ckpt = torch.load(CHECKPOINT, map_location='cpu')
model = RDDIM(
    input_size=ckpt['hyperparams']['size'],
    n_channels=len(ckpt['hyperparams']['sequences']),
    base_dim=ckpt['hyperparams']['base_dim'],
    gru_n_layers=ckpt['hyperparams']['gru_layers'],
    n_res_blocks=ckpt['hyperparams']['res_blocks'],
    T=ckpt['hyperparams']['T'],
    eta=ckpt['hyperparams']['eta'],
    beta_schedule=ckpt['hyperparams']['beta_schedule']
)
frames = [
    load_session_tensor(row, data_root, ckpt['hyperparams']['sequences'], ckpt['hyperparams']['size'])
    for _, row in session_table.iterrows()
]

n_visits = len(frames)

# Select initial images for hidden state
n_input = min(N_INPUT, n_visits)
pre_images = [frames[i] for i in range(n_input)]

model.load_state_dict(ckpt['model_state'])
model.eval()

# Predict future images using sample_and_save with pre_images
to_predict = list(range(n_input, n_visits))
lt_seq = torch.arange(n_input, n_visits, dtype=torch.long)
pre_times = list(range(n_input))  # Use the time indices of the initial images
sample_and_save(
    model=model,
    gt_frames=None,
    lt_seq=lt_seq,
    img_size=ckpt['hyperparams']['size'],
    sequences=ckpt['hyperparams']['sequences'],
    out_dir=OUT_DIR,
    T=T_STEPS,
    compare=False,
    pre_images=pre_images,
    pre_times=pre_times
)
