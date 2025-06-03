import sys
import os
from pathlib import Path
from dotenv import load_dotenv

working_dir = Path.cwd()
while working_dir.name != 'CausalMTR-BC':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'CausalMTR-BC' not found")
os.chdir(working_dir)

# Load env variables from config/.env 
dotenv_path = Path("configs/.env")
load_dotenv(dotenv_path=dotenv_path)

# GPUDRIVE_PATH = os.environ.get("GPUDRIVE_PATH")
# sys.path.append(os.path.abspath(GPUDRIVE_PATH))

from gpudrive.env.config import EnvConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

DYNAMICS_MODEL = "delta_local" # "delta_local" / "state" / "classic"
DATA_PATH = "data/processed/examples" # Your data path
MAX_NUM_OBJECTS = 6
NUM_ENVS = 2
DEVICE = "cuda:0"

# Configs
env_config = EnvConfig(dynamics_model=DYNAMICS_MODEL)

# Make dataloader
data_loader = SceneDataLoader(
    root="gpudrive/data/processed/examples", # Path to the dataset
    batch_size=NUM_ENVS, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=NUM_ENVS, # Total number of different scenes we want to use
    sample_with_replacement=False, 
    seed=42, 
    shuffle=True,   
)

print("Making the environment")

# Make environment
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=data_loader,
    max_cont_agents=MAX_NUM_OBJECTS, # Maximum number of agents to control per scenario
    device="cuda", 
    action_type="continuous" # "continuous" or "discrete"
)