import sys
import os
import pathlib
import hydra
from omegaconf import OmegaConf, DictConfig
from train import TrainDP3Workspace
import pdb

# =================
# bash scripts/eval_policy.sh robot_dp3 robot pick_and_place 0 0
DEBUG=False

alg_name='robot_dp3'
task_name='robot'
config_name=alg_name
addition_info='pick_and_place'
seed=0
exp_name=f'{task_name}-{alg_name}-{addition_info}'
run_dir=f"data/outputs/{exp_name}_seed{seed}"

gpu_id=0
# ================

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Define the configuration path and file name
config_path = pathlib.Path(__file__).parent.joinpath('diffusion_policy_3d', 'config')
config_name = 'robot_dp3.yaml'  # Replace with your actual config file name

# Load the YAML file as a DictConfig object
cfg = OmegaConf.load(os.path.join(config_path, config_name))

pdb.set_trace()
# Access configuration values
print("Configuration loaded:")
print(OmegaConf.to_yaml(cfg))


# # Modify the configuration values as needed
# # For example, setting the seed and debug mode
cfg.hydra.run.dir=run_dir
cfg.training.debug = DEBUG  # Replace with your desired seed
cfg.training.seed = seed
cfg.training.device = 'cuda:0'
cfg.exp_name = exp_name

# Now you can pass the modified configuration to your main function
@hydra.main(config_path=config_path)
def main(cfg: DictConfig) -> None:
    print("Configuration loaded and modified:")
    print(OmegaConf.to_yaml(cfg))  # Optional: Print the modified config for debugging
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main(cfg)