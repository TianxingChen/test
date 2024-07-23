import os
import pathlib
import sys
import hydra
from omegaconf import OmegaConf
from train import TrainDP3Workspace
import pdb

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
# pythn .py --nconfig
def main(alg_name, task_name, addition_info, seed, gpu_id, wandb_mode, save_ckpt):
    # Create the experiment name and run directory
    exp_name = f"{task_name}-{alg_name}-{addition_info}"
    run_dir = f"data/outputs/{exp_name}_seed{seed}"

    # Set the environment variables
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Initialize Hydra and load the configuration
    # config_path = os.path.join('diffusion_policy_3d', 'config')
    # hydra.initialize(config_path=config_path)
    # config_name = alg_name
    # cfg = hydra.compose(config_name=f"{config_name}.yaml")
    config_path = pathlib.Path(__file__).parent.joinpath('diffusion_policy_3d', 'config')
    config_name = 'robot_dp3.yaml'  # Replace with your actual config file name

    # Load the YAML file as a DictConfig object
    cfg = OmegaConf.load(os.path.join(config_path, config_name))
    pdb.set_trace()

    # Update the loaded configuration with hardcoded values
    cfg.task = task_name
    cfg.hydra.run.dir = run_dir
    cfg.training.debug = False  # Assuming DEBUG is False as in the shell script
    cfg.training.seed = seed
    cfg.training.device = f"cuda:{gpu_id}"
    cfg.exp_name = exp_name
    cfg.logging.mode = wandb_mode
    cfg.checkpoint.save_ckpt = save_ckpt

    # Initialize the workspace with the configuration and run evaluation
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    # Hardcode the parameters as you would in the shell script
    alg_name = "robot_dp3"
    task_name = "robot"
    addition_info = "pick_and_place"
    seed = 0
    gpu_id = 0
    wandb_mode = "disabled"  # Set the logging mode as per your requirement
    save_ckpt = True  # Set whether to save checkpoints or not

    main(alg_name, task_name, addition_info, seed, gpu_id, wandb_mode, save_ckpt)