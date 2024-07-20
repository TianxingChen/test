# 3D-Diffusion-Policy-lite
A cleaned DP3 code base for 3D Robotic Manipulation tasks. Credit to [Yanjie Ze](https://github.com/yanjieze).
Need [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) and [OWL-ViT](https://huggingface.co/docs/transformers/en/model_doc/owlvit) also.

You should modify: [robot.yaml](./3D-Diffusion-Policy/diffusion_policy_3d/config/task/robot.yaml), [robot_dp3.yaml](./3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3.yaml), [robot_runner.py (get_action method)](/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py), maybe [robot_dataset.py](/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py) also (if your dataset's format is different from the default one: action, agent_pos, point_cloud).

0. Move to `$YOUR_REPO_PATH/3D-Diffusion-Policy/` first.

1. Run the following script in terminal to **train** your policy:
```
bash scripts/train_policy.sh robot_dp3 robot {task-name} 0 0
```
2. Run the following script in terminal to **evaluate** your policy:
```
bash scripts/eval_policy.sh robot_dp3 robot {task-name} 0 0
```
3. Run the following script in terminal to **run your policy on real robot**:
```
bash scripts/run_policy.sh robot_dp3 robot {task-name} 0 0
```


# 📊 Benchmark of DP3

**Simulation environments.** We provide dexterous manipulation environments and expert policies for `Adroit` and `DexArt` in this codebase. the 3D modality generation (depths and point clouds) has been incorporated for these environments.


**Real-world robot data** is also provided [here](https://drive.google.com/file/d/1G5MP6Nzykku9sDDdzy7tlRqMBnKb253O/view?usp=sharing).


**Algorithms**. We provide the implementation of the following algorithms:
- DP3: `dp3.yaml`
- Simple DP3: `simple_dp3.yaml`

Among these, `dp3.yaml` is the proposed algorithm in our paper, showing a significant improvement over the baselines. During training, DP3 takes ~10G gpu memory and ~3 hours on an Nvidia A40 gpu, thus it is feasible for most researchers.

`simple_dp3.yaml` is a simplified version of DP3, which is much faster in training (1~2 hour) and inference (**25 FPS**) , without much performance loss, thus it is more recommended for robotics researchers.

# 💻 Installation

See [INSTALL.md](./docs/INSTALL.md) for installation instructions. 

See [ERROR_CATCH.md](./docs/ERROR_CATCH.md) for error catching I personally encountered during installation.

# 📚 Data
You could generate demonstrations by yourself using our provided expert policies.  Generated demonstrations are under `$YOUR_REPO_PATH/3D-Diffusion-Policy/data/`.
- Download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.
- Download DexArt assets from [Google Drive](https://drive.google.com/file/d/1JdReXZjMaqMO0HkZQ4YMiU2wTdGCgum1/view?usp=sharing) and put the `assets` folder under `$YOUR_REPO_PATH/third_party/dexart-release/`.

**Note: since you are generating demonstrations by yourselves, the results could be slightly different from the results reported in the paper. This is normal since the results of imitation learning highly depend on the demonstration quality.** Please re-generate demonstrations if you encounter some not-good results and no need to open an issue.

# 🛠️ Usage
Scripts for generating demonstrations, training, and evaluation are all provided in the `scripts/` folder. 

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.

For more detailed arguments, please refer to the scripts and the code. We here provide a simple instruction for using the codebase.

1. Generate demonstrations by `gen_demonstration_adroit.sh` and `gen_demonstration_dexart.sh`. See the scripts for details. For example:
    ```bash
    bash scripts/gen_demonstration_adroit.sh hammer
    ```
    This will generate demonstrations for the `hammer` task in Adroit environment. The data will be saved in `3D-Diffusion-Policy/data/` folder automatically.


2. Train and evaluate a policy with behavior cloning. For example:
    ```bash
    bash scripts/train_policy.sh dp3 adroit_hammer test 0 0
    ```
    This will train a DP3 policy on the `hammer` task in Adroit environment using point cloud modality. By default we **save** the ckpt (optional in the script).


3. Evaluate a saved policy or use it for inference. Please set  For example:
    ```bash
    bash scripts/eval_policy.sh dp3 adroit_hammer test 0 0
    ```
    This will evaluate the saved DP3 policy you just trained. **Note: the evaluation script is only provided for deployment/inference. For benchmarking, please use the results logged in wandb during training.**

# 🤖 Real Robot

**Hardware Setup**
1. Franka Robot
2. Allegro Hand
3. **L515** Realsense Camera (**Note: using the RealSense D435 camera might lead to failure of DP3 due to the very low quality of point clouds**)
4. Mounted connection base [[link](https://drive.google.com/file/d/1kg6yOFxVqP8azxPoXsuyig5DEQnAJjwC/view?usp=sharing)] (connect Franka with Allegro hand)
5. Mounted finger tip [[link](https://github.com/yzqin/dexpoint-release/blob/main/assets/robot/allegro_hand_description/meshes/modified_tip.STL)]

**Software**
1. Ubuntu 20.04.01 (tested)
2. [Franka Interface Control](https://frankaemika.github.io/docs/index.html) 
3. [Frankx](https://github.com/pantor/frankx) (High-Level Motion Library for the Franka Emika Robot)
4. [Allegro Hand Controller - Noetic](https://github.com/NYU-robot-learning/Allegro-Hand-Controller-DIME)


Every collected real robot demonstration (episode length: T) is a dictionary:
1. "point_cloud": Array of shape (T, Np, 6), Np is the number of point clouds, 6 denotes [x, y, z, r, g, b]. **Note: it is highly suggested to crop out the table/background and only leave the useful point clouds in your observation, which demonstrates effectiveness in our real-world experiments.**
2. "image": Array of shape (T, H, W, 3)
3. "depth": Array of shape (T, H, W)
4. "agent_pos": Array of shape (T, Nd), Nd is the action dim of the robot agent, i.e. 22 for our dexhand tasks (6d position of end effector + 16d joint position)
5. "action": Array of shape (T, Nd). We use *relative end-effector position control* for the robot arm and *relative joint-angle position control* for the dex hand.

For training and evaluation, you should process the point clouds (cropping using a bounding box and FPS downsampling) as described in the paper. We also provide an example script ([here](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master/scripts/convert_real_robot_data.py)). 

You can try using our provided real world data to train the policy.
1. Download the real robot data. Put the data under `3D-Diffusion-Policy/data/` folder, e.g. `3D-Diffusion-Policy/data/realdex_drill.zarr`, please keep the path the same as 'zarr_path' in the task's yaml file.
2. Train the policy. For example:
  ```bash
    bash scripts/train_policy.sh dp3 realdex_drill 0112 0 0
  ```
   
# 🔍 Visualizer
We provide a simple visualizer to visualize point clouds for the convenience of debugging in headless machines. You could install it by
```bash
cd visualizer
pip install -e .
```
Then you could visualize point clouds by
```python
import visualizer
your_pointcloud = ... # your point cloud data, numpy array with shape (N, 3) or (N, 6)
visualizer.visualize_pointcloud(your_pointcloud)
```
This will show the point cloud in a web browser.