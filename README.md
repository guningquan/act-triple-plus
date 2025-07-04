# TactileAloha_ML
#### Main Project Website: https://guningquan.github.io/TactileAloha

This repository is one of the repositories in the TactileAloha project.  We organize the code of the TactileAloha project as follows:  

- **This repository** focuses on robot teleoperation, dataset collection, and imitation learning algorithms. It can be placed anywhere on your computer. If you want to run this repository, please build the TactileAloha robot system, and install [TactileAloha_Hardware](https://github.com/guningquan/TactileAloha_Hardware) first.  
- **Another repository,** [TactileAloha_Hardware](https://github.com/guningquan/TactileAloha_Hardware), contains the hardware-related code, which is used for launching ROS hardware and should be placed in the ROS workspace.

## 📂 Repo Structure
- ``aloha_scripts`` Folders for controlling the robot and camera. You can use it to test teleoperation, record datasets, put the robot to sleep, and visualize the data. You can define your task in ``aloha_scripts/constants.py`` 
- ``detr`` Model definitions
- ``imitate_episodes_multi_gpu.py`` Train and Evaluate policy  with multiple GPUs
- ``policy.py`` An adaptor for ACT, ACTNormalsPolicy, DiffusionPolicy, CNNMLP, and our policy
- ``utils.py`` Utils such as data loading and helper functions

---


## 🏗️ Quick Start Guide

### 🖥️ Software Selection – OS Compatibility

This project has been tested and confirmed to work with the following configuration:  

- ✅ **Ubuntu 20.04 + ROS 1 Noetic** (Fully tested and verified)  

Other configurations may work as well, but they have not been tested yet. If you successfully run this project on a different setup, feel free to contribute by sharing your experience! 🚀

## 🛠️ Installation
```sh    
    git clone https://github.com/guningquan/act-triple-act
    cd act-triple-act
    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install -r requirements.txt
    cd detr && pip install -e .
    cd .. && cd robomimic  && pip install -e .
```
## 📑 Dataset Collection
1. 🤖 **TactileAloha robot system launch:**
We assume you have installed your robot system according to [TactileAloha_Hardware](https://github.com/guningquan/TactileAloha_Hardware). This step launches the four robot arms, three cameras, and a GelSight sensor.
    ``` ROS
    # ROS terminal
    conda deactivate
    source /opt/ros/noetic/setup.sh && source ~/interbotix_ws/devel/setup.sh
    roslaunch aloha tactile_aloha.launch
    ```

2. 📝 **Define the type of robotic manipulation dataset:**  
Including the task name, `dataset_dir`, length of each episode, and the cameras or GelSight sensor used.
You can set this information in `TASK_CONFIGS` of `aloha_scripts/constants.py`. An example is as follows:
    ```python
    'zip_tie': {
        'dataset_dir': DATA_DIR + '/saved_folder_name',
        'episode_len': 900, # This value may be modified according to the length of your task.
        'camera_names': ['cam_high', 
                           'cam_left_wrist', 
                           'cam_right_wrist', 
                           'gel']}
    ```
3. 🚀 **Star to teleoperation to task manipulation**: 
    ```
   cd act-triple-plus
   source ~/interbotix_ws/devel/setup.sh
    python aloha_scripts/record_episodes_compress.py \
    --task_name Task_name \
   --start_idx 0 --end_idx 50
    ```
   After each episode collection, you can enter `c` to save this episode and continue, or enter `r` to recollect this episode. If you want to quit this process, you can enter `q`. We use [foot pedals](https://www.amazon.co.jp/-/en/gp/product/B07FRMY4XB/ref=ox_sc_act_title_1?smid=A35GGB9A6044W2&psc=1) to assist with this confirmation, which can facilitate this work.
4. 📊 **Data visualization** :
    ```
    python aloha_scripts/visualize_episodes_multi.py --dataset_dir <data save dir> --episode_idx 0
    ```
    If you want to visualize multiple data points, you can input the `--episode_idx` parameter like this: `--episode_idx 3, 5, 8` or `--episode_idx 5-19`.
5. 🔄 **Robot shut down or sleep**:
    ```
    python aloha_scripts/sleep_plus.py --shut_down       # All robots will move to zero position and turn off the torque.
   python aloha_scripts/sleep_plus.py --shut_down_puppet   # Only the puppet robots will move to zero position and turn off the torque.
   python aloha_scripts/sleep_plus.py --sleep      # All robots will move to zero position but don't turn off the torque.
    ```
## 🧠 **Policy Training**  
   1. ✅ Set up your training configuration in ``aloha_scripts/constants.py``:
      ```python
          'zip_tie': {
           'dataset_dir': DATA_DIR + '/zip_tie_random',
           'episode_len': 900,
           'camera_names': ['cam_high',
                            # 'cam_low',
                            'cam_left_wrist',
                            'cam_right_wrist',
                            'gel'
                            ]
                    },
        'zip_tie_cotrain': {
           'dataset_dir': [
               DATA_DIR + '/zip_tie_random',
               DATA_DIR + '/mobile_aloha',
           ],  # only the first dataset_dir is used for val
           'stats_dir': [
               DATA_DIR + '/zip_tie_random',
           ],
           'sample_weights': [7.5, 2.5], 
           'train_ratio': 0.95,  # ratio of train data from the first dataset_dir
           'episode_len': 900,
           'camera_names': ['cam_high',
                            # 'cam_low',
                            'cam_left_wrist',
                            'cam_right_wrist',
                            'gel'
                            ]
       }
      ```
2. 🚀 Train your policy:
   ``` sh
   export CUDA_VISIBLE_DEVICES= 0, 1
   python imitate_episodes_multi_gpu.py  \
   --task_name zip_tie \
   --ckpt_dir  <data save dir>  \
   --policy_class TactileACT \
   --kl_weight 10 --chunk_size 100 \
   --hidden_dim 512 --batch_size 16 \
   --dim_feedforward 3200 --lr 1e-5 --seed 0 \
   --num_steps 100000 --eval_every 2000 \
   --validate_every 2000 --save_every 2000
   ```

## 📡 **Policy Deployment**
   ```sh
   export CUDA_VISIBLE_DEVICES= 0
   python imitate_episodes_multi_gpu.py  \
   --task_name zip_tie_insert \
   --ckpt_dir  <data save dir>  \
   --policy_class TactileACT \
   --kl_weight 10 --chunk_size 100 \
   --hidden_dim 512 --batch_size 16 \
   --dim_feedforward 3200 --lr 1e-5 --seed 0 \
   --num_steps 100000 --eval_every 2000 \
   --validate_every 2000 --save_every 2000 \
   --temporal_ensemble --eval --num_rollouts 20 --num_selected 50
   ```

## Note
1. The author finds that the decoder for tactile input can also be changed from ResNet to U-Net and still achieve good results. The author provides the code to allow readers to evaluate this change.
Specifically, one can easily comment out lines 393–394 in detr_vae.py and uncomment lines 391–392.
2. During deployment, different tasks or platforms may require tuning the aggregation parameters for the inferred action chunks. It is recommended to test these settings on your specific scenario to achieve the best results.

## 🙏 Acknowledgements
   This project codebase is built based on [ALOHA](https://github.com/tonyzhaozh/aloha) and [ACT](https://github.com/tonyzhaozh/act).