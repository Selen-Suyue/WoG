# World Guidance: World Modeling in Condition Space for Action Generation

[![arXiv](https://img.shields.io/badge/arXiv-2602.22010-b31b1b.svg)](https://arxiv.org/abs/2602.22010)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://selen-suyue.github.io/WoGNet/)

This is an unofficial implemention of [WoG](https://selen-suyue.github.io/WoGNet/) based on open source framework and datasets.

Thanks to [OpenVLA](https://github.com/openvla/openvla) and [CogACT](https://github.com/microsoft/CogACT) for their awesome codebase.

![Teaser](assets/teaser.png)


## Installation
```bash
conda create -n wog python=3.10
```
Next, clone our repo and install the required packages:
```bash
git clone https://github.com/Selen-Suyue/WoG
cd WoG
pip install -e .
make setup
```
[Flash Attention](https://github.com/Dao-AILab/flash-attention) is needed for training. You can simply run:
```bash
pip install -e .[train]
```
or install it manually:
```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```


## Training WoG from Scratch
You can start the trainging from the weights of [OpenVLA](https://github.com/openvla/openvla) for greater efficiency. Please follow the instruction of [OpenVLA](https://github.com/openvla/openvla) to download their weights:
```bash
mkdir -p pretrained
cd pretrained

git clone git@hf.co:openvla/openvla-7b-prismatic

cd openvla-7b-prismatic
git lfs pull
```
Also, you can download the pretrained `ckpts(.pth)` of Vision Foundation Models in `pretrained/vision`. The Wan VAE is available at [Wan VAE](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/blob/main/Wan2.1_VAE.pth). The DINO and SigLIP weights can be downloaded via [utils/download_weights.py](utils/download_weights.py).

The data of [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) can be download following [OXE](https://robotics-transformer-x.github.io/) and [OpenVLA](https://github.com/openvla/openvla). Then launch the training script. For one node with 8 A100 GPUs as an example:

```bash
## For Stage I:
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint pretrained/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix oxe_magic_soup_plus \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir <path_to_dataset_dir> \
  --run_root_dir <path_to_log/checkpoint_dir> \                 
  --run_id <optional_run_id_for_wandb> \
  --image_aug <True_or_False> \
  --wandb_project <your_wandb_project> \
  --wandb_entity <your_wandb_entity> \
  --save_interval <num_of_steps_to_save_checkpoint> \
  --repeated_diffusion_steps 8 \
  --Ta 16 \
  --action_model_type DiT-L \
  --is_resume False \
  --train_venc True 

## For Stage II:
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint <ckpt predtrained in stage I> \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix oxe_magic_soup_plus \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir <path_to_dataset_dir> \
  --run_root_dir <path_to_log/checkpoint_dir> \                 
  --run_id <optional_run_id_for_wandb> \
  --image_aug <True_or_False> \
  --wandb_project <your_wandb_project> \
  --wandb_entity <your_wandb_entity> \
  --save_interval <num_of_steps_to_save_checkpoint> \
  --repeated_diffusion_steps 8 \
  --Ta 16 \
  --action_model_type DiT-L \
  --is_resume False
```

## Pretrained Checkpoints
We provide pretrained checkpoints of the two stages in [huggingface](https://huggingface.co/selensu/WoG). `WoG-V` is the first-stage and `WoG-A` is the second.

## Evaluation in SIMPLER
In this section, we provide a minimal evaluation for our models in [SIMPLER](https://simpler-env.github.io/). First, please follow the instruction of [SimplerEnv](https://github.com/simpler-env/SimplerEnv) to install the simulation environment. Next, add our [./deploy](./deploy) to [SimplerEnv/simpler_env/policies](https://github.com/simpler-env/SimplerEnv/tree/main/simpler_env/policies).
```bash
cp ./deploy <your_path_to_simpler>/simpler_env/policies -r
```
Then add a new policy model in [SimplerEnv/simpler_env/main_inference.py](https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/main_inference.py) as below:
```python
elif args.policy_model == "wog":
    from simpler_env.policies.deploy import WoGSimInference
    assert args.ckpt_path is not None
    model = WoGSimInference(
        saved_model_path=args.ckpt_path,  
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
        action_model_type='DiT-L',            
    )
```
After that, you can modify and launch the scripts in [`deploy/scripts`](deploy/scripts) like:
```bash
cd <your_path_to_simpler>
bash simpler_env/policies/deploy/scripts/wog_put_in_drawer_visual_matching.sh
```
<details>
<summary><strong>Ps: For Calvin Deployment</strong></summary>

We now support [CALVIN](https://github.com/mees/calvin) deployment. You can convert the CALVIN dataset to RLDS format with [calvin_rlds_builder](https://github.com/hyy02/calvin_rlds_builder). We've registered the calvin dataset in:
- [prismatic/vla/datasets/rlds/oxe/transforms.py](prismatic/vla/datasets/rlds/oxe/transforms.py)
- [prismatic/vla/datasets/rlds/oxe/mixtures.py](prismatic/vla/datasets/rlds/oxe/mixtures.py)
- [prismatic/vla/datasets/rlds/oxe/configs.py](prismatic/vla/datasets/rlds/oxe/configs.py)

for training. For deployment:
```bash
mv deploy/wog_policy_calvin.py your_path_to_calvin/calvin_models/calvin_agent/evaluation/ 
```
</details>

## Evaluation in RealWorld
We provide a sample of RealWorld Deploy wrapper in [deploy/wog_policy_real.py](deploy/wog_policy_real.py). It's supported by [Maniunicon](https://github.com/Universal-Control/ManiUniCon) and you can also use it in other platforms. 

ManiUnicon is a universal real-world robot control platform for data-collection and model deploy. You can refer the [README](https://github.com/Universal-Control/ManiUniCon/blob/main/README.md) to collect data in rlds format.

Once you have collected rlds dataset, modify the following files in our project:
- [prismatic/vla/datasets/rlds/oxe/transforms.py](prismatic/vla/datasets/rlds/oxe/transforms.py)
- [prismatic/vla/datasets/rlds/oxe/mixtures.py](prismatic/vla/datasets/rlds/oxe/mixtures.py)
- [prismatic/vla/datasets/rlds/oxe/configs.py](prismatic/vla/datasets/rlds/oxe/configs.py)

Then only stage-II is needed for training:
```bash
## For RealWorld:
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint <ckpt predtrained in stage I> \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix real_dataset_mix \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir <path_to_dataset_dir> \
  --run_root_dir <path_to_log/checkpoint_dir> \                 
  --run_id <optional_run_id_for_wandb> \
  --image_aug <True_or_False> \
  --wandb_project <your_wandb_project> \
  --wandb_entity <your_wandb_entity> \
  --save_interval <num_of_steps_to_save_checkpoint> \
  --repeated_diffusion_steps 8 \
  --Ta 16 \
  --action_model_type DiT-L \
  --is_resume False
```

After training, You can follow [wog config in maniunicon](https://github.com/Universal-Control/ManiUniCon/blob/main/configs/policy/wog.yaml) and [wog class in maniunicon](https://github.com/Universal-Control/ManiUniCon/blob/main/maniunicon/customize/policy_model/wog_model.py) to deploy WoG.


## 🔗 Citation

```bibtex
@article{WoG,
        title={World Guidance: World Modeling in Condition Space for Action Generation}, 
        author={Yue Su and Sijin Chen and Haixin Shi and Mingyu Liu and Zhengshen Zhang and Ningyuan Huang and Weiheng Zhong and Zhengbang Zhu and Yuxiao Liu and Xihui Liu},
        journal={arXiv preprint arXiv:2602.22010},
        year={2026},
  }
```

## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).
