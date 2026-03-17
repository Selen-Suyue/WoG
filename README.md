# World Guidance: World Modeling in Condition Space for Action Generation

[![arXiv](https://img.shields.io/badge/arXiv-2602.22010-b31b1b.svg)](https://arxiv.org/abs/2602.22010)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://selen-suyue.github.io/WoGNet/)

This is an unofficial implemention of [WoG](https://selen-suyue.github.io/WoGNet/) based on open source framework and datasets.

Thanks to [OpenVLA](https://github.com/openvla/openvla) and [CogACT](https://github.com/microsoft/CogACT) for their awesome codebase.

![Teaser](assets/teaser.png)


## 📝 TODO List

We are working on releasing the code. Stay tuned!

- [ ✓ ] Release training code
- [ ] Release pre-trained checkpoints
- [ ] Release Real-World and Simulation inference code

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
