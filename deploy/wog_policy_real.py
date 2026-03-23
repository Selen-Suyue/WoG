import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
import os
import argparse
import json
import math
from flask import Flask, request, jsonify
import tempfile
import torch
from vla import load_vla
from deploy.adaptive_ensemble import AdaptiveEnsembler
from maniunicon.utils.vla_utils import euler_pose_to_quat

app = Flask(__name__)


class WoGpolicy:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        action_model_type: str = "DiT-L",
        future_action_window_size: int = 15,
        use_bf16: bool = True,
        action_dim: int = 7,
        action_ensemble: bool = True,
        adaptive_ensemble_alpha: float = 0.1,
        action_ensemble_horizon: int = 2,
        action_chunking: bool = False,
        action_chunking_window: Optional[int] = None,
        args=None,
        task_name: str = None,
        **kwargs,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert not (
            action_chunking and action_ensemble
        ), "Now 'action_chunking' and 'action_ensemble' cannot both be True."

        self.unnorm_key = unnorm_key
        self.task_name = task_name

        print(f"*** unnorm_key: {unnorm_key} ***")
        self.vla = load_vla(
            saved_model_path,
            load_for_training=False,
            action_model_type=action_model_type,
            future_action_window_size=future_action_window_size,
            action_dim=action_dim,
        )
        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()

        self.image_size = image_size
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.action_chunking = action_chunking
        self.action_chunking_window = action_chunking_window
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None

        self.args = args
        self.reset()

    def reset(self) -> None:
        if self.action_ensemble:
            self.action_ensembler.reset()

    def __call__(
        self, obs, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        task_description = self.task_name
        image_numpy = obs["image"]["camera_0"].squeeze().cpu().detach().numpy()
        image_pil = Image.fromarray(image_numpy)

        resized_image = resize_image(image_pil, size=self.image_size)
        unnormed_actions, normalized_actions = self.vla.predict_action(
            image=resized_image,
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )

        N, A = unnormed_actions.shape
        unnormed_actions = np.concatenate(
            [
                euler_pose_to_quat(
                    unnormed_actions[..., :-1].reshape(-1, A - 1)
                ).reshape(N, -1),
                unnormed_actions[..., -1:],
            ],
            axis=-1,
        )

        print(f"Instruction: {task_description}")
        return unnormed_actions[:8, :]  # better for real-time inference


def resize_image(image: Image, size=(224, 224), shift_to_left=0):
    w, h = image.size
    # assert h < w, "Height should be less than width"
    left_margin = (w - h) // 2 - shift_to_left
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    image = image.resize(size, resample=Image.LANCZOS)

    image = scale_and_resize(
        image, target_size=(224, 224), scale=0.9, margin_w_ratio=0.5, margin_h_ratio=0.5
    )
    return image


def scale_and_resize(
    image: Image,
    target_size=(224, 224),
    scale=0.9,
    margin_w_ratio=0.5,
    margin_h_ratio=0.5,
):
    w, h = image.size
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w_max = w - new_w
    margin_h_max = h - new_h
    margin_w = int(margin_w_max * margin_w_ratio)
    margin_h = int(margin_h_max * margin_h_ratio)
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(target_size, resample=Image.LANCZOS)
    return image