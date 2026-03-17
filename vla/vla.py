from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast
from .qformer import QFormer

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from prismatic.models.backbones.vision.dinosiglip_vit import FutureEnc

from action_model.action_model import ActionModel
from action_model.models import DiT

overwatch = initialize_overwatch(__name__)
IGNORE_INDEX = -100


class WoG(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = "DiT-B",
        token_size: int = 4096,
        action_dim: int = 7,
        Ta: int = 16,
        scale: int = 4,  # for key frames
        use_ema: bool = False,
        pool_size=8,
        train_venc=False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            Ta=16,
            train_venc=train_venc,
        )
        self.vlm = vlm
        self.Ta = Ta
        self.scale = scale
        self.train_venc = train_venc
        self.pool_size = pool_size

        self.v_enc = FutureEnc()
        self.v_enc.eval()
        self.v_enc.requires_grad_(False)

        self.qformer = QFormer(hidden_dim=1024, nheads=8, dim_feedforward=2048, num_layers=6, num_query_tokens=16)

        if not self.train_venc:
            print(f"Freeze QFormer!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.qformer.eval()
            self.qformer.requires_grad_(False)
            self.oformer = QFormer(
                hidden_dim=token_size, nheads=8, dim_feedforward=2048, num_layers=4, num_query_tokens=16
            )

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ["action_model", "ema_diffusion", "qformer"]
        else:
            self.all_module_keys = ["action_model", "qformer"]
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)
        self._trainable_module_keys = ["action_model", "qformer"]
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys

    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone

    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone

    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        future_pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks=None,
    ) -> Tuple:
        output: CausalLMOutputWithPast = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden = output.hidden_states[-1]

        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif (
            hasattr(self.vlm.vision_backbone, "siglip_featurizer")
            and self.vlm.vision_backbone.siglip_featurizer is not None
        ):
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = last_hidden[:, num_patch:]
        obs_ins = last_hidden[:, -4:]

        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        ################ Train Vision Enc ##################
        B, T = next(iter(future_pixel_values.values())).shape[:2]
        reshaped_fpv = {k: v.view(B * T, *v.shape[2:]) for k, v in future_pixel_values.items()}  # [B*T, C, H, W]
        base_pv = {k: v for k, v in pixel_values.items()}  # [B, C, H, W]

        future_patch_features, vae_latent = self.v_enc(reshaped_fpv, base_pv)

        _BT, N, D = future_patch_features.shape
        H = W = int(N**0.5)
        x = future_patch_features.view(B * T, H, W, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, self.pool_size * self.pool_size, D)
        future_patch_features = x.view(B, T * (self.pool_size**2), -1).contiguous()
        future_patch_features = self.qformer(future_patch_features, vae_latent)
        ##################################################################

        future_repeated = future_patch_features.repeat(repeated_diffusion_steps, 1, 1)
        actions_future = actions[:, -self.Ta :, :]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(
            repeated_diffusion_steps, 1, 1
        )  # [repeated_diffusion_steps*B, 1, D]
        act_loss = self.action_model.loss(actions_repeated, cognition_features_repeated, future_repeated)
        if self.train_venc:
            obs_loss = torch.zeros_like(act_loss)
        else:
            obs_pre = self.oformer(obs_ins)

            target = future_patch_features.detach()
            obs_pre_n = F.normalize(obs_pre, dim=-1)
            target_n = F.normalize(target, dim=-1)
            obs_loss = (1.0 - (obs_pre_n * target_n).sum(dim=-1)).mean()

        loss = act_loss + 0.3 * obs_loss
        return loss, output, act_loss, obs_loss

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        Ta: int = 16,
        action_model_type: str = "DiT-B",
        use_ema: bool = False,
        norm_stats=None,
        train_venc=False,
        **kwargs,
    ) -> WoG:
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize WoG
        wog = WoG(
            vlm,
            token_size=vlm.llm_backbone.llm.lm_head.in_features,
            action_dim=action_dim,
            Ta=Ta,
            action_model_type=action_model_type,
            use_ema=use_ema,
            norm_stats=norm_stats,
            train_venc=train_venc,
        )

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            print("\033[92mLoading ActionModel from checkpoint.\033[0m")
            wog.action_model.load_state_dict(model_state_dict["action_model"], strict=False)
            if "ema_diffusion" in model_state_dict and use_ema:
                wog.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"], strict=False)
            elif use_ema:
                wog.ema_diffusion.load_state_dict(model_state_dict["action_model"], strict=False)
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")

        # Load QFormer from Checkpoint
        if "qformer" in model_state_dict:
            qformer_state_dict = model_state_dict["qformer"]
            model_keys = set(wog.qformer.state_dict().keys())
            checkpoint_keys = set(qformer_state_dict.keys())
            if model_keys != checkpoint_keys:
                print("\033[93mWARNING: qformer is not match: maybe the version is old\033[0m")
            else:
                print("\033[92mLoading QFormer from checkpoint.\033[0m")
                wog.qformer.load_state_dict(qformer_state_dict, strict=True)
        else:
            print("\033[93mNo QFormer found in the pretrained checkpoint. Initializing a new one.\033[0m")

        return wog

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        calvin_deploy: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        cognition_features = output.hidden_states[0][-1][:, -1, :]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (
            1,
            4096,
        ), "Batch size must be 1 for action prediction"
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]
        samples = self.action_model.inference(cognition_features)

        normalized_actions = samples[0].cpu().numpy()
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        if not calvin_deploy:
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        else:
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.0, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "Your model was trained on more than one dataset, "
                "please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            "The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
