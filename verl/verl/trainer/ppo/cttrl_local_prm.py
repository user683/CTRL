"""Local PRM client using Qwen2.5-Math-PRM (or compatible) models.

Loads the model into GPU memory and scores reasoning steps by extracting
reward scores at ``<extra_0>`` step-boundary positions.  Implements the same
interface as ``OpenAICompatiblePRMClient`` so it can be used as a drop-in
replacement.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
from transformers import AutoModel, AutoTokenizer

from verl.trainer.ppo.cttrl_prm_client import PRMExample


class LocalPRMClient:
    """Score reasoning steps using a locally loaded PRM model.

    Designed for Qwen2.5-Math-PRM-7B and compatible reward models that use
    ``<extra_0>`` as step boundaries and output scalar rewards.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        step_tag: str = "<extra_0>",
        max_length: int = 8192,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.step_tag = step_tag
        self.max_length = max_length
        self.system_prompt = system_prompt or (
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        self._ensure_dynamic_cache_compat()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        pt_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self._device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=pt_dtype, trust_remote_code=True,
        ).to(self._device).eval()

        # Resolve step tag token ID
        self.step_tag_id = self._resolve_token_id(step_tag)

        self.debug = os.getenv("CTTRL_PRM_DEBUG", "").lower() in {"1", "true", "yes"}
        if self.debug:
            print(f"[LocalPRM] model={model_path}, device={self._device}")
            print(f"[LocalPRM] step_tag_id={self.step_tag_id}")

    @staticmethod
    def _ensure_dynamic_cache_compat() -> None:
        """Patch DynamicCache for old model code expecting get_usable_length."""
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return

        if hasattr(DynamicCache, "get_usable_length"):
            return

        def _get_usable_length(self, new_seq_len, layer_idx=0):
            if hasattr(self, "get_seq_length"):
                return self.get_seq_length(layer_idx)
            return 0

        DynamicCache.get_usable_length = _get_usable_length

    def _resolve_token_id(self, token_str: str) -> int:
        ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Expected token '{token_str}' to map to a single id, got {ids}. "
                f"Check the tokenizer for model {self.model_path}."
            )
        return ids[0]

    @staticmethod
    def _resolve_device(requested_device: str | None) -> torch.device:
        device_str = requested_device or "cuda"
        device = torch.device(device_str)
        if device.type != "cuda":
            raise ValueError(
                f"LocalPRMClient requires a CUDA device, but got '{device_str}'. "
                "Set CTTRL_PRM_DEVICE to a CUDA device such as 'cuda' or 'cuda:0'."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"LocalPRMClient requires CUDA, but device '{device_str}' is not available in this process. "
                "If you are running under Ray, make sure the actor constructing LocalPRMClient requests GPU "
                "resources and that the cluster reserves one GPU for the local PRM."
            )
        return device

    @property
    def enabled(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Public API (matches OpenAICompatiblePRMClient)
    # ------------------------------------------------------------------

    def score_batch(self, examples: Iterable[PRMExample]) -> list[list[float]]:
        return [self.score_example(ex) for ex in examples]

    def score_example(self, example: PRMExample) -> list[float]:
        n_segments = len(example.segments)
        if n_segments == 0:
            return []

        # Build the assistant content by joining segments with step tags
        assistant_content = (f" {self.step_tag}\n").join(example.segments)
        # Ensure the last segment also ends with a step tag
        if not assistant_content.endswith(self.step_tag):
            assistant_content += f" {self.step_tag}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": assistant_content},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        input_ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length,
        ).input_ids.to(self._device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)

        # Qwen2.5-Math-PRM outputs scalar rewards: logits shape (1, seq_len, 1)
        reward_logits = outputs.logits  # (1, seq_len, 1)

        # Find positions of the step tag tokens
        step_positions = (input_ids[0] == self.step_tag_id).nonzero(as_tuple=True)[0]

        if self.debug:
            print(f"[LocalPRM] input_len={input_ids.shape[1]}, step_positions={step_positions.tolist()}")

        # Extract reward scores at step boundary positions and convert to [0,1]
        scores: list[float] = []
        for pos in step_positions:
            reward = reward_logits[0, pos, 0]
            prob = torch.sigmoid(reward).item()
            scores.append(prob)

        # Pad or truncate to match expected segment count
        return self._normalize_scores(scores, n_segments)

    @staticmethod
    def _normalize_scores(scores: list[float], expected_len: int) -> list[float]:
        clamped = [min(1.0, max(0.0, float(v))) for v in scores[:expected_len]]
        if len(clamped) < expected_len:
            clamped.extend([0.0] * (expected_len - len(clamped)))
        return clamped
