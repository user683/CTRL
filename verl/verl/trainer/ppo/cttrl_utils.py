from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from verl.trainer.ppo.cttrl_memory import MemoryItem
from verl.trainer.ppo.cttrl_prm_client import PRMExample
from verl.utils.reward_score.ttrl_math import extract_answer, grade


@dataclass
class TrajectoryStats:
    prompt_index: int
    rollout_index: int
    prompt_text: str
    response_text: str
    segments: list[str]
    segment_scores: list[float]
    reliability: float
    consistency: float
    answer: str | None
    prompt_ids: list[int]
    response_ids: list[int]
    consensus_strength: float = 0.0
    trajectory_score: float = 0.0
    is_gold: bool = False
    update_selected: bool = False


def split_reasoning_segments(response_text: str, segment_delimiter: str = "\n\n") -> list[str]:
    if segment_delimiter == "<extra_0>" and "<extra_0>" in response_text:
        segments = [segment.strip() for segment in response_text.split("<extra_0>") if segment.strip()]
        if segments:
            return segments

    if "\n\n" in response_text:
        segments = [segment.strip() for segment in response_text.split("\n\n") if segment.strip()]
        if segments:
            return segments

    if segment_delimiter and segment_delimiter not in ["<extra_0>", "\n\n"] and segment_delimiter in response_text:
        segments = [segment.strip() for segment in response_text.split(segment_delimiter) if segment.strip()]
        if segments:
            return segments

    transitioned = re.split(r"(?=\b(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|So|Now|On\s+\w+day)\b)", response_text)
    segments = [segment.strip() for segment in transitioned if segment.strip()]
    if len(segments) > 1:
        return segments

    numbered = re.split(r"\n(?=\s*(?:Step\s*\d+[:.]|\d+[.)]))", response_text)
    segments = [segment.strip() for segment in numbered if segment.strip()]
    if len(segments) > 1:
        return segments

    return [response_text.strip()] if response_text.strip() else [""]


def split_reasoning_steps(response_text: str, step_delimiter: str = "\n\n") -> list[str]:
    return split_reasoning_segments(response_text, segment_delimiter=step_delimiter)


def build_prm_examples(prompt_texts: list[str], response_texts: list[str], step_delimiter: str) -> list[PRMExample]:
    examples = []
    for prompt_text, response_text in zip(prompt_texts, response_texts):
        segments = split_reasoning_segments(response_text, segment_delimiter=step_delimiter)
        examples.append(PRMExample(prompt=prompt_text, response=response_text, segments=segments))
    return examples


def compute_reliability(step_scores: list[float], bottom_m: int) -> float:
    if not step_scores:
        return 0.0
    m = max(1, min(bottom_m, len(step_scores)))
    tail = sorted(step_scores)[:m]
    return float(sum(tail) / m)


def compute_consistency(step_scores: list[float]) -> float:
    if not step_scores:
        return 0.0
    return float(sum(step_scores) / len(step_scores))


def compute_trajectory_score(step_scores: list[float]) -> float:
    """Compute trajectory score: S(s_i) = Mean(s_i) = (1/T) Σ r_{i,j}."""
    return compute_consistency(step_scores)


def compute_majority_answers(stats: list[TrajectoryStats]) -> tuple[dict[int, str | None], dict[int, float]]:
    grouped_answers: dict[int, list[str]] = defaultdict(list)
    grouped_total: dict[int, int] = defaultdict(int)
    for item in stats:
        grouped_total[item.prompt_index] += 1
        if item.answer is not None:
            grouped_answers[item.prompt_index].append(item.answer)

    majority_answers: dict[int, str | None] = {}
    consensus_strength: dict[int, float] = {}
    for prompt_index, total in grouped_total.items():
        answers = grouped_answers.get(prompt_index, [])
        if not answers:
            majority_answers[prompt_index] = None
            consensus_strength[prompt_index] = 0.0
            continue

        answer, count = Counter(answers).most_common(1)[0]
        majority_answers[prompt_index] = answer
        consensus_strength[prompt_index] = count / max(total, 1)
    return majority_answers, consensus_strength


def build_reward_tensor(scores: list[float], response_mask: torch.Tensor) -> torch.Tensor:
    reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
    for row_idx, score in enumerate(scores):
        valid_positions = torch.nonzero(response_mask[row_idx], as_tuple=False).flatten()
        if len(valid_positions) == 0:
            continue
        reward_tensor[row_idx, valid_positions[-1]] = float(score)
    return reward_tensor


def clip_advantage_scores(scores: list[float], clip_value: float) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    normalized = (arr - mean) / (std + 1e-6)
    return np.clip(normalized, -clip_value, clip_value).tolist()


def select_gold_trajectory(
    candidates: list[TrajectoryStats],
    gamma_r: float,
    gamma_c: float,
    lambda_reliability: float,
    lambda_consistency: float,
    lambda_length: float,
) -> TrajectoryStats | None:
    filtered = [
        item for item in candidates if item.reliability >= gamma_r and item.consistency >= gamma_c and item.answer is not None
    ]
    if not filtered:
        return None

    def score(item: TrajectoryStats) -> float:
        length_penalty = lambda_length * math.log(max(len(item.response_text), 2))
        return lambda_reliability * item.reliability + lambda_consistency * item.consistency - length_penalty

    return max(filtered, key=score)


def should_update(consensus_strength: float, max_reliability: float, gamma_mv_update: float, gamma_r_update: float) -> bool:
    return consensus_strength >= gamma_mv_update and max_reliability >= gamma_r_update


def build_memory_item(
    item: TrajectoryStats,
    lambda_reliability: float,
    lambda_consistency: float,
    lambda_length: float,
) -> MemoryItem:
    length_penalty = lambda_length * math.log(max(len(item.response_text), 2))
    priority = lambda_reliability * item.reliability + lambda_consistency * item.consistency - length_penalty
    variance = float(np.var(item.segment_scores)) if item.segment_scores else 0.0
    return MemoryItem(
        priority=priority,
        prompt=item.prompt_text,
        response=item.response_text,
        reliability=item.reliability,
        consistency=item.consistency,
        variance=variance,
        answer=item.answer,
        metadata={
            "prompt_index": item.prompt_index,
            "rollout_index": item.rollout_index,
            "segment_scores": item.segment_scores,
            "prompt_ids": item.prompt_ids,
            "response_ids": item.response_ids,
        },
    )


def build_segment_token_mask(
    n_response_tokens: int,
    segments: list[str],
    segment_scores: list[float],
    gamma_segment: float,
) -> list[int]:
    return build_step_token_mask(
        n_response_tokens=n_response_tokens,
        steps=segments,
        step_scores=segment_scores,
        gamma_step=gamma_segment,
    )


def build_step_token_mask(
    n_response_tokens: int,
    steps: list[str],
    step_scores: list[float],
    gamma_step: float,
) -> list[int]:
    """Build a token-level binary mask from step-level PRM scores.

    Uses character-proportional mapping to approximate token boundaries.
    Tokens belonging to steps with PRM score < gamma_step are masked out
    so that only high-quality reasoning steps contribute to the supervised loss.

    Corresponds to:
        w_{t,tau}^{gold} = 1[PRM(s_{t,tau}) >= gamma_step]
    in the algorithm specification (Step 12 & Step 15).
    """
    if not steps or not step_scores or n_response_tokens == 0:
        return [1] * n_response_tokens

    n_steps = min(len(steps), len(step_scores))
    char_lengths = [max(len(steps[i]), 1) for i in range(n_steps)]
    total_chars = sum(char_lengths)

    if total_chars == 0:
        return [1] * n_response_tokens

    mask = [0] * n_response_tokens
    pos = 0
    for i in range(n_steps):
        if i == n_steps - 1:
            end = n_response_tokens
        else:
            end = int(round(sum(char_lengths[: i + 1]) / total_chars * n_response_tokens))
        end = min(end, n_response_tokens)

        if step_scores[i] >= gamma_step:
            for t in range(pos, end):
                mask[t] = 1
        pos = end

    # Fallback: if every step was filtered out, include all tokens to avoid
    # a degenerate zero-loss batch.
    if sum(mask) == 0:
        return [1] * n_response_tokens

    return mask


def safe_extract_answer(response_text: str) -> str | None:
    answer = extract_answer(response_text)
    if answer is not None:
        return answer

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1][-128:]


def compute_cttrl_external_eval_metrics(
    stats: list[TrajectoryStats],
    prompt_count: int,
    n_rollouts: int,
    ground_truths: list[str],
    consensus_strengths: dict[int, float],
) -> dict[str, float]:
    """Compute legacy-style external evaluation metrics for C-TTRL.

    These metrics preserve the original TTRL evaluation semantics and are used
    only for logging/analysis, not for defining the current training reward.
    """
    assert len(ground_truths) == prompt_count

    label_hits = []
    prompt_ground_truth_rewards = []
    prompt_pass_at_k = []

    for prompt_index in range(prompt_count):
        prompt_stats = stats[prompt_index * n_rollouts : (prompt_index + 1) * n_rollouts]
        gt_answer = ground_truths[prompt_index]

        answers = [item.answer for item in prompt_stats if item.answer is not None]
        if answers:
            majority_answer, _ = Counter(answers).most_common(1)[0]
            label_hits.append(1.0 if grade(majority_answer, gt_answer) else 0.0)
        else:
            label_hits.append(0.0)

        gt_rewards = []
        for item in prompt_stats:
            if item.answer is None:
                gt_rewards.append(0.0)
            else:
                gt_rewards.append(1.0 if grade(item.answer, gt_answer) else 0.0)

        prompt_ground_truth_rewards.append(float(sum(gt_rewards) / max(len(gt_rewards), 1)))
        prompt_pass_at_k.append(1.0 if sum(gt_rewards) >= 1 else 0.0)

    majority_ratio = float(np.mean(list(consensus_strengths.values()) or [0.0]))
    return {
        "label_accuracy": float(np.mean(label_hits or [0.0])),
        "ground_truth_reward": float(np.mean(prompt_ground_truth_rewards or [0.0])),
        f"pass@{n_rollouts}": float(np.mean(prompt_pass_at_k or [0.0])),
        "majority_ratio": majority_ratio,
    }
