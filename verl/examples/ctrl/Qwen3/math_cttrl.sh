#!/bin/bash
export RAY_DISABLE_DASHBOARD=1
export CTTRL_PRM_DEBUG=0
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1=11

# ===== PRM configuration =====
# Mode selection: "api" uses a remote API, "local" uses a local model
export CTTRL_PRM_MODE="api"

# --- local mode settings ---
# Local Qwen2.5-7B-PRM model path
export CTTRL_PRM_DEVICE="${CTTRL_PRM_DEVICE:-cuda:0}"
export CTTRL_PRM_MODEL_PATH="${CTTRL_PRM_MODEL_PATH:-$HOME/models/Qwen2.5-Math-PRM-7B}"

# --- api mode settings ---
# Example:
#   export CTTRL_PRM_MODE=api
#   export CTTRL_PRM_API_BASE=http://<PRM_SERVER_IP>:8008
#   export CTTRL_PRM_API_KEY=dummy
#   export CTTRL_PRM_MODEL=local-prm
#
# DeepSeek v4 example:
#   export CTTRL_PRM_API_BASE=https://api.deepseek.com
#   export CTTRL_PRM_API_KEY_ENV=DEEPSEEK_API_KEY
#   export CTTRL_PRM_MODEL=deepseek-v4-pro
#   export CTTRL_PRM_ENDPOINT_PATH=/chat/completions
#   export CTTRL_PRM_REASONING_EFFORT=high
#   export CTTRL_PRM_EXTRA_BODY='{"thinking":{"type":"enabled"}}'
export CTTRL_PRM_API_BASE="${CTTRL_PRM_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export CTTRL_PRM_API_KEY_ENV="${CTTRL_PRM_API_KEY_ENV:-CTTRL_PRM_API_KEY}"
export CTTRL_PRM_API_KEY="${CTTRL_PRM_API_KEY:-}"
export CTTRL_PRM_MODEL="${CTTRL_PRM_MODEL:-qwen-plus}"
export CTTRL_PRM_ENDPOINT_PATH="${CTTRL_PRM_ENDPOINT_PATH:-/chat/completions}"
export CTTRL_PRM_CONCURRENCY="${CTTRL_PRM_CONCURRENCY:-8}"
export CTTRL_PRM_RATE_LIMIT_PER_MINUTE="${CTTRL_PRM_RATE_LIMIT_PER_MINUTE:-600}"

echo "Using API PRM:"
echo "  CTTRL_PRM_API_BASE=$CTTRL_PRM_API_BASE"
echo "  CTTRL_PRM_API_KEY_ENV=$CTTRL_PRM_API_KEY_ENV"
echo "  CTTRL_PRM_MODEL=$CTTRL_PRM_MODEL"
echo "  CTTRL_PRM_ENDPOINT_PATH=$CTTRL_PRM_ENDPOINT_PATH"
echo "  CTTRL_PRM_CONCURRENCY=$CTTRL_PRM_CONCURRENCY"
echo "  CTTRL_PRM_RATE_LIMIT_PER_MINUTE=$CTTRL_PRM_RATE_LIMIT_PER_MINUTE"


DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AMC-TTT"
BACKBONE="Qwen3-4B-Base"
ADVANTAGE="grpo"

K=32
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * 3))

EPISODE=20
DATA_TRAIN_BATCH_SIZE=8
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1


DATA_LOCAL_DIR="$PWD/verl/data"
BACKBONE_PATH="$HOME/checkpoints/TTRL-verl/AIME-TTT-Qwen3-4B-Base/actor/merged_hf_lora"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="C-TTRL"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"


python -m verl.trainer.main_ppo \
--config-name='ppo_trainer_cttrl.yaml'\
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \boxed{}."' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.lora_rank=128 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=$K \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  algorithm.kl_ctrl.kl_coef=0.01 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  cttrl.enable=True \
  cttrl.n_rollouts=$K \
  cttrl.posterior_correction.enable=True \
  cttrl.prm.endpoint_path=$CTTRL_PRM_ENDPOINT_PATH \
  trainer.logger=['console','tensorboard'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=60 \
  trainer.test_freq=-1 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@" \
  trainer.val_before_train=False

echo "Output directory: $OUTPUT_DIR"
