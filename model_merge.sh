# Set variables
CKPT_DIR="$HOME/checkpoints/TTRL-verl/AIME-TTT-Qwen3-1.7B-Base/actor"

# Merge LoRA
python -m peft.merge_adapter \
    --base_model_path "$CKPT_DIR/huggingface" \
    --adapter_path "$CKPT_DIR/lora_adapter" \
    --output_dir "$CKPT_DIR/merged_hf_lora"

echo "✅ LoRA merged successfully to: $CKPT_DIR/merged_hf_lora"
