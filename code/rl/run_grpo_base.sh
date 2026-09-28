#!/usr/bin/env bash
set -euo pipefail

stage=${1:-stage1}
case "$stage" in
  smoke|stage1) data_stage=stage1; max_prompt_length=256; max_response_length=2048 ;;
  smoke2|stage2) data_stage=stage2; max_prompt_length=256; max_response_length=4096 ;;
  smoke3|stage3) data_stage=stage3; max_prompt_length=256; max_response_length=16384 ;;
  smoke4|stage4) data_stage=stage4; max_prompt_length=2048; max_response_length=8192 ;;
  *) echo "usage: $0 {smoke,smoke2,smoke3,smoke4,stage1,stage2,stage3,stage4} [Hydra overrides...]" >&2; exit 2 ;;
esac
shift || true
max_model_len=$((max_prompt_length + max_response_length))

user_home=${LONGSTORY_USER_HOME:-$HOME}
project_root=${LONGSTORY_GRPO_ROOT:-$user_home/longstory-grpo}
env_name=${LONGSTORY_GRPO_ENV:-longstory-qwen35-grpo}
model_path=${MODEL_PATH:-$user_home/Qwen/Qwen3.5-4B}
train_file="$project_root/data/$data_stage/train.parquet"
val_file="$project_root/data/$data_stage/val.parquet"
reward_file="$project_root/word_count_reward.py"
if [[ "$stage" == smoke4 || "$stage" == stage4 ]]; then
  reward_file="$project_root/framework_word_count_reward.py"
fi
experiment_name=${EXPERIMENT_NAME:-qwen35-4b-word-count-$stage}
checkpoint_dir="$project_root/checkpoints/$experiment_name"
log_dir="$project_root/logs"
max_actor_ckpt_to_keep=2
save_freq=20
if [[ "$stage" == stage4 ]]; then
  # The shared filesystem has room for one 4B FSDP checkpoint, but retaining
  # both step-20 and step-40 would consume roughly another 102 GiB, so keep
  # only the newest Stage4 checkpoint.
  max_actor_ckpt_to_keep=1
  save_freq=40
fi
mkdir -p "$checkpoint_dir" "$log_dir" "$project_root/ray"

source "$user_home/miniconda3/etc/profile.d/conda.sh"
conda activate "$env_name"
unset RAY_ADDRESS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export RAY_TMPDIR="$project_root/ray"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

common_overrides=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="$train_file"
  data.val_files="$val_file"
  data.train_batch_size=8
  data.max_prompt_length="$max_prompt_length"
  data.max_response_length="$max_response_length"
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=True
  # Word-count control is a direct generation task.  With Qwen3.5's default
  # thinking mode, every smoke rollout exhausted 2048 tokens before closing
  # <think>, leaving no visible prose and therefore no reward signal.  Disable
  # hidden reasoning through the official chat-template switch; this does not
  # add any user/system text to the one-turn prompt.
  +data.apply_chat_template_kwargs.enable_thinking=False
  actor_rollout_ref.model.path="$model_path"
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.fsdp_config.fsdp_size=8
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
  actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  # Torch 2.10 FSDP2 cannot materialize a state_dict for VeRL's rollout weight
  # sync while CPUOffloadPolicy owns the actor shards. The 4B actor easily fits
  # sharded across eight A800s, so keep its policy on GPU.
  actor_rollout_ref.actor.fsdp_config.offload_policy=False
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.ref.fsdp_config.offload_policy=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.ignore_eos=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_model_len="$max_model_len"
  actor_rollout_ref.rollout.max_num_seqs=64
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.enable_prefix_caching=False
  reward.custom_reward_function.path="$reward_file"
  reward.custom_reward_function.name=compute_score
  trainer.critic_warmup=0
  trainer.logger='[console]'
  trainer.project_name=LongStory-word-count-GRPO
  trainer.experiment_name="$experiment_name"
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  trainer.balance_batch=False
  trainer.default_local_dir="$checkpoint_dir"
  trainer.resume_mode=auto
  trainer.val_before_train=True
  trainer.save_freq="$save_freq"
  trainer.test_freq=20
  trainer.max_actor_ckpt_to_keep="$max_actor_ckpt_to_keep"
  trainer.total_epochs=1
)

if [[ "$stage" == smoke4 || "$stage" == stage4 ]]; then
  common_overrides+=(
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$project_root/write_tool_config.yaml"
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.agent.num_workers=2
    actor_rollout_ref.rollout.disable_log_stats=False
  )
fi

if [[ "$stage" == smoke* ]]; then
  common_overrides+=(
    data.train_batch_size=8
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.rollout.n=4
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_training_steps=1
    trainer.resume_mode=disable
  )
fi

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="$log_dir/${experiment_name}_${timestamp}.log"
python -m verl.trainer.main_ppo "${common_overrides[@]}" "$@" 2>&1 | tee "$log_file"
