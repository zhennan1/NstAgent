#!/usr/bin/env bash
set -euo pipefail
root=${NARRATIVE_ROOT:?set NARRATIVE_ROOT to the training workspace (see README.md)}
export PYTHONPATH="$root:$root/narrative_grpo${PYTHONPATH:+:$PYTHONPATH}"
export LONGSTORY_USER_HOME=$HOME
export LONGSTORY_GRPO_ROOT="$root/narrative_grpo"
export MODEL_PATH=${MODEL_PATH:-$HOME/Qwen/Qwen3.5-4B}
export NARRATIVE_SEGMENTED_TRAINER=1
stage=${1:-smoke}
case "$stage" in
  smoke) steps=1; data=smoke; save=-1; experiment=narrative-ops-base-smoke ;;
  train) steps=64; data=train; save=32; experiment=narrative-ops-base-2epoch ;;
  *) exit 2 ;;
esac
if [[ "$stage" == smoke && -f "$root/narrative_grpo/results/smoke_replay_manifest.json" ]]; then
  export NARRATIVE_SMOKE_REPLAY_MANIFEST="$root/narrative_grpo/results/smoke_replay_manifest.json"
  export NARRATIVE_SMOKE_REPLAY_CLAIMS="$root/narrative_grpo/results/smoke-replay-claims/${HOSTNAME:-local}"
else
  unset NARRATIVE_SMOKE_REPLAY_MANIFEST NARRATIVE_SMOKE_REPLAY_CLAIMS
fi
export NARRATIVE_AUDIT_DIR="$root/narrative_grpo/results/$experiment"
export EXPERIMENT_NAME="$experiment"
mkdir -p "$NARRATIVE_AUDIT_DIR"
# Reuse the stage4 FSDP2/vLLM launcher; its dataset, reward, model, and
# checkpoint defaults are overridden below.
python "$root/narrative_grpo/run_native_base.py" stage4 \
  data.train_files="$root/narrative_grpo/data/training/$data.parquet" \
  data.val_files="$root/narrative_grpo/data/training/smoke.parquet" \
  data.train_batch_size=32 data.shuffle=False data.filter_overlong_prompts=False \
  ++data.apply_chat_template_kwargs.enable_thinking=True \
  data.max_prompt_length=114688 data.max_response_length=16384 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_fused_kernels=True \
  ++actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,extra]' \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.max_model_len=131072 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$root/narrative_grpo/native_agent.yaml" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=30 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=16 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  reward.custom_reward_function.path="$root/narrative_grpo/reward_fallback.py" \
  trainer.project_name=LongStory-narrative-ops trainer.balance_batch=True \
  trainer.total_epochs=1 trainer.total_training_steps="$steps" \
  trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq="$save" \
  trainer.max_actor_ckpt_to_keep=1 trainer.resume_mode=disable "${@:2}"
