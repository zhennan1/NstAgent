# RL study: chapter-level GRPO on the NstAgent loop

The training stack behind the appendix "Reinforcement Learning for NstAgent", as it ran.
It is built on VeRL (commit `16bdafaada14bccf317bfae13332df08fc2be857`, 0.8.0.dev0) and is not needed
for anything else in this repository. The run used one node with 8×A800, Python 3.12, torch 2.10.0,
vLLM 0.18.0, ray 2.55.1, and flash-attn 2.8.3.

## Where each part of the appendix lives

| Appendix | Files |
|---|---|
| Premises: 256 English premises, 32 genres × 8, disjoint from ConStory-Bench | `make_prompts.py` → `data/rl/train_premises.jsonl` |
| First nine chapters prepared with the initial model and saved as prefixes | `prepare.py`, `narrative_config.py` (generation profile) |
| Prefixes audited before training (well-formed state and chapter plan) | `audit_prefixes.py` |
| Batches: 32 distinct contexts per step, prefix index j sampled per length | `build_schedule.py` → `data/rl/train_schedule.json` |
| Rollout through the deployed agent loop, same tools and update | `native_agent.yaml` → `native_agent_loop.py`, `native_rollout.py`, `agent_loop.py`, `longstoryagent_dynamic_tokens.py` |
| GRPO at chapter level, group size 8; tool-response tokens masked | `chapter_advantages.py` (normalization per chapter group, before turn expansion), `segment_batch.py` (each model turn becomes one sample whose response mask covers only the model's tokens), `native_trainer.py` |
| Hyperparameters: lr 5e-7, KL 0.01 (low-variance KL loss), 32 contexts × 8 rollouts | `run_train.sh` (experiment overrides) on top of `run_grpo_base.sh` (base VeRL arguments) |
| Reward, length gate, zero reward unless write + update + DONE | `reward.py::score_chapter` |
| The three judges, shared preamble, consistency anchors, state rubric | `reward.py` (`COMMON`, `RUBRICS`) |
| Judge budget: 32,768 output tokens, 900 s timeout, concurrency 8 | `reward.py`, `judge_rate_limit.py` |
| Evaluation on 20 ConStory-Bench prompts at 10K words, shared outlines | `benchmark16.py` (initial model and step 16), `benchmark48.py` (step 48; step 32 is the same script with 32), `export_step48.py` |
| Judging with WritingBench and every-chapter ConStory-Bench | `judge_benchmark16.py`, `judge_benchmark48.py`, `rejudge_fullmarker64.py` (initial model), `rejudge_fullmarker_ckpt.py` (steps 16/32/48) |
| Table of checkpoint results | `code/analysis/rl_table.py` on `results/rl/` |

## Launch chain

`run_train.sh train` → `run_staged_base.py` → `run_native_base.py`, which runs `run_grpo_base.sh` with the
VeRL entry replaced by `staged_train_entry.py`. That entry installs the chapter-aware trainer
(`native_trainer.py`) and the checkpoint publisher (`staged_checkpoint.py`) into an unmodified VeRL, and
the agent loop named in `native_agent.yaml` supplies the reward, so `reward_fallback.py` only raises if the
integration is broken. `native_train_entry.py` is the same entry without the checkpoint override.

The scripts expect the training workspace layout, with `NARRATIVE_ROOT` pointing at its root:

```
$NARRATIVE_ROOT/
  longstoryagent_dynamic_tokens.py   (this directory's copy)
  generation_common.py               (identical to code/agent/generation_common.py)
  narrative_grpo/                    (the other files of this directory, plus data/, results/, models/)
```

`VERL_ROOT` points at the VeRL checkout and `MODEL_PATH` at Qwen3.5-4B. The judge key is read from
`NARRATIVE_JUDGE_CREDENTIAL_FILE`; no key or endpoint is included.

## Notes on the run

- The run was resumed several times after infrastructure failures. The resumptions changed only memory
  offloading, context limits, the judge endpoint, and judge concurrency (12 for steps 62–64, after the
  reported checkpoints); none changed the algorithm or the reward.
- The judges return a bare score (the consistency judge first reports whether the new chapter contains
  an explicit same-time contradiction); they do not return a reason or quoted evidence.
- Checkpoints 16, 32, and 48 are the ones reported. `data/rl/` holds the premises, the batch schedule,
  and the 20 evaluation prompts; the frozen prefixes (3.7 GB) and model weights are not included.
