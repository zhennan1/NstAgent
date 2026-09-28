"""Experiment-local VeRL integration; never modifies the installed VeRL checkout."""
import math
from verl.trainer.ppo import ray_trainer as upstream
from segment_batch import expand_native_batch,apply_native_advantages

_original_compute_advantage=upstream.compute_advantage


def compute_native_advantage(batch, adv_estimator, **kwargs):
    if 'native_chapter_advantage' not in batch.non_tensor_batch:
        return _original_compute_advantage(batch,adv_estimator=adv_estimator,**kwargs)
    if getattr(adv_estimator,'value',adv_estimator)!='grpo':
        raise ValueError('Native chapter normalization supports GRPO only')
    if not kwargs.get('norm_adv_by_std_in_grpo',True):
        raise ValueError('Unexpected GRPO normalization change')
    return apply_native_advantages(batch)


class NativePPOTrainer(upstream.RayPPOTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.config.trainer.balance_batch:
            raise ValueError('Native segment expansion requires the pre-forward balance hook')
        if self.config.algorithm.use_kl_in_reward or self.use_critic:
            raise ValueError('Only chapter-outcome GRPO is supported; actor KL loss remains available')
        if self.config.actor_rollout_ref.actor.loss_agg_mode!='token-mean':
            raise ValueError('Per-turn sequence averaging would change the chapter objective')
        if getattr(self,'use_prefix_grouper',False):
            raise ValueError('Prefix grouping has not been validated with unequal native turn counts')

    def _balance_batch(self,batch,metrics,logging_prefix='global_seqlen',keep_minibatch=False):
        if 'native_turns' in batch.non_tensor_batch:
            n=self.config.actor_rollout_ref.rollout.n
            dp=self._get_dp_size(self.actor_rollout_wg,'actor')
            expanded=expand_native_batch(batch,self.tokenizer.pad_token_id,n,math.lcm(n,dp))
            # fit owns this DataProto reference; replace its contents in place before
            # old-policy/reference forward passes and all subsequent tensor creation.
            batch.batch=expanded.batch
            batch.non_tensor_batch=expanded.non_tensor_batch
            batch.meta_info=expanded.meta_info
            metrics['native/chapter_candidates']=batch.meta_info['native_chapter_count']
            metrics['native/turns']=batch.meta_info['native_turn_count']
            metrics['native/padding_rows']=int(batch.non_tensor_batch['native_padding'].sum())
        return super()._balance_batch(batch,metrics,logging_prefix,keep_minibatch)

    def _update_actor(self,batch):
        if 'native_chapter_advantage' not in batch.non_tensor_batch:
            raise ValueError('Refusing actor update on representative-only chapter rows')
        config=self.config.actor_rollout_ref.actor
        n=self.config.actor_rollout_ref.rollout.n
        if len(batch)%n:
            raise ValueError('Expanded batch is not aligned to rollout group size')
        previous=config.ppo_mini_batch_size
        try:
            # Preserve ONE optimizer minibatch for the original 32 prompts x 8
            # candidates, irrespective of how many tool turns those chapters needed.
            config.ppo_mini_batch_size=len(batch)//n
            return super()._update_actor(batch)
        finally:
            config.ppo_mini_batch_size=previous


def install_native_trainer():
    upstream.compute_advantage=compute_native_advantage
    from verl.trainer import main_ppo
    main_ppo.RayPPOTrainer=NativePPOTrainer
