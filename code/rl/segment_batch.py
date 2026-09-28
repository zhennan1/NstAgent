"""Expand native turns after chapter-level grouping, before actor/reference forward passes."""
import copy
import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from chapter_advantages import chapter_advantages


def expand_native_batch(batch, pad_token_id, group_size=8, multiple=8):
    if 'native_turns' not in batch.non_tensor_batch:
        raise ValueError('Expected native chapter trajectories')
    if 'native_chapter_advantage' in batch.non_tensor_batch:
        raise ValueError('Batch was already expanded')
    records=batch.non_tensor_batch['native_turns']
    trajectory_ids=batch.non_tensor_batch['trajectory_id']
    rewards=batch.batch['rm_scores'].sum(-1).tolist()
    advantages=chapter_advantages([dict(trajectory_id=str(trajectory_ids[i]),
        prompt_uid=str(batch.non_tensor_batch['uid'][i]),reward=rewards[i]) for i in range(len(batch))],group_size)
    rows=[(i,turn,False) for i,turns in enumerate(records) for turn in turns]
    if any(not turns for turns in records) or not rows:
        raise ValueError('Every chapter must preserve at least one generated turn')
    while len(rows)%multiple:
        rows.append((0,records[0][0],True))
    n=len(rows)
    p=batch.batch['prompts'].shape[-1];r=batch.batch['responses'].shape[-1]
    device=batch.batch['prompts'].device
    prompts=torch.full((n,p),pad_token_id,dtype=torch.long,device=device)
    responses=torch.full((n,r),pad_token_id,dtype=torch.long,device=device)
    attention=torch.zeros((n,p+r),dtype=torch.long,device=device)
    masks=torch.zeros((n,r),dtype=torch.long,device=device)
    scores=torch.zeros((n,r),dtype=torch.float32,device=device)
    logprobs=torch.zeros((n,r),dtype=torch.float32,device=device)
    have_logprobs=all(len(t['response_logprobs'])==len(t['response_ids']) for _,t,_ in rows)
    if not have_logprobs and any(t['response_logprobs'] for _,t,_ in rows):
        raise ValueError('Mixed missing/available rollout logprobs')
    adv=[]
    for j,(i,turn,padding) in enumerate(rows):
        a=turn['prompt_ids'];b=turn['response_ids']
        if not 0<len(a)<=p or not 0<len(b)<=r:
            raise ValueError('Native turn would be truncated by PPO padding')
        prompts[j,-len(a):]=torch.tensor(a,device=device)
        responses[j,:len(b)]=torch.tensor(b,device=device)
        attention[j,p-len(a):p+len(b)]=1
        if not padding:
            masks[j,:len(b)]=1
            scores[j,len(b)-1]=rewards[i]
        if have_logprobs:
            logprobs[j,:len(b)]=torch.tensor(turn['response_logprobs'],device=device)
        adv.append(0. if padding else advantages[str(trajectory_ids[i])])
    positions=(attention.cumsum(-1)-1).clamp_min(0)*attention
    old_positions=batch.batch['position_ids']
    if old_positions.ndim==3:
        positions=positions.unsqueeze(1).expand(-1,old_positions.shape[1],-1).clone()
    elif old_positions.ndim!=2:
        raise ValueError('Unsupported position ID layout')
    tensors=dict(prompts=prompts,responses=responses,input_ids=torch.cat((prompts,responses),-1),
                 attention_mask=attention,response_mask=masks,position_ids=positions,rm_scores=scores)
    if have_logprobs:tensors['rollout_log_probs']=logprobs
    indexes=np.array([i for i,_,_ in rows])
    metadata={k:v[indexes] for k,v in batch.non_tensor_batch.items() if k!='native_turns'}
    metadata['native_chapter_advantage']=np.array(adv)
    metadata['native_padding']=np.array([padding for _,_,padding in rows])
    meta=copy.deepcopy(batch.meta_info)
    meta['native_chapter_count']=len(batch)
    meta['native_turn_count']=sum(len(turns) for turns in records)
    return DataProto(batch=TensorDict(tensors,batch_size=[n]),non_tensor_batch=metadata,meta_info=meta)


def apply_native_advantages(batch):
    values=torch.tensor(batch.non_tensor_batch['native_chapter_advantage'],
                        device=batch.batch['response_mask'].device,dtype=torch.float32)
    batch.batch['advantages']=values[:,None]*batch.batch['response_mask']
    batch.batch['returns']=batch.batch['advantages'].clone()
    return batch
