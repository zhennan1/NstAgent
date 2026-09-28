"""Per-turn native rollout. Requires a chapter-aware segment-expanding trainer."""
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.tools.schemas import OpenAIFunctionToolSchema
from agent_loop import NarrativeOpsLoop
from native_rollout import NativeChapterRollout
from reward import score_chapter
from narrative_config import verify_snapshot
from smoke_replay import take_smoke_replay


class SegmentedNarrativeOpsLoop(NarrativeOpsLoop):
    async def run(self, sampling_params, **kwargs):
        if os.environ.get('NARRATIVE_SEGMENTED_TRAINER') != '1':
            raise RuntimeError('Native per-turn outputs require the segment-expanding trainer; stock PPO would discard turns')
        snapshot=json.loads(Path(kwargs['extra_info']['snapshot_path']).read_text())
        verify_snapshot(snapshot)
        replay=take_smoke_replay(kwargs['extra_info'])
        if replay is not None:
            replay['sample']=dict(kwargs['extra_info'])
            path=Path(os.environ['NARRATIVE_AUDIT_DIR'])/'rollouts'/f"{replay['trajectory_id']}.json"
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(replay,ensure_ascii=False))
            return self._chapter_output(replay,replay['scores'],replay['trajectory_id'],replayed=True)
        trajectory_id=uuid4().hex

        async def generate(messages, schemas, budget):
            # vLLM's OpenAI frontend performs this normalization before rendering;
            # direct tokenizer calls must do the same on our private history copy.
            for message in messages:
                for call in message.get('tool_calls') or []:
                    function=call['function']
                    if isinstance(function.get('arguments'),str):
                        function['arguments']=json.loads(function['arguments'])
            prompt_ids=await self.apply_chat_template(messages,tools=schemas)
            if len(prompt_ids)>self.prompt_length or budget>self.response_length:
                raise ValueError(f'Native per-turn context exceeds training allocation: prompt={len(prompt_ids)}, response_budget={budget}')
            params=dict(sampling_params,max_tokens=budget)
            if self.tool_parser.stop_token_ids:
                params['stop_token_ids']=list(set(params.get('stop_token_ids',[])+self.tool_parser.stop_token_ids))
            response=await self.server_manager.generate(request_id=uuid4().hex,prompt_ids=prompt_ids,
                sampling_params=params,image_data=None,video_data=None,audio_data=None,mm_processor_kwargs={})
            ids=list(response.token_ids)
            text=self.tokenizer.decode(ids,skip_special_tokens=False)
            schema_models=[OpenAIFunctionToolSchema.model_validate(t) for t in schemas]
            _,calls=await self.tool_parser.extract_tool_calls(ids,schema_models)
            reasoning,sep,visible=text.partition('</think>')
            if not sep:
                # Unfinished thinking is not visible prose or an executable tool call.
                visible=''
            visible=visible.replace('<|im_end|>','').strip()
            if calls:
                visible=re.sub(r'<tool_call>.*?</tool_call>','',visible,flags=re.DOTALL).strip()
            message=SimpleNamespace(content=visible,reasoning_content=reasoning,
                tool_calls=[SimpleNamespace(id=uuid4().hex,type='function',
                    function=SimpleNamespace(name=c.name,arguments=c.arguments)) for c in calls])
            return dict(prompt_ids=prompt_ids,response_ids=ids,response_logprobs=response.log_probs,
                message=message,finish_reason='length' if len(ids)>=budget else ('tool_calls' if calls else 'stop'))

        audit=Path(os.environ['NARRATIVE_AUDIT_DIR'])
        runner=NativeChapterRollout(snapshot,generate)
        trace_dir=audit/'native_failures'/trajectory_id
        trace_dir.mkdir(parents=True,exist_ok=True)
        result=await runner.run_chapter(trace_dir=str(trace_dir))
        scores=await score_chapter(snapshot,result['chapter'],result['updates'],result['post_state'],
                                   result['complete'],audit/'judges')
        path=audit/'rollouts'/f'{trajectory_id}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(sample=kwargs['extra_info'],scores=scores,
            trajectory_id=trajectory_id,**result),ensure_ascii=False))
        return self._chapter_output(result,scores,trajectory_id)

    def _chapter_output(self,result,scores,trajectory_id,replayed=False):
        first=result['turns'][0]
        # A representative row keeps the 32 x 8 chapter batch intact until the trainer
        # computes chapter advantages and explicitly expands ALL native turns.
        return AgentLoopOutput(prompt_ids=first['prompt_ids'],response_ids=first['response_ids'],
            response_mask=[1]*len(first['response_ids']),response_logprobs=first['response_logprobs'] or None,
            num_turns=len(result['turns']),metrics={},reward_score=scores['score'],
            extra_fields=dict(native_turns=result['turns'],trajectory_id=trajectory_id,
                              native_replayed=replayed,reward_extra_info=scores))
