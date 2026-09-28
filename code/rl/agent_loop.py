"""VeRL adapter: real NstAgent tools and isolated mutable state per GRPO candidate."""
import asyncio
import ast
import contextvars
import copy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentState
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser, FunctionCall
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from longstoryagent_dynamic_tokens import LongStoryAgent, StoryState, ChapterRuntime, chapter_tools, is_done_signal
from reward import score_chapter
from narrative_config import PROFILE, MAX_TOKENS, verify_snapshot

SESSION = contextvars.ContextVar('narrative_session')

class SafeXMLParser(Qwen3XMLToolParser):
    """Keep VeRL's XML extraction, without its eval fallback on model-generated text."""
    async def extract_tool_calls(self,response_ids,tools=None):
        text=self.tokenizer.decode(response_ids,skip_special_tokens=False)
        # Each thinking-enabled assistant prefix ends at <think>. Reasoning
        # tokens remain in the PPO trajectory, but are never executed as tools.
        if '</think>' not in text:
            return text,[]
        visible=text.split('</think>',1)[1]
        return await super().extract_tool_calls(self.tokenizer.encode(visible,add_special_tokens=False),tools)

    def _parse_xml_function_call(self, block, tools):
        name, parameters = block.split('>', 1)
        schema = next((t.function.parameters.properties for t in tools if t.function.name == name), {})
        args = {}
        for match in self.tool_call_parameter_regex.findall(parameters):
            key, raw = (match[0] or match[1]).split('>', 1)
            kind = schema[key].type if key in schema else 'string'
            if kind == 'string':
                value = raw.removeprefix('\n').removesuffix('\n')
            else:
                try: value = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    try: value = ast.literal_eval(raw)
                    except (SyntaxError, ValueError): value = raw
            args[key] = value
        return FunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False))

class NarrativeOpsLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.apply_chat_template_kwargs.get('enable_thinking') is not True:
            raise ValueError('Narrative GRPO requires enable_thinking=True')
        self.tool_parser = SafeXMLParser(self.tokenizer)

    async def run(self, sampling_params, **kwargs):
        snapshot=json.loads(Path(kwargs['extra_info']['snapshot_path']).read_text())
        verify_snapshot(snapshot)
        agent=LongStoryAgent(None,'Qwen3.5-4B',MAX_TOKENS,.7,snapshot['length'],logging.getLogger(__name__),**PROFILE)
        session=dict(snapshot=snapshot,agent=agent,chapters=copy.deepcopy(snapshot['chapters']),
                     state=StoryState.from_dict(copy.deepcopy(snapshot['state'])),
                     runtime=ChapterRuntime(snapshot['k'],snapshot['outline'][snapshot['k']]['name']),
                     updates=[],done=False,lock=asyncio.Lock())
        token=SESSION.set(session)
        try:
            output=await super().run(sampling_params,**kwargs)
            runtime=session['runtime']
            successful=[u for u in session['updates'] if u['result'].get('ok')]
            complete=session['done'] and runtime.write_succeeded and bool(successful)
            chapter=runtime.written_chapter or runtime.last_rejected_draft
            audit=os.environ['NARRATIVE_AUDIT_DIR']
            scores=await score_chapter(snapshot,chapter,session['updates'],session['state'].to_dict(),complete,audit+'/judges')
            output.reward_score=scores['score']
            output.extra_fields['reward_extra_info']=scores
            path=Path(audit)/'rollouts'/f'{uuid4().hex}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(sample=kwargs['extra_info'],scores=scores,chapter=chapter,
                                           trace=runtime.tool_trace,updates=session['updates'],
                                           post_state=session['state'].to_dict()),ensure_ascii=False))
            return output
        finally:
            SESSION.reset(token)

    async def _handle_pending_state(self, data, sampling_params):
        s=SESSION.get(); snapshot=s['snapshot']
        schemas=chapter_tools(snapshot['outline'][snapshot['k']]['word_count'],'en',PROFILE['length_control_mode'],'narrative_ops')
        data._active_tool_schemas=schemas
        data._active_tools={t['function']['name']:SimpleNamespace(tool_schema=OpenAIFunctionToolSchema.model_validate(t)) for t in schemas}
        state=await super()._handle_pending_state(data,sampling_params)
        if len(data.prompt_ids)>self.prompt_length:
            raise ValueError(f'Tool-augmented prompt exceeds {self.prompt_length} tokens: {len(data.prompt_ids)}')
        return state

    async def _handle_generating_state(self,data,sampling_params,ignore_termination=False):
        remaining=self.response_length-len(data.response_mask)
        if remaining<=0:
            return AgentState.TERMINATED
        s=SESSION.get();snapshot=s['snapshot']
        from longstoryagent_dynamic_tokens import requires_dialogue_only
        budget=s['agent']._chapter_max_tokens(snapshot['outline'][snapshot['k']]['word_count'],'en',requires_dialogue_only(snapshot['prompt']))
        sampling_params=dict(sampling_params,max_tokens=min(budget,remaining))
        state=await super()._handle_generating_state(data,sampling_params,ignore_termination)
        if state==AgentState.TERMINATED:
            text=self.tokenizer.decode(data.response_ids,skip_special_tokens=False)
            text=text.split('</think>',1)[-1].replace('<|im_end|>','').strip()
            SESSION.get()['done']=is_done_signal(text)
        return state

    async def _call_tool(self,tool_call,tools_kwargs,data):
        s=SESSION.get();r=s['runtime'];snapshot=s['snapshot']
        name=tool_call.name
        args=tool_call.arguments
        if isinstance(args,str):
            try: args=json.loads(args)
            except json.JSONDecodeError: args={}
        # Tool calls in one assistant message must execute in emitted order. The parent
        # gathers them concurrently; the lock serializes dispatch and every mutation.
        async with s['lock']:
            result=s['agent']._dispatch_tool(name,args,snapshot['k'],s['chapters'],s['state'],r.read_used,r.search_used,
                    r.written_chapter,r.correct_used,r.write_succeeded,snapshot['outline'][snapshot['k']]['word_count'],
                    'en',r.current_chapter_corrected,snapshot['prompt'])
            r.record_tool_result(data.assistant_turns,name,args,result)
            if name=='update':
                s['updates'].append(dict(arguments=args,result=result))
            return ToolResponse(text=json.dumps(result,ensure_ascii=False)),None,{}
