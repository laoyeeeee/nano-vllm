# Streaming Generate / Prefix Cache Notes

## 1. Background

Nano-vLLM currently exposes a blocking `LLM.generate()` API. The engine only returns outputs after a sequence is finished:

```python
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

This means sampled tokens are appended to `Sequence` during decode, but callers cannot observe them until the whole completion finishes.

The first target feature is to add token-level streaming output while keeping the existing non-streaming API unchanged.

## 2. Goals

- Add a synchronous streaming API: `LLM.generate_stream(...)`.
- Yield token-level deltas as soon as valid completion tokens are generated.
- Support multiple prompts in the same streaming call.
- Preserve the current `LLM.generate()` return format and behavior.
- Keep existing scheduler, paged KV cache, prefix cache, chunked prefill, CUDA graph, and tensor parallel paths compatible.

## 3. Non-Goals

- No OpenAI-compatible HTTP server in the first version.
- No SSE transport in the first version.
- No asyncio request queue or background async engine in the first version.
- No request cancellation in the first version.
- No tokenizer incremental decoder optimization in the first version.
- No radix-tree prefix cache implementation as part of streaming v1.

## 4. Public API

Add:

```python
for output in llm.generate_stream(prompts, sampling_params):
    print(output.seq_id, output.text, output.finished)
```

Recommended output type:

```python
from dataclasses import dataclass


@dataclass(slots=True)
class StreamOutput:
    seq_id: int
    token_id: int | None
    text: str
    finished: bool
    finish_reason: str | None
```

Field semantics:

- `seq_id`: Sequence id corresponding to the input prompt.
- `token_id`: Newly generated token id for this event.
- `text`: Decoded text delta for this token.
- `finished`: Whether this sequence has finished after this token.
- `finish_reason`: `None`, `"eos"`, or `"length"`.

Proposed method signature:

```python
def generate_stream(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    decode_special_tokens: bool = False,
):
    ...
```

## 5. Expected Behavior

### 5.1 Single Prompt

```python
for out in llm.generate_stream(["Write a haiku about GPUs."], SamplingParams(max_tokens=8)):
    print(out.text, end="", flush=True)
```

Expected behavior:

- The caller receives token deltas during decode.
- The final event for the sequence has `finished=True`.
- The concatenated token ids match the output from `generate()` under equivalent sampling conditions.

### 5.2 Multiple Prompts

```python
prompts = [
    "Write a haiku about GPUs.",
    "List three benefits of prefix caching.",
]

for out in llm.generate_stream(prompts, SamplingParams(max_tokens=32)):
    print(out.seq_id, out.text, out.finished)
```

Expected behavior:

- Outputs from different prompts may be interleaved.
- Each event carries `seq_id`, so callers can route deltas to the correct stream.
- Callers do not need to wait for all prompts to finish before seeing the first tokens.

Example shape:

```text
0 Tiny False
1 1 False
0 cores False
1 Faster False
...
0  True
1  True
```

## 6. Why Existing `generate()` Cannot Do This

Current `generate()` only records outputs when `seq.is_finished` is true. During decode, tokens are generated and appended internally, but intermediate deltas are not exposed.

Limitations of the existing behavior:

- No token-level visibility during generation.
- No typing-effect UI.
- No server-side streaming transport.
- No immediate first-token response.
- Multi-prompt generation can run concurrently internally, but the caller only receives completed sequences.

Streaming fixes this by emitting an event immediately after a new completion token is appended.

## 7. Engine Implementation Plan

### 7.1 Scheduler Postprocess Events

Modify `Scheduler.postprocess(...)` so it returns token events for newly generated completion tokens.

Important behavior:

- During normal prefill, if the final prompt token is processed and a sampled token is appended, return an event.
- During chunked prefill, do not return an event until the prompt is fully cached.
- During preemption re-prefill, do not re-emit already generated completion tokens.
- During decode, return one event per scheduled sequence.

Event data should include:

- `seq_id`
- `token_id`
- `finished`
- `finish_reason`

Finish reason rules:

- `"eos"` when `token_id == eos` and `ignore_eos=False`.
- `"length"` when `num_completion_tokens == max_tokens`.
- `None` otherwise.

### 7.2 `LLMEngine.step()`

Keep `step()` as one scheduler/model/postprocess iteration.

Recommended return:

```python
events, num_tokens = self.step()
```

Where:

- `events` are newly generated completion token events.
- `num_tokens` remains the throughput accounting value.

### 7.3 `generate_stream(...)`

Implementation outline:

```python
def generate_stream(self, prompts, sampling_params, decode_special_tokens=False):
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)

    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    while not self.is_finished():
        events, _ = self.step()
        for event in events:
            text = ""
            if event.token_id is not None:
                text = self.tokenizer.decode(
                    [event.token_id],
                    skip_special_tokens=not decode_special_tokens,
                )
            yield StreamOutput(
                seq_id=event.seq_id,
                token_id=event.token_id,
                text=text,
                finished=event.finished,
                finish_reason=event.finish_reason,
            )
```

### 7.4 Preserve `generate()`

`generate()` should keep returning:

```python
[{"text": "...", "token_ids": [...]}]
```

It can either:

- Continue collecting finished sequence completions, or
- Reuse token events internally and aggregate final outputs.

Prefer reusing token events if it avoids duplicate postprocess logic.

## 8. Prefix Cache Compatibility

Streaming does not conflict with prefix caching.

Prefix caching affects prefill:

```text
prompt tokens -> block hash / prefix cache lookup -> KV reuse
```

Streaming affects decode output:

```text
decode token -> append to sequence -> yield token delta
```

They operate at different stages.

Example:

```python
prompts = [
    "You are a helpful assistant. Answer briefly.\nQuestion: What is KV cache?",
    "You are a helpful assistant. Answer briefly.\nQuestion: What is prefix caching?",
]
```

The shared system prefix can still hit prefix cache during prefill. Streaming only changes when generated completion tokens are returned to the caller.

Required safety rules:

- Do not yield during chunked prefill before a real completion token is produced.
- Do not yield during preemption re-prefill.
- Do not change `BlockManager.allocate(...)` prefix-cache semantics for streaming v1.
- Do not include already streamed completion tokens in prefix cache decisions unless they are part of a later request's actual prompt.

## 9. Testing Requirements

### 9.1 Streaming Correctness

- Single prompt stream returns token events until finished.
- Multi-prompt stream returns interleaved `seq_id` events.
- Aggregated streaming token ids match non-streaming output under equivalent sampling.
- `max_tokens=1` returns one event with `finished=True` and `finish_reason="length"`.
- EOS termination returns `finished=True` and `finish_reason="eos"`.
- `ignore_eos=True` continues until `max_tokens`.

### 9.2 Existing Feature Compatibility

- Chunked prefill does not emit fake stream events.
- Prefix cache hit still produces correct streaming output.
- Preemption re-prefill does not duplicate already emitted tokens.
- CUDA graph decode still works for streaming.
- Tensor parallel smoke test passes when multi-GPU environment is available.

### 9.3 Backward Compatibility

- Existing `example.py` still works.
- Existing `bench.py` still works.
- `LLM.generate()` output schema is unchanged.

## 10. Acceptance Criteria

- `LLM.generate_stream(...)` exists and yields token-level `StreamOutput` events.
- Existing `LLM.generate(...)` behavior is unchanged.
- Multi-prompt streaming can be consumed by grouping events by `seq_id`.
- Streaming does not modify prefix-cache allocation logic.
- No events are emitted for cache-only prefill work.
- Basic examples and benchmark scripts continue to run.
