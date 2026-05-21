import os
from collections import defaultdict

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    completions = defaultdict(list)
    for output in llm.generate_stream(prompts, sampling_params):
        completions[output.seq_id].append(output.text)
        print(f"[{output.seq_id}] {output.text}", end="", flush=True)
        if output.finished:
            print(f"\n[{output.seq_id}] finished: {output.finish_reason}")

    for seq_id in sorted(completions):
        print(f"\nCompletion {seq_id}: {''.join(completions[seq_id])!r}")


if __name__ == "__main__":
    main()
