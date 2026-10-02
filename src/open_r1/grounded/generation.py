"""Batched greedy generation for evaluation.

Two backends with the same interface (`generate(list of chat messages) -> list of str`):

* `HFGenerator` runs a Hugging Face model in-process, optionally with a LoRA adapter.
* `OpenAICompatibleGenerator` calls a local OpenAI-compatible chat endpoint, as served
  by vLLM (`vllm serve`) or Ollama. It is used for models that are served separately,
  never for a paid API.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

Messages = list[dict[str, str]]


class Generator(Protocol):
    description: dict

    def generate(self, conversations: list[Messages]) -> list[str]: ...


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str = "auto"):
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16 if device == "mps" else torch.float32


class HFGenerator:
    def __init__(
        self,
        model_name: str,
        adapter: str | None = None,
        batch_size: int = 16,
        max_new_tokens: int = 256,
        dtype: str = "auto",
        device: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or pick_device()
        torch_dtype = pick_dtype(self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.description = {
            "backend": "hf",
            "model": model_name,
            "adapter": adapter,
            "dtype": str(torch_dtype),
            "device": self.device,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "decoding": "greedy",
        }

    def generate(self, conversations: list[Messages]) -> list[str]:
        import torch

        prompts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in conversations
        ]
        # Batch prompts of similar length together to limit padding.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs = [""] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [prompts[i] for i in indices], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            completions = generated[:, batch["input_ids"].shape[1] :]
            for index, text in zip(indices, self.tokenizer.batch_decode(completions, skip_special_tokens=True)):
                outputs[index] = text
        return outputs


class OpenAICompatibleGenerator:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_new_tokens: int = 256,
        concurrency: int = 8,
        timeout: float = 300.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.concurrency = concurrency
        self.timeout = timeout
        self.description = {
            "backend": "openai_compatible",
            "base_url": base_url,
            "model": model_name,
            "max_new_tokens": max_new_tokens,
            "decoding": "temperature=0",
        }

    def _complete(self, messages: Messages) -> str:
        body = json.dumps(
            {"model": self.model_name, "messages": messages, "temperature": 0, "max_tokens": self.max_new_tokens}
        ).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"] or ""

    def generate(self, conversations: list[Messages]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._complete, conversations))
