"""
hf_generation.py

Local Hugging Face generation backend for the benchmark pipeline.

This is the transformers-based counterpart to the Ollama generate() call in
annotate.py: instead of talking to a local Ollama server over HTTP, it downloads
the model weights straight from the Hugging Face Hub (cached under ~/.cache/
huggingface, or wherever HF_HOME points) and runs generation in-process with
transformers.

Used by hf/annotate.py exactly where ollama/annotate.py uses generate(). Keeping
it in its own module mirrors embeddings.py: the model is expensive to load, so it's
wrapped in a class that loads once and is reused for every row.

Determinism: temperature <= 0 selects greedy decoding (do_sample=False), the
generation analogue of Ollama's temperature=0.0, so tagging is reproducible.

Prompting: annotate.py builds a single flat prompt string. If the tokenizer
carries a chat template (all instruct-tuned models on the Hub do), the prompt is
wrapped as a single user turn via apply_chat_template so the model sees it in the
format it was tuned on; otherwise it's fed as raw text. Only the newly generated
continuation is returned -- the echoed prompt is stripped.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _auto_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(explicit: str | None, device: str) -> "torch.dtype":
    if explicit:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[explicit]
    # bf16 on CUDA/MPS, fp32 on CPU (fp16 on CPU is slow and often unsupported).
    return torch.float32 if device == "cpu" else torch.bfloat16


class HFGenerator:
    """Download + run a Hugging Face causal LM for tagged-text generation.

    Loads tokenizer and model once at construction; call generate() per row.
    """

    def __init__(self, model_name: str, device: str | None = None, dtype: str | None = None,
                 load_in_4bit: bool = False, load_in_8bit: bool = False,
                 max_new_tokens: int = 1024, temperature: float = 0.0):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = _auto_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            # Needed for batched/padded generation and to silence a transformers warning;
            # fall back to EOS when the model ships without a distinct pad token.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {}
        if load_in_4bit or load_in_8bit:
            # 4-/8-bit quantization needs bitsandbytes + accelerate; import lazily so the
            # module still loads (for full-precision use) when they aren't installed.
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
            model_kwargs["device_map"] = "auto"  # accelerate places the quantized weights
        else:
            model_kwargs["torch_dtype"] = _auto_dtype(dtype, self.device)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        # Quantized loads are already placed by device_map="auto"; move only otherwise.
        if not (load_in_4bit or load_in_8bit):
            self.model.to(self.device)
        self.model.eval()

    def _render(self, prompt: str) -> str:
        """Apply the tokenizer's chat template if it has one; else return prompt as-is."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return prompt

    @torch.inference_mode()
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate tagged output for a single prompt, returning only the continuation.

        Retries with backoff for parity with the Ollama path; local generation rarely
        fails transiently, but a transient OOM or CUDA hiccup shouldn't abort the run.
        """
        rendered = self._render(prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)  # greedy == deterministic tagging

        last_err = None
        for attempt in range(max_retries):
            try:
                output_ids = self.model.generate(**inputs, **gen_kwargs)
                # Slice off the echoed prompt so only the model's continuation remains.
                continuation = output_ids[0][prompt_len:]
                return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Generation call failed after {max_retries} retries: {last_err}")
