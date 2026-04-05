"""
llm.py
------
HuggingFace Transformers LLM loader — fully local, no API keys, no server.

SUPPORTED MODELS:
─────────────────────────────────────────────────────────────────────────────
Model ID                                    | RAM    | Quality  | Speed
─────────────────────────────────────────────────────────────────────────────
TinyLlama/TinyLlama-1.1B-Chat-v1.0         | ~2 GB  | ★★☆☆☆   | ⚡ Fastest
microsoft/phi-2                             | ~6 GB  | ★★★☆☆   | Medium
microsoft/Phi-3-mini-4k-instruct            | ~8 GB  | ★★★★☆   | Medium
google/flan-t5-base                         | ~1 GB  | ★★☆☆☆   | ⚡ Fastest
google/flan-t5-large                        | ~3 GB  | ★★★☆☆   | Fast
─────────────────────────────────────────────────────────────────────────────

All models are downloaded on first use and cached in ~/.cache/huggingface.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Model catalogue ────────────────────────────────────────────────────────────
MODEL_CATALOGUE = {
    "TinyLlama-1.1B (Ultra-light, ~2 GB RAM)": {
        "id"      : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "task"    : "text-generation",
        "ram_gb"  : 2,
    },
    "Phi-2 (Balanced, ~6 GB RAM)": {
        "id"      : "microsoft/phi-2",
        "task"    : "text-generation",
        "ram_gb"  : 6,
    },
    "Phi-3-mini (Best quality, ~8 GB RAM)": {
        "id"      : "microsoft/Phi-3-mini-4k-instruct",
        "task"    : "text-generation",
        "ram_gb"  : 8,
    },
    "Flan-T5-Base (Fastest, ~1 GB RAM)": {
        "id"      : "google/flan-t5-base",
        "task"    : "text2text-generation",
        "ram_gb"  : 1,
    },
    "Flan-T5-Large (Fast, ~3 GB RAM)": {
        "id"      : "google/flan-t5-large",
        "task"    : "text2text-generation",
        "ram_gb"  : 3,
    },
}


def get_model_display_names() -> list:
    """Return friendly model names for UI dropdowns."""
    return list(MODEL_CATALOGUE.keys())


def load_huggingface_llm(
    model_key: str = "TinyLlama-1.1B (Ultra-light, ~2 GB RAM)",
    temperature: float = 0.1,
    max_new_tokens: int = 512,
):
    """
    Load a HuggingFace model locally via transformers pipeline.

    Models download on first use and are cached at ~/.cache/huggingface.

    Args:
        model_key      : Friendly name from MODEL_CATALOGUE
        temperature    : Sampling temperature (0 = deterministic)
        max_new_tokens : Max tokens to generate

    Returns:
        dict: { "llm": HuggingFacePipeline, "task": str, "model_id": str }
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline,
    )
    from langchain_huggingface import HuggingFacePipeline

    if model_key not in MODEL_CATALOGUE:
        raise ValueError(
            f"Unknown model: '{model_key}'. "
            f"Choose from: {list(MODEL_CATALOGUE.keys())}"
        )

    cfg      = MODEL_CATALOGUE[model_key]
    model_id = cfg["id"]
    task     = cfg["task"]

    logger.info(f"Loading '{model_id}' (task={task}) …")
    logger.info("First run downloads weights — may take several minutes.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )

    if task == "text2text-generation":
        # Seq2Seq: Flan-T5 family
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )

    else:
        # Causal LM: TinyLlama, Phi-2, Phi-3
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,   # float32 = safest for CPU
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            repetition_penalty=1.15,
            return_full_text=False,      # new text only, not the prompt
        )

    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info(f"'{model_id}' ready ✓")

    return {"llm": llm, "task": task, "model_id": model_id}
