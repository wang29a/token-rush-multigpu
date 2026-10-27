#!/usr/bin/env python3
"""Independent reference: Hugging Face Qwen3 on CPU, last-prompt-token logits.

One forward pass, no generation. Used only to check that lab/05_tp2_decode.py
implements the architecture correctly; not a performance baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/tmp/qwen3-8b")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1].float().numpy()
    np.save(out / "hf_logits.npy", logits)
    (out / "ids.json").write_text(json.dumps(ids) + "\n")
    print(json.dumps({"prompt_ids": ids, "argmax": int(logits.argmax()),
                      "logits_shape": list(logits.shape), "finite": bool(np.isfinite(logits).all())}))


if __name__ == "__main__":
    main()
