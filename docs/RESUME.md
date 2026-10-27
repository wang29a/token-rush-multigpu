# Resume on a fresh instance

What survives, what has to be rebuilt, and how to continue without the old
chat. Written 2026-10-27 at the S6 stop checkpoint.

## What survives what

| thing | stop / start | recycle or destroy |
|---|---|---|
| container filesystem (venv, `/tmp`, `/root`) | survives | **lost** |
| chat session store `/root/.dsh/sessions` (~2 MB) | survives | **lost** |
| `/workspace` (repo checkout, bundles) | survives | **lost**: it is not a volume |
| GitHub `wang29a/token-rush-multigpu` | survives | survives |
| model weights `/tmp/qwen3-8b` (16 GB) | survives | lost, re-download |

One check: `vast-capabilities | jq .instance.workspace_is_volume` → `false` on
this box, so nothing local is durable.

## If the chat itself must be kept

The transcript lives on this machine, not in the repository:
`/root/.dsh/sessions/` (about 2 MB). Before destroying the instance, copy it
off-box (for example `scp -r` or a download from the GUI). It contains the whole
conversation; keep it out of the public repository. Keeping the instance
**stopped** instead of destroyed preserves the session, the venv and the weights
as they are; storage is still billed while stopped.

The repository does not need the transcript: `HANDOFF.md` and `lab/LOG.md` carry
the state and the next action, which is why they are written for a reader with
no prior conversation.

## Rebuild on a new box

The repository is public, so a fresh instance can clone it with no credentials:

```bash
git clone https://github.com/wang29a/token-rush-multigpu.git
cd token-rush-multigpu
uv venv .venv --python /venv/main/bin/python
uv pip install --python .venv/bin/python torch==2.11.0 numpy \
  --index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python huggingface_hub safetensors tokenizers regex
.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.device_count())"
```

`transformers` is needed only for the independent reference check
(`lab/05b_hf_reference.py`). Dependency resolution pulled a second torch build
when it was installed here, so fetch the wheel and install it with `--no-deps`
(plus `regex`, which it needs): the wheel name must include the version, and a
truncated download fails with "unable to locate the end of central directory".

Weights are public and do not need a token:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", local_dir="/tmp/qwen3-8b",
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*",
                                  "merges.txt", "vocab.json"])
PY
```

`/tmp` is cleared on restart, so weights are re-downloaded per instance
(~16.4 GB). Keep them out of git.

Pushing needs write credentials. The deploy key that pushed here lives on this
instance and dies with it: on the new box generate a new key and add it under
Settings -> Deploy keys with write access, or connect the GitHub plugin first.
Until then, treat the clone as read-only and keep work committed locally in a
bundle.

## Check that the state came back

```bash
git log --oneline -1
ls results/2026-10-27-vast-57952342/{s1,s2,s3,s4,s6}
.venv/bin/python lab/report_s1.py results/2026-10-27-vast-57952342
.venv/bin/python lab/report_s2.py results/2026-10-27-vast-57952342/s2
.venv/bin/python lab/report_s2_sync.py results/2026-10-27-vast-57952342/s2-sync
.venv/bin/python lab/report_s3.py results/2026-10-27-vast-57952342/s3
.venv/bin/python lab/report_s4.py results/2026-10-27-vast-57952342/s4
.venv/bin/python lab/report_s6.py results/2026-10-27-vast-57952342/s6
```

The generators are order-independent and deterministic; if the files change,
something in the environment differs from the recorded one. This drill was run
on 2026-10-27: a fresh clone of `main` regenerated all six reports and both
summary files with an empty `git status`.

## What to do first on resume

Follow the stop checkpoint in `HANDOFF.md`: S6's speed ratio is recorded
(TP=1 80.3 → TP=2 117.6 tok/s) but its correctness is **not established**, and
the saved-token comparison has a diagnostic defect (eight warmup tokens are not
saved, and the generator reports total agreement instead of a contiguous
prefix). Fix token recording and the prefix gate, then compare TP=1, TP=2 and
the Hugging Face reference on one forced token sequence before any precision
change. Do not start M1 and do not quote a production speed number.

Hardware matters for comparability: the recorded results are from 2 × RTX 5090,
PCIe Gen5 x16, `NODE` topology, runtime peer access false, ~400 W. A different
box is a different baseline; re-run the relevant stage rather than mixing.
