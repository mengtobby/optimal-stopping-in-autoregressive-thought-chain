# Run the real-data pipeline on free cloud GPUs

Nothing in this guide requires payment, a paid subscription, or a local model run.
Use Kaggle first and free Colab when Kaggle GPU capacity is unavailable. The laptop
only uploads notebooks and downloads reports. Python 3.11 or newer is required.

## Kaggle: unattended primary route

1. Create a free Kaggle account. Complete phone verification in account settings
   if Kaggle requests it before enabling accelerators or Internet.
2. Create a notebook and import `notebooks/kaggle_run.ipynb` from this repository.
3. In notebook Settings / Session options, enable **Internet** and select
   **GPU T4 x2**. P100 also works when available. Run the `nvidia-smi` cell to
   confirm a cloud GPU; stop if it fails. The notebook uses one GPU for the 1.5B
   model, so it does not assume that two T4s form a single larger device.
4. Review the workdir `/kaggle/working/osc-runs/math500`. The setup clones the
   public repository and installs `pip install -e '.[model]'` on the cloud VM.
5. Select **Save Version > Save & Run All (Commit)** to execute every cell in a
   fresh background session. You can close the browser after starting the commit.
   Files created only in an earlier interactive session are not automatically
   copied into that fresh session.
6. The defaults use the 1.5B model, math500, 100 problems, an 11-hour total budget,
   float16, and reduced sampling/token limits. Check your account's current
   session limit first: some sessions allow 12 hours, while Kaggle's GPU tips
   currently describe 9-hour sessions. Lower `--max-hours` below the displayed
   limit (for example 8) when needed. Reserve time for installation and downloads.
7. After completion, open the saved version's **Output** files and download
   `osc-runs/math500/results.zip`. Also preserve the full workdir to resume traces.
   The notebook displays `SUMMARY.md`, including failed or budget-skipped stages.

Kaggle generally provides about **30 GPU-hours per week**, subject to account
availability and the quota displayed in Settings. Stop unused interactive GPU
sessions. See the official [notebook guide](https://www.kaggle.com/docs/notebooks)
and [GPU quota tips](https://www.kaggle.com/page/GPU-tips-and-tricks).

## Colab: free fallback

1. Upload `notebooks/colab_run.ipynb` to free Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** (subject to availability).
   Run `nvidia-smi`; do not continue with a local runtime or CPU runtime.
3. Set `USE_DRIVE = True` in the persistence cell to mount Google Drive and use
   `/content/drive/MyDrive/osc-runs/math500`. Approve the Drive mount in Colab.
   With `False`, outputs live under `/content/osc-runs/math500` and disappear
   when the VM is reclaimed. Mounting Drive initially requires interaction.
4. Run all cells. The defaults use 30 math500 problems and a 3-hour budget with
   the same small model and float16 settings.
5. Read the summary and download `results.zip` from the Files sidebar or Drive.
   Retain the full Drive workdir for resume.

Free Colab does not guarantee a T4, runtime duration, or unattended completion.
Idle disconnects and resource limits still apply while a browser tab is open.
Do not use artificial keep-alive scripts; use persisted shards and restart after
interruptions. See the official [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Budgets, stages, and expected time

The notebook command is equivalent to this command **inside the cloud VM**:

```bash
python scripts/run_pipeline.py --workdir /kaggle/working/osc-runs/math500 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dataset math500 \
  --limit 100 --max-hours 11 --dtype float16 \
  --n-forced 2 --k-siblings 2 --sibling-stride 4 --sibling-max-tokens 128 \
  --max-thinking-tokens 2048 --max-steps 16
```

Order is collect, headroom, elicit, flips. Collection and elicitation use SDPA; only the flips pilot loads eager attention
for attention probes. Each backend uses the requested dtype. Use
`--score-chunk-size` (default 8) to tune elicitation candidate batch size. `auto`
selects a suitable dtype through HFBackend; notebooks explicitly use float16 for
T4/P100. Headroom defaults to lambda 0, 0.05, 0.1 (`--lam` accepts a list).

Plan for hours for collection and scoring, seconds to minutes for headroom, plus
several minutes for installation and the initial model download. These are planning
estimates, not measured benchmarks. Sequence lengths, forced answers, and sibling
sampling dominate runtime; eager attention also increases pilot memory use.
Start with `--limit 5 --max-hours 0.5` to measure your workload before a large run.

The total budget starts before problem loading. Each stage subprocess has a hard
timeout equal to the remaining budget. Collection's soft deadline reserves the
larger of **30 minutes or 25% of the remaining budget** for the in-flight problem,
merging completed shards, and downstream stages. Adjust the minimum with
`--collect-reserve-minutes` if measured per-problem runtimes require more margin.
For example, with 11 hours remaining, collection stops starting new problems
after 8.25 hours, leaving 2.75 hours for finishing, merging, and pilots. If the
remaining budget is smaller than the reserve, the soft collection window is
floored at one second (or 10% of the remaining time when below ten seconds).
The collector checks this deadline between problems; a problem taking longer
than the reserved margin can still hit the hard timeout. Later stages
are attempted after failures while time remains. If collection consumes the
budget, pilots wait for a later invocation; free sessions cannot promise that all
stages finish in one run. Setup time before the runner is outside its budget.
A nonzero exit status means at least one stage failed or ran out of time.

## Resume and retrieve results

Rerun the same command or notebook with the same workdir. Collection resumes from
completed `traces/shards` until every ID in `problems.jsonl` has a shard, even when
`traces/merged` already contains a subset. The runner also validates that the merged
trace set is readable and contains every expected ID; an interrupted or stale merge
is rebuilt from the completed shards. No deletion or manual cleanup is needed.
`SUMMARY.md` reports collected X of Y problems. Reports record their merged trace
count in `.resume.json` sidecars and rerun automatically when that count changes;
unchanged reports are skipped. Reports without sidecars are regenerated once.

`problems.jsonl` is retained, including with `--force`, to keep prompts paired with
existing traces. `--force` reruns selected stages; for collection it disables shard
reuse. To regenerate reports explicitly, use
`--stages headroom,elicit,flips --force`. To replace the saved problem list, select
collection and pass `--refresh-problems` (normally together with `--force` to
recollect). This flag has no effect when collection is not selected. Use a fresh
workdir when changing model, dataset, problem limit, seed, or collection settings
to avoid mixing experiments.

On Kaggle, attach a previous version's output as an input dataset, then copy its
**entire workdir** from the read-only `/kaggle/input/...` into `/kaggle/working/...`
before the run cell. On Colab, remount the same Drive directory. `results.zip`
contains reports and summary, not the trace shards needed to resume collection.
The runner writes `SUMMARY.md` and `results.zip` even after stage failures; a
platform killing the entire VM cannot guarantee those final writes.

Download the archive, extract it under your local repository's `results/`
(which is gitignored), for example `results/cloud-math500/`. No local model
execution is needed. Copy `SUMMARY.md` into `docs/` if you want a versioned report;
review its experiment settings and status before sharing it.
