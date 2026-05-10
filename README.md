# Layer Skip Bypass

`layer-skip-bypass` is a research fork of
[Distributed Llama](https://github.com/b4rtaz/distributed-llama) for studying
pipeline-parallel LLM inference with stage-level layer bypass.

The base runtime keeps Distributed Llama's root/worker execution model, model
conversion tools, CPU/Vulkan backends, and tensor-parallel communication. This
fork adds pipeline-parallel instrumentation and an experimental PP stage skip
path that can bypass a selected pipeline stage when a delta-based verifier
accepts the current token.

## Goals

- Reduce decode latency in communication-bound multi-node inference.
- Measure per-stage receive, forward, send, and wall-clock costs.
- Evaluate shadow-only skip decisions before enabling real skip routing.
- Log token-level skip decisions for threshold tuning and safety analysis.

## Repository Layout

- `src/` - C++ runtime, CLI, model execution, networking, and pipeline logic.
- `src/nn/nn-pipeline.*` - stage-to-stage activation transfer and bypass
  marker handling.
- `converter/` - Hugging Face and tokenizer conversion utilities.
- `scripts/` - local experiment and deployment helpers.

## Build

```bash
make dllama
```

Useful test targets:

```bash
make nn-cpu-test
make nn-cpu-ops-test
make nn-topology-test
make nn-pipeline-test
```

The code is intended to remain portable across Linux, macOS, Windows, ARM64,
and x86_64 where the upstream project is supported.

## Basic Inference

Start a worker:

```bash
./dllama worker --port 9999 --nthreads 4
```

Run inference from the root node:

```bash
./dllama inference \
  --model dllama_model_llama3-8b_q40.m \
  --tokenizer dllama_tokenizer_llama3_8B.t \
  --buffer-float-type q80 \
  --prompt "Hello world" \
  --steps 64 \
  --nthreads 4 \
  --collective auto \
  --workers 100.78.3.114:9999
```

For an 8-node run, pass seven worker endpoints after `--workers`.

## Pipeline and Stage-Skip Options

Core distributed options:

| Option | Purpose |
| --- | --- |
| `--collective <auto|star|ring>` | Select tensor-parallel collective policy. |
| `--pipeline-float-type <f32|f16|q40|q80>` | Stage activation transport dtype. Defaults to `--buffer-float-type`. |
| `--pipeline-chunk-bytes <n>` | Split activation payloads into chunks. `0` disables chunking. |
| `--pipeline-delta <0|1>` | Send activation deltas when possible. |
| `--pipeline-delta-min-bytes <n>` | Minimum payload size for delta transport. |
| `--stage-timing <0|1>` | Print per-stage timing for decode steps. |
| `--pp-topk <n>` | Send top-k logits from the last PP stage in the decode fast path. |

Experimental stage-skip options:

| Option | Purpose |
| --- | --- |
| `--pp-stage-skip-shadow <0|1>` | Score skip decisions while still executing the full route. |
| `--pp-stage-skip <0|1>` | Enable real skip routing. |
| `--pp-stage-skip-target <rank>` | PP rank to bypass. Default is `4`. |
| `--pp-stage-skip-theta <f>` | Delta-norm threshold for gate and verifier acceptance. Default is `0.10`. |
| `--pp-stage-skip-verifier <delta>` | Verifier type. Current implementation supports `delta`. |
| `--pp-stage-skip-max-consecutive <n>` | Force a full route after this many consecutive accepts. Default is `3`. |
| `--pp-stage-skip-max-reject-streak <n>` | Dampens skip after repeated rejects. Default is `8`. |
| `--pp-stage-skip-log <0|1>` | Print per-token skip information on the target stage. |
| `--pp-stage-skip-log-file <prefix>` | Append TSV logs as `<prefix>.node<N>.tsv`. |

Recommended workflow:

1. Run baseline with skip disabled.
2. Enable `--pp-stage-skip-shadow 1` and collect token-level logs.
3. Tune `--pp-stage-skip-target` and `--pp-stage-skip-theta` from observed
   accept, reject, fallback, and verifier-cost rates.
4. Enable `--pp-stage-skip 1` only after shadow results are stable.

Example shadow run:

```bash
./dllama inference \
  --model dllama_model_llama3-8b_q40.m \
  --tokenizer dllama_tokenizer_llama3_8B.t \
  --buffer-float-type q80 \
  --prompt "Hello world" \
  --steps 64 \
  --nthreads 4 \
  --collective auto \
  --stage-timing 1 \
  --pp-stage-skip-shadow 1 \
  --pp-stage-skip-target 4 \
  --pp-stage-skip-theta 0.10 \
  --pp-stage-skip-log-file skiplogs/run \
  --workers 100.78.3.114:9999 100.68.147.68:9999
```

Example execute run:

```bash
./dllama inference \
  --model dllama_model_llama3-8b_q40.m \
  --tokenizer dllama_tokenizer_llama3_8B.t \
  --buffer-float-type q80 \
  --prompt "Hello world" \
  --steps 64 \
  --nthreads 4 \
  --collective auto \
  --pp-stage-skip 1 \
  --pp-stage-skip-target 4 \
  --pp-stage-skip-theta 0.10 \
  --workers 100.78.3.114:9999 100.68.147.68:9999
```

## Logs

When `--pp-stage-skip-log-file <prefix>` is set, each node appends a TSV file
with fields such as:

- `run_id`
- `node`
- `pp`
- `pos`
- `target_stage`
- `delta_norm`
- `gate_pass`
- `verifier_score`
- `verifier_pass`
- `was_skip`
- `fallback`
- `consecutive_skip`
- `reject_streak`
- `recv_wait_ms`
- `wall_step_ms`
- `token_id`
- `is_special`
- `token_text`

These logs are intended for threshold selection and for checking whether bypass
decisions stay stable across prompts.

## Model Files

Large converted model files and tokenizer files are ignored by git:

- `dllama_model_*.m`
- `dllama_tokenizer_*.t`

Use `launch.py` or the scripts in `converter/` to download or convert models
locally.

## Notes

- Stage skip is experimental and should be treated as an evaluation feature.
- Shadow mode is the safest default for collecting evidence.
- Worker binaries must match the root binary because protocol fields are shared
  between root and workers.
- This repository inherits the upstream MIT license from Distributed Llama.

## License

MIT. See `LICENSE`.
