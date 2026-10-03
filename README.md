# kolibri-mlx

MLX port of Aleph Alpha's Kolibri 1 (`Aleph-Alpha/Kolibri-1`) for mlx-lm, with a converter, a torch fp32 reference and verification scripts.

## Status

The 4-bit model loads with mlx-lm 0.32.0 on an Apple M4 Pro with 64 GB. It takes 44 GB on disk and 44.1 GB peak memory at load. Generation runs at about 50 tok/s. The OpenAI-compatible server works with reasoning on and off and parses a tool call. No upload, PR or other public action has been done.

Verification verdict as currently known:

* The model code is exact. Teacher-forced, layer by layer against the fp32 torch reference: relative error at most 6.8e-6 (median 3.6e-7), expert sets identical on 100.0 % of tokens (2 of 64,100 token-layer pairs differ, both exact ties).
* End to end, the 4-bit model is noticeably worse than the reference. Top-1 agreement per prompt is 0.54 to 1.00 and KL (mean) is 0.08 to 1.05, see section 5c.
* Even unquantised bf16 drifts from the fp32 reference when the layers are chained. Top-1 is 0.77 to 1.00 and KL (mean) 0.0007 to 0.49, see section 5b.

### Verdict (to be finalised)

{{VERDICT}}

## 1. Quick start

Needs `uv`, Python 3.12, about 78 GB of disk for the FP8 source and 44 GB for the 4-bit output.

```bash
uv sync

# 1. download the FP8 checkpoint (78 GB) into the Hugging Face cache
hf download Aleph-Alpha/Kolibri-1

# 2. convert to 4-bit MLX (about 3 minutes, peak MLX memory 8.75 GB)
uv run python scripts/convert.py            # writes models/Kolibri-1-4bit

# 3. generate (recommended sampling from Aleph Alpha)
uv run python generate.py --model models/Kolibri-1-4bit \
  -p "Erkläre in zwei Sätzen, warum der Himmel blau ist." -m 400 \
  --temp 1.0 --top-p 0.97 --top-k 128

# 4. serve (OpenAI-compatible)
uv run python serve.py --model models/Kolibri-1-4bit --port 8080
```

Reasoning is switched through the chat template. Pass `reasoning_effort` (`none`, `low`, `medium`, `high`) or `enable_thinking` as `chat_template_kwargs`:

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Erkläre in zwei Sätzen, warum der Himmel blau ist."}],
  "chat_template_kwargs": {"reasoning_effort": "low"},
  "temperature": 1.0, "top_p": 0.97, "top_k": 128, "max_tokens": 400
}'

# reasoning off
#   "chat_template_kwargs": {"enable_thinking": false}
```

With reasoning on, the reply carries the thinking in `message.reasoning`. Tool calls (Hermes style) come back in `message.tool_calls`.

Smoke test (starts its own server, writes `verify/out/server-smoke.json`):

```bash
uv run python scripts/smoke_server.py --model models/Kolibri-1-4bit
```

### Why mlx-lm is not patched

mlx-lm resolves `model_type` with `importlib.import_module("mlx_lm.models.<type>")`. That call returns an existing `sys.modules` entry first. `kolibri_mlx/register.py` inserts our model file as `sys.modules["mlx_lm.models.kolibri1"]`. `generate.py` and `serve.py` import it before mlx-lm loads the model. In your own code, run `import kolibri_mlx.register` before `mlx_lm.load`.

`--trust-remote-code` is not needed. mlx-lm never executes code from the model directory. The tokenizer is a plain `tokenizer.json`.

## 2. What was ported

Source of truth: Aleph Alpha's vLLM plugin (`aleph_alpha_inference/kolibri1.py`). No transformers or mlx-lm implementation existed on 2026-10-03.

| Item | Value |
|---|---|
| Layers | 50 (4 sliding : 1 full, from `layer_types`) |
| Hidden size | 2560 |
| Attention | 48 heads, 4 KV heads, head_dim 128, per-head q/k RMSNorm |
| Sliding window | 513 |
| RoPE | NeoX style (`nn.RoPE(traditional=False)`), base 10000, sliding layers only |
| Experts | 384, 6 routed per token, plus 1 shared; expert width 512 |
| Router | fp32 logits, top-k on logits + `expert_bias`, weights = sigmoid(logits), `norm_topk_prob=false` |
| Vocabulary | 128000, untied `lm_head` |
| Checkpoint | FP8 `float8_e4m3fn`, 128x128 block scales (BF16 repo also exists) |

What the model file (`kolibri_mlx/models/kolibri1.py`) does differently from `qwen3_moe`:

* RoPE only in sliding-window layers. Full-attention layers have no positional encoding at all.
* Sliding layers use `RotatingKVCache(max_size=513, keep=0)`. Full layers use `KVCache`. Masks use `window_size=513`.
* The router runs in fp32 on an fp32 copy of the input. Top-k is taken on `logits + expert_bias`. The weights are the sigmoid of the unbiased logits at the chosen experts, not renormalised.
* The shared expert is ungated. Its output is added to the routed sum.
* Sandwich norms: `post_attn_norm` on the attention output and `post_ffn_norm` on the MoE output, both before the residual add. `post_attention_layernorm` is the pre-MoE norm (Qwen naming).
* The router gate (`mlp.gate`) is never quantised (`quant_predicate`). `expert_bias` is kept in fp32.
* `sanitize` maps `moe.router.expert_bias` to `mlp.expert_bias` and stacks the per-expert weights into `switch_mlp`.

## 3. Conversion

`scripts/convert.py` reads the FP8 checkpoint from the Hugging Face cache and writes an mlx-lm directory.

* FP8 block dequantisation: `float(w) * weight_scale_inv[i // 128, j // 128]` (`kolibri_mlx/fp8.py`). It is multiplied, not divided.
* Validation against `Aleph-Alpha/Kolibri-1-BF16`, shard 1 (`scripts/check_dequant.py --shard 1`): 1362 FP8 tensors, worst max-abs error relative to the tensor's max is 0.0357 (FP8 e4m3 has 3 mantissa bits, so up to 1/16 is expected). All 13 bf16 tensors (embeddings, `lm_head`, norms, gate, `expert_bias`) are bit-identical. Only shard 1 was checked.
* Streaming: one decoder layer at a time is dequantised in torch, stacked into the MLX layout, quantised with `nn.quantize` and appended to the shard writer. The full model is never in memory.
* Cost of the 4-bit run: 194.8 s, peak MLX memory 8.75 GB, process RSS under 2 GB.
* Output: 9 safetensors shards (at most 5 GB each), `model.safetensors.index.json`, `config.json` with a `quantization` block, tokenizer files, `LICENSE`. 44.0 GB on disk.
* Quantisation: affine, 4 bit, group size 64, applied to attention, routed and shared experts, embeddings and `lm_head`. Router gates and norms stay bf16, `expert_bias` stays fp32.

| Flag | Default | Meaning |
|---|---|---|
| `--hf-path` | `Aleph-Alpha/Kolibri-1` | source repo or directory |
| `--mlx-path` | `models/Kolibri-1-4bit` | output directory (refuses if shards exist) |
| `--bits` | 4 | 2, 3, 4, 5, 6 or 8 |
| `--group-size` | 64 | quantisation group size |
| `--no-quant` | off | write bf16 without quantisation |
| `--layers N` | all | debug: convert only the first N layers |
| `--dry-run` | off | print the layer 0 tensor layout and stop |
| `--log` | off | also log to `run/convert-<timestamp>.out` |

## 4. Verification

### Method

* Ground truth is a pure-PyTorch fp32 CPU forward built from the vLLM plugin's math (`kolibri_mlx/reference_torch.py`). `scripts/reference_run.py` runs it layer by layer with streamed, dequantised weights and saves every layer's residual stream, expert ids and expert weights, plus the final logits, to `verify/out/reference/`. 50 layers, five prompts, about 2.7 s per layer, peak RSS 26.2 GiB.
* Two independent implementations of the reference math exist: `kolibri_mlx/reference_torch.py` and a second one inside `tests/test_kolibri1.py` (tiny random model). Both were written from the same trace of the plugin by separate subagents.
* `scripts/verify_layerwise.py` compares the MLX decoder layer with the reference residual stream. Teacher-forced: each layer gets the reference's previous output as input. `--chain`: each layer gets the MLX layer's own previous output, starting from the real embedding, and the final logits are compared too.
* `scripts/verify_e2e.py` loads the converted model with mlx-lm, runs a prefill and compares the logits with the reference logits.
* Metrics: `rel` is the relative error of the layer output against the reference; `top1` is the share of positions with the same argmax; `top5` is the mean overlap of the top-5 sets; KL is KL(reference || MLX) per position.

Prompts (`verify/prompts.py`, raw text without chat template, no BOS token is added):

| Name | Tokens | Content |
|---|---|---|
| `de_short` | 13 | German factual sentence (Swiss cities) |
| `en_short` | 15 | English sentence (fox, capital of France) |
| `de_code` | 24 | German instruction plus Python stub |
| `en_long` | 618 | English text on the history of the bicycle |
| `de_long` | 612 | German text on the history of the bicycle |

`en_long` and `de_long` exceed the 513-token window, so the window boundary and cache rotation are exercised.

### Sliding-window semantics

Both sides were checked. vLLM 0.29 (flash-attn backend) passes `window_size=(sliding_window - 1, 0) = (512, 0)`, so a query at position i sees keys `i-512 <= j <= i`: 513 keys including itself. mlx-lm `create_causal_mask(window_size=W)` allows `i-W < j <= i`: W keys including itself. Therefore `window_size = 513` and `RotatingKVCache(max_size=513)`. Two of the five prompts are longer than 513 tokens. The tests also check the window on a tiny model (window 7: position 6 sees token 0, position 7 does not).

### 5a. Teacher-forced, layer by layer (250 rows: 50 layers x 5 prompts)

| MLX dtype | max rel | median rel | expert-set agreement | notes |
|---|---|---|---|---|
| float32 | 6.76e-6 | 3.64e-7 | 100.0 % of tokens (2 of 64,100 token-layer pairs differ) | passes the 1e-3 bound. The 2 tokens are in layer 0 of `en_long`: exact ties at fp32 noise. |
| bfloat16 | 0.0382 | 0.00417 | 98.4 % mean over rows, minimum 80 % (13-token prompt) | one row above the 2e-2 bound: layer 35, `de_long`. |

The float32 row proves that the MLX model implements the reference math. In bf16, the worst row (layer 35, `de_long`) is one token with max-abs error 661 at a massive-activation token where an expert flipped. Its mean-abs error is 0.0377 and the cosine similarity of the last position is 1.0 to 6 digits.

### 5b. Chained bf16 (no teacher forcing, unquantised weights)

Final logits against the reference. Source: `verify/out/chain-bf16.json`.

| Prompt | Top-1 | Top-5 | KL mean | KL p95 | KL max |
|---|---|---|---|---|---|
| `de_short` (13) | 0.769 | 0.800 | 0.490 | 2.13 | 3.22 |
| `en_short` (15) | 0.933 | 0.960 | 0.00572 | 0.0215 | 0.0504 |
| `de_code` (24) | 1.00 | 0.950 | 0.000667 | 0.00303 | 0.00390 |
| `en_long` (618) | 0.968 | 0.957 | 0.00820 | 0.0220 | 1.55 |
| `de_long` (612) | 0.931 | 0.906 | 0.0359 | 0.115 | 2.18 |

Expert-set agreement over all layers and tokens drops to 81.2 % when the layers are chained (per prompt: 55.8 %, 78.7 %, 90.1 %, 84.7 %, 78.0 %).

### 5c. 4-bit end to end (converted model, prefill)

Source: `verify/out/e2e-Kolibri-1-4bit.json`. The on-the-fly quantised chain run (`verify/out/chain-4bit.json`, bits 4, group 64) gives the same logit numbers to three digits, so the converter output equals in-process quantisation of the same weights.

| Prompt | Top-1 | Top-5 | KL mean | KL p95 | KL max |
|---|---|---|---|---|---|
| `de_short` (13) | 0.538 | 0.708 | 1.05 | 2.14 | 2.18 |
| `en_short` (15) | 1.00 | 0.880 | 0.0938 | 0.391 | 1.08 |
| `de_code` (24) | 0.917 | 0.875 | 0.0846 | 0.0794 | 1.57 |
| `en_long` (618) | 0.905 | 0.872 | 0.0796 | 0.272 | 2.66 |
| `de_long` (612) | 0.845 | 0.815 | 0.154 | 0.568 | 4.87 |

Other quantisation variants, chained layer by layer (`scripts/verify_layerwise.py --chain`):

| Variant | `de_short` top-1 / KL | `en_short` | `de_code` | `en_long` | `de_long` |
|---|---|---|---|---|---|
| 8-bit | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}} |
| mixed (attn 8, experts 4) | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}} | {{TBD}} |

The first attempts of both runs failed with a script error (`Checkpoint(None)`, no `--checkpoint` given). `verify/out/chain-8bit.json` and `chain-mixed-a8e4.json` do not exist yet. An 8-bit model is about twice the size of the 4-bit one and cannot be loaded for generation on 64 GB, so these variants can only be checked layer by layer.

### 5d. Position bands (4-bit, prefill KL mean / top-1; decode path in the last two columns)

Source: `run/position-bands.out`. Decode means: feed the first N tokens through the cache one at a time.

`en_long` (618 tokens)

| Positions | Prefill KL | Prefill top-1 | Decode KL | Decode top-1 |
|---|---|---|---|---|
| [0, 1) | 0.0691 | 1.000 | 3.98 | 0.000 |
| [1, 4) | 0.845 | 1.000 | 1.87 | 1.000 |
| [4, 16) | 0.125 | 0.750 | 0.688 | 0.667 |
| [16, 64) | 0.0490 | 0.938 | 0.186 | 0.896 |
| [64, 256) | 0.0653 | 0.917 | n/a | n/a |
| [256, 513) | 0.0781 | 0.899 | n/a | n/a |
| [513, 618) | 0.0967 | 0.895 | n/a | n/a |

`de_long` (612 tokens)

| Positions | Prefill KL | Prefill top-1 | Decode KL | Decode top-1 |
|---|---|---|---|---|
| [0, 1) | 0.301 | 1.000 | 0.297 | 1.000 |
| [1, 4) | 0.703 | 0.333 | 0.340 | 0.667 |
| [4, 16) | 0.586 | 0.500 | 0.552 | 0.667 |
| [16, 64) | 0.351 | 0.729 | 0.356 | 0.708 |
| [64, 256) | 0.176 | 0.844 | n/a | n/a |
| [256, 513) | 0.107 | 0.875 | n/a | n/a |
| [513, 612) | 0.0665 | 0.879 | n/a | n/a |

Early positions are worse. Positions past 513 (after the window starts to slide) are not worse than the middle.

The reference shows massive activations at early positions. Residual-stream max|x| of the fp32 reference:

| Prompt | Layer | Position 0 | Position 1 | Position 2 | Median, positions >= 16 |
|---|---|---|---|---|---|
| `en_long` | 0 | 6 | 6 | 8 | 6 |
| `en_long` | 24 | 1696 | 56 | 58 | 47 |
| `en_long` | 35 | 6393 | 129 | 113 | 77 |
| `en_long` | 49 | 4793 | 648 | 702 | 481 |
| `de_long` | 0 | 6 | 6 | 7 | 6 |
| `de_long` | 24 | 24 | 30 | 34 | 58 |
| `de_long` | 35 | 75 | 196 | 1973 | 81 |
| `de_long` | 49 | 686 | 879 | 339 | 545 |

In `en_long` the first token carries the large value (layer 35: 6393 against a median of 77). In `de_long` it sits at position 2 (layer 35: 1973). It is not always the first token.

### 5e. Decode path against prefill (same 4-bit weights)

Source: `run/decode-consistency.out`. Prefill is deterministic: two runs give max logit difference 0.

| Prompt | Comparison | KL mean | KL max | Top-1 | Max abs logit diff |
|---|---|---|---|---|---|
| `de_short` (13) | decode tail vs prefill | 0.103 | 0.447 | 0.875 | 3.02 |
| `de_short` (13) | token by token, first 13 vs prefill | 0.0981 | 0.528 | 0.923 | 3.31 |
| `en_long` (618) | decode tail vs prefill | 0.00350 | 0.00836 | 0.750 | 0.59 |
| `en_long` (618) | token by token, first 13 vs prefill | 0.739 | 4.15 | 0.769 | 10.1 |

Late positions agree between decode and prefill. The first positions do not, even with identical weights.

The tiny-model test (`tests/test_kolibri1.py::test_cache_consistency`, `scripts/smoke_tiny.py`) shows decode with cache and full prefill agree to 1e-6 across the window boundary in fp32. The bf16 gap above is numerics, not a cache bug, as far as this evidence goes.

### Interpretation

{{INTERPRETATION}}

### What vLLM does differently from the fp32 reference

* The residual stream is rounded to bf16 after each fused add. The torch reference keeps fp32 throughout.
* FP8 dynamic activation quantisation on the FP8 linear layers. The reference uses dequantised weights and unquantised activations.
* The router gate multiplies bf16 input and bf16 weights with fp32 accumulation and fp32 output. The MLX model feeds an fp32 input to the gate so the logits are not rounded to bf16 before top-k.
* `head_dtype: float32` means fp32 accumulation over bf16 operands for `lm_head` on GPU. MLX does the same accumulation, so there is no special handling.

So even a perfect port will not be bit-exact with the production stack. A bf16-level difference is expected.

## 5. Eval readiness

### Server smoke test

`scripts/smoke_server.py` against `models/Kolibri-1-4bit` through `serve.py`, sampling `temperature 1.0, top_p 0.97, top_k 128`, max 400 tokens. Model load took 5.0 s. Source: `verify/out/server-smoke.json`. All three cases passed.

| Case | Reasoning | Result | Prompt tok | Completion tok | Wall s | Completion tok / wall s |
|---|---|---|---|---|---|---|
| German, two-sentence explanation | on (`low`) | thinking in `reasoning`, two-sentence German answer, finish `stop` | 57 | 193 | 12.4 | 15.5 |
| English, three commit-message tips | off | three tips, finish `stop` | 48 | 91 | 2.21 | 41.1 |
| German weather question with `get_weather` tool | off | `get_weather {"city": "Zurich"}`, finish `tool_calls` | 187 | 23 | 1.01 | 22.9 |

The wall-clock rates include prompt processing and request overhead, so they are lower than the pure generation rate below. Server RSS was 43.1 GB.

### Speed and memory (`generate.py`, mlx_lm.generate, 256 new tokens)

| Prompt | Prompt tok/s | Generation tok/s | Peak memory |
|---|---|---|---|
| German, 61 tokens | 61.6 | 54.5 | 44.1 GB |
| English, 58 tokens | 113 | 49.4 | 44.1 GB |

### Memory headroom

System free memory was 8 to 13 % while serving on this 64 GB machine with Docker running. There were no memory-pressure events. It is tight. Close other large processes before long evaluation runs. `scripts/memguard.sh <floor_pct> <logfile> <cmd...>` kills a job when system free memory falls under the floor.

### Known mlx-lm limitations for this model

* `--kv-bits` does not work. `RotatingKVCache.to_quantized` is not implemented in mlx-lm.
* `--max-kv-size` is ignored, because the model defines `make_cache`.
* Prompt-cache trimming stops working once a sliding layer has passed 513 tokens, so prefix caching beyond that point cannot be reused.

## 6. Known gaps and open questions

* 4-bit quality. End to end the 4-bit model disagrees with the reference more than bf16 does. Whether this is inherent to 4-bit affine quantisation with group size 64 or can be reduced (8-bit attention, finer groups, different treatment of embeddings and `lm_head`) is open. The 8-bit and mixed runs are pending.
* Early-position sensitivity. Positions 0 to 15 are the worst, in prefill and in decode. Massive activations in the reference at those positions are the likely cause. This is a hypothesis, not a result.
* Even bf16 drifts when chained, and the drift is concentrated on short prompts. Short prompts are a large share of typical eval items. Check this in your own suites.
* No 8-bit model can be loaded on 64 GB. Checks of 8-bit are layer by layer only.
* The transformers warning about an "incorrect regex pattern" (Mistral regex) is a false alarm. The raw `tokenizers` library, the transformers default and `fix_mistral_regex=True` give identical token ids on all verification prompts and on edge cases. The other warning, about model type `kolibri1`, is harmless too.
* No reference with the real production stack (vLLM on CUDA, FP8 activations) was available. Everything is compared with our fp32 CPU reference.
* The dequantisation check covers shard 1 of the BF16 repo only.
* The upstream mlx-lm PR needs your own description. mlx-lm policy: AI use must be disclosed and PR text must not be AI-written. `drafts/mlx-lm-pr.md` lists facts and tasks only.
* No upload to Hugging Face has been done. `drafts/model-card.md` is a draft with `{{...}}` fields.
* The repo is private.

## 7. Repository layout

| Path | Purpose |
|---|---|
| `BRIEF.md` | task brief and constraints |
| `NOTES.md` | running log: decisions and numbers |
| `README.md` | this file |
| `pyproject.toml`, `uv.lock` | `uv` project, Python 3.12 |
| `generate.py` | `mlx_lm.generate` with the model registered |
| `serve.py` | `mlx_lm.server` with the model registered |
| `kolibri_mlx/models/kolibri1.py` | the model file, written in mlx-lm form |
| `kolibri_mlx/register.py` | inserts the model into `sys.modules` for mlx-lm |
| `kolibri_mlx/fp8.py` | FP8 block dequantisation |
| `kolibri_mlx/checkpoint.py` | streamed access to the FP8 checkpoint |
| `kolibri_mlx/reference_torch.py` | torch fp32 reference of the forward pass |
| `kolibri_mlx/verify_utils.py` | shared metrics and helpers for the verification scripts |
| `scripts/convert.py` | streaming FP8 to MLX converter |
| `scripts/check_dequant.py` | dequant check against the BF16 checkpoint |
| `scripts/reference_run.py` | layer-streamed fp32 reference run, writes `verify/out/reference/` |
| `scripts/verify_layerwise.py` | per-layer MLX vs reference, teacher-forced or `--chain` |
| `scripts/verify_e2e.py` | end-to-end logits of a converted model vs reference |
| `scripts/check_decode_consistency.py` | decode path vs prefill |
| `scripts/check_position_bands.py` | error by position band, residual maxima |
| `scripts/smoke_server.py` | server smoke test |
| `scripts/smoke_tiny.py` | tiny-model smoke test (cache, sanitize, quantise) |
| `scripts/memguard.sh` | kills a job when system free memory is low |
| `tests/test_kolibri1.py` | pytest suite on a tiny random model with an independent torch reference |
| `verify/prompts.py` | the five verification prompts |
| `verify/out/` | result JSON files and reference activations (git-ignored) |
| `drafts/mlx-lm-pr.md` | notes for the upstream PR |
| `drafts/model-card.md` | draft model card |
| `models/` | converted models (git-ignored) |
| `run/` | run output files (git-ignored) |

## 8. Licences

* Upstream weights (`Aleph-Alpha/Kolibri-1`, `Kolibri-1-BF16`) and the reference vLLM plugin (`Aleph-Alpha/aleph-alpha-inference`): Apache 2.0, Aleph Alpha.
* mlx-lm: MIT. The model file follows mlx-lm's header convention and would be contributed under mlx-lm's licence.
* This repo's own code: licence to be decided by Max. CC0 is intended. No `LICENSE` file has been added yet.

## 9. Reproduction checklist

Run from the repo root. Steps 3 to 6 need the 4-bit model from step 2. Steps 4 and 5 need the reference from step 3.

```bash
# 0. environment and unit tests (tiny random model, no downloads)
uv sync
uv run pytest tests -q

# 1. download weights (FP8 for conversion; BF16 only for the dequant check)
hf download Aleph-Alpha/Kolibri-1
hf download Aleph-Alpha/Kolibri-1-BF16 --include "config.json" "model.safetensors.index.json" "model-00001-of-00032.safetensors"   # optional, for check_dequant
uv run python scripts/check_dequant.py --shard 1                  # optional

# 2. convert
uv run python scripts/convert.py --log

# 3. fp32 torch reference (about 2.5 min per prompt set; peak RSS about 26 GiB)
uv run python scripts/reference_run.py

# 4. layer-by-layer checks
uv run python scripts/verify_layerwise.py --dtype float32  --out verify/out/layerwise-float32.json
uv run python scripts/verify_layerwise.py --dtype bfloat16 --out verify/out/layerwise-bfloat16.json
uv run python scripts/verify_layerwise.py --dtype bfloat16 --chain --out verify/out/chain-bf16.json
uv run python scripts/verify_layerwise.py --dtype bfloat16 --chain --bits 4 --out verify/out/chain-4bit.json

# 5. end to end, position bands, decode consistency (4-bit model)
uv run python scripts/verify_e2e.py --out verify/out/e2e-Kolibri-1-4bit.json
uv run python scripts/check_position_bands.py
uv run python scripts/check_decode_consistency.py

# 6. serve, smoke test, speed
uv run python scripts/smoke_server.py
uv run python generate.py --model models/Kolibri-1-4bit -p "Schreibe einen kurzen Absatz über die Geschichte der Stadt Zürich." -m 256 --temp 1.0 --top-p 0.97 --top-k 128
```

The exact flags used for the shipped result files are in the `opts` field of each JSON file. Check them before comparing numbers. Keep at least 15 GB of free memory when running steps 5 and 6, because the 4-bit model alone takes 44 GB.
