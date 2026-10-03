# kolibri-mlx

This is an MLX port of Aleph Alpha's Kolibri 1 (`Aleph-Alpha/Kolibri-1`) for mlx-lm, with a reproducible FP8-to-4-bit conversion and a numerical verification against a reference rebuilt from Aleph Alpha's vLLM plugin. The upstream mlx-lm pull request has not been opened yet.

## Status

Done. The shipped conversion `models/Kolibri-1-4bit-mixed` (routed experts 4-bit, everything else 8-bit, 42 GiB on disk, 45.4 GB peak) loads with mlx-lm 0.32.0 on an Apple M4 Pro with 64 GB. A plain uniform 4-bit conversion (41 GiB, 44.1 GB peak) was measured for comparison and then deleted to free disk; its numbers are kept in section 5. Generation runs at about 50 tok/s. The OpenAI-compatible server answers German and English prompts with reasoning on and off and parses a tool call. The converted model is published on the Hugging Face Hub as [`eins78/Kolibri-1-mlx-mixed-4-8-bit`](https://huggingface.co/eins78/Kolibri-1-mlx-mixed-4-8-bit) (Apache 2.0, same files as `models/Kolibri-1-4bit-mixed` plus the model card). The owner's own evaluation suites were run against it: the MeteoSwiss forecast suite passes 78/91 deterministic cases and 4/4 judged ones; the Hermes B1 tool-calling eval is partial (3 of 8 cases, blocked by memory headroom on this machine). Numbers and baselines in `evals/RESULTS.md`. No upstream mlx-lm PR or issue has been opened; the owner writes those.

### Verdict

**The port is correct.** Run in fp32, layer by layer from the real embedding through `lm_head` with no teacher forcing, the MLX implementation reproduces the fp32 torch reference on all five prompts: top-1 agreement 1.000, top-5 1.000, mean KL about 1e-11, max logit difference at most 2.3e-4, identical expert selection at every layer (`verify/results/chain-fp32.json`).

**The shipped model is a 4-bit quantisation and is measurably lossy.** The reference deployment itself runs in bf16 with FP8 activations, so the yardstick for a quantised model is the bf16 chained run, not fp32. Against the fp32 reference on the two long prompts, bf16 gives top-1 0.968 / 0.931 and mean KL 0.008 / 0.036; the shipped mixed 4-bit model gives top-1 0.966 / 0.892 and mean KL 0.017 / 0.069 (plain 4-bit: 0.905 / 0.845 and 0.080 / 0.154). 8-bit would be lossless relative to bf16 but does not fit in 64 GB. Numbers in section 5. Whether this loss is acceptable depends on the evaluation; see Interpretation.

## AI assistance

The port, the scripts, the tests and this README were written by an AI coding agent (Claude Code, Claude Fable 5.1, with Opus and Sonnet subagents). It worked from the brief in `docs/BRIEF-phase1.md` and was directed and reviewed by the repository owner. The verification numbers were produced by the scripts in this repository. mlx-lm's contribution policy requires disclosure of AI use and a human-written PR description for any upstream submission.

## 1. Quick start

Needs `uv` and Python 3.12. To skip the conversion, download the converted model from the Hub (42 GiB) instead of steps 1 and 2; stock mlx-lm cannot load it without this repository's `kolibri_mlx.register`, which `generate.py` and `serve.py` import.

```bash
uv sync

# 0. (shortcut) use the published conversion instead of steps 1 and 2
hf download eins78/Kolibri-1-mlx-mixed-4-8-bit --local-dir models/Kolibri-1-4bit-mixed

# 1. download the FP8 checkpoint (78 GB) into the Hugging Face cache
hf download Aleph-Alpha/Kolibri-1

# 2. convert to 4-bit MLX (about 3 minutes, peak MLX memory 8.75 GB)
uv run python scripts/convert.py --attn-bits 8 --embed-bits 8 --lm-head-bits 8 \
    --mlx-path models/Kolibri-1-4bit-mixed   # routed experts 4-bit, rest 8-bit (recommended)
# plain uniform 4-bit: uv run python scripts/convert.py   # writes models/Kolibri-1-4bit

# 3. generate (recommended sampling from Aleph Alpha)
uv run python generate.py --model models/Kolibri-1-4bit-mixed \
  -p "Erkläre in zwei Sätzen, warum der Himmel blau ist." -m 400 \
  --temp 1.0 --top-p 0.97 --top-k 128

# 4. serve (OpenAI-compatible)
uv run python serve.py --model models/Kolibri-1-4bit-mixed --port 8080
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

Smoke test (starts its own server, writes `verify/results/server-smoke.json`):

```bash
uv run python scripts/smoke_server.py --model models/Kolibri-1-4bit-mixed
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
| `--attn-bits`, `--expert-bits`, `--embed-bits`, `--lm-head-bits` | same as `--bits` | per-group bits; 0 keeps that group in bf16. Overrides are written into `config.json` `quantization` as mlx-lm does |
| `--bits` | 4 | 2, 3, 4, 5, 6 or 8 |
| `--group-size` | 64 | quantisation group size |
| `--no-quant` | off | write bf16 without quantisation |
| `--layers N` | all | debug: convert only the first N layers |
| `--dry-run` | off | print the layer 0 tensor layout and stop |
| `--log` | off | also log to `run/convert-<timestamp>.out` |

## 4. Verification

### Method

* Ground truth is a pure-PyTorch fp32 CPU forward built from the vLLM plugin's math (`kolibri_mlx/reference_torch.py`). `scripts/reference_run.py` runs it layer by layer with streamed, dequantised weights and saves every layer's residual stream, expert ids and expert weights, plus the final logits, to `verify/out/reference/`. 50 layers, five prompts, about 2.7 s per layer, peak RSS 26.2 GiB.
* Caveat: the reference is a PyTorch fp32 rebuild of the plugin's math. The plugin itself was not run, because it needs vLLM on CUDA. The production stack (bf16 residual stream, FP8 activations) therefore has its own bf16-level deviation from this reference.
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
| float32 | 6.76e-6 | 3.64e-7 | 100.0 % of tokens (2 token-layer pairs differ at fp32 noise level) | passes the 1e-3 bound. The 2 tokens are in layer 0 of `en_long`: exact ties at fp32 noise. |
| bfloat16 | 0.0382 | 0.00417 | 98.4 % mean over rows, minimum 80 % (13-token prompt) | one row above the 2e-2 bound: layer 35, `de_long`. |

The float32 row proves that the MLX model implements the reference math. In bf16, the worst row (layer 35, `de_long`) is one token with max-abs error 661 at a massive-activation token where an expert flipped. Its mean-abs error is 0.0377 and the cosine similarity of the last position is 1.0 to 6 digits.

### 5b. Chained bf16 (no teacher forcing, unquantised weights)

Final logits against the reference. Source: `verify/results/chain-bf16.json`.

| Prompt | Top-1 | Top-5 | KL mean | KL p95 | KL max |
|---|---|---|---|---|---|
| `de_short` (13) | 0.769 | 0.800 | 0.490 | 2.13 | 3.22 |
| `en_short` (15) | 0.933 | 0.960 | 0.00572 | 0.0215 | 0.0504 |
| `de_code` (24) | 1.00 | 0.950 | 0.000667 | 0.00303 | 0.00390 |
| `en_long` (618) | 0.968 | 0.957 | 0.00820 | 0.0220 | 1.55 |
| `de_long` (612) | 0.931 | 0.906 | 0.0359 | 0.115 | 2.18 |

Expert-set agreement over all layers and tokens drops to 81.2 % when the layers are chained (per prompt: 55.8 %, 78.7 %, 90.1 %, 84.7 %, 78.0 %).

### 5c. 4-bit end to end (converted model, prefill)

Source: `verify/results/e2e-Kolibri-1-4bit.json`. The on-the-fly quantised chain run (`verify/results/chain-4bit.json`, bits 4, group 64) gives the same logit numbers to three digits, so the converter output equals in-process quantisation of the same weights.

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
| bf16, unquantised (`chain-bf16.json`) | 0.769 / 0.490 | 0.933 / 0.0057 | 1.000 / 0.00067 | 0.968 / 0.0082 | 0.931 / 0.036 |
| 8-bit g64 (`chain-8bit.json`) | 0.692 / 0.373 | 1.000 / 0.0070 | 0.958 / 0.0023 | 0.969 / 0.0091 | 0.931 / 0.034 |
| 4-bit g64 (`chain-4bit.json`, = converted model) | 0.538 / 1.05 | 1.000 / 0.094 | 0.917 / 0.085 | 0.905 / 0.080 | 0.845 / 0.154 |
| 4-bit g32 (`chain-4bit-g32.json`) | 0.538 / 1.18 | 0.933 / 0.172 | 0.917 / 0.025 | 0.929 / 0.049 | 0.855 / 0.121 |
| mixed: experts 4-bit g64, attention, shared expert, embeddings, lm_head 8-bit (`chain-mixed-a8e4.json`) | 0.385 / 0.526 | 0.867 / 0.015 | 0.958 / 0.010 | 0.966 / 0.017 | 0.892 / 0.069 |

An 8-bit model is about 83 GB and cannot be loaded for generation on 64 GB, so it was only checked layer by layer. The mixed variant is the one shipped as `models/Kolibri-1-4bit-mixed` (see 5c-2). 
### 5c-2. Shipped model: `models/Kolibri-1-4bit-mixed` end to end

Converted with `--attn-bits 8 --embed-bits 8 --lm-head-bits 8` (routed experts 4-bit g64, router bf16). 42 GiB on disk, 45.4 GB peak memory. Source: `verify/results/e2e-Kolibri-1-4bit-mixed.json`. The numbers equal the chained on-the-fly mixed run, so the converter output is again identical to in-process quantisation.

| Prompt | Top-1 | Top-5 | KL mean | KL p95 | KL max | decode-vs-prefill max abs |
|---|---|---|---|---|---|---|
| `de_short` (13) | 0.385 | 0.754 | 0.526 | 1.75 | 1.77 | 5.25 |
| `en_short` (15) | 0.867 | 0.880 | 0.0146 | 0.0631 | 0.120 | 2.38 |
| `de_code` (24) | 0.958 | 0.925 | 0.00996 | 0.0478 | 0.108 | 0.719 |
| `en_long` (618) | 0.966 | 0.935 | 0.0174 | 0.0428 | 1.55 | 1.33 |
| `de_long` (612) | 0.892 | 0.875 | 0.0690 | 0.282 | 2.24 | 0.875 |

`verify_e2e.py` reports FAIL for this run because its default `--min-top1 0.9` gate is stricter than the bf16 floor on the short prompts; the gate is a tunable, not a verdict. Server smoke test on this model (`verify/results/server-smoke-mixed.json`): all three cases pass, 18 to 40 completion tokens/s, server RSS 44.4 to 44.6 GB. `generate.py`: prompt 227 to 247 tok/s, generation 50.3 tok/s, peak 45.36 GB.


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

Three effects are separated by the runs above:

1. **Implementation error: none measurable.** fp32 teacher-forced per layer (rel error <= 7e-6) and fp32 chained end to end (KL 1e-11, top-1 1.000) both match. The mask semantics (513 keys including self), RoPE placement, router, shared expert and sandwich norms are therefore right, including beyond the sliding window.
2. **bf16 numerics.** In bf16 the same code drifts from fp32 over 50 layers: residual relative error grows to about 0.1 to 0.3 by layer 24 to 35 and expert-set agreement falls to about 70 %. Two properties of this model cause it. The top-6 selection among 384 experts is made on `logits + bias` where the bias dominates and margins between the 6th and 7th expert are tiny, so bf16 rounding flips experts; and sink tokens carry massive activations (residual max 6393 at layer 35, position 0 of `en_long`, against a median of 77), which amplify rounding. The 13-token `de_short` prompt is all early positions and has close next-token distributions, so it is noisy in every run including bf16 and should not be read as a quantisation signal. vLLM's own bf16 + FP8-activation path has the same floor, so bit-exactness with production is not a meaningful target.
3. **Quantisation loss.** 8-bit matches bf16. Plain 4-bit g64 multiplies the KL gap by 4 to 10 on the long prompts. Keeping attention, shared expert, embeddings and `lm_head` at 8 bits (1.2 GB more) halves that gap; the remainder comes from the routed experts, which hold 75.5 of 78 B parameters and must stay 4-bit to fit in 64 GB. Group size 32 for the experts helps a little more but costs 4.7 GB and was not shipped.

The decode-versus-prefill differences on early positions (section 5e) are the same bf16 amplification seen from a different angle: in fp32 the two paths agree to 1e-6, and in bf16 at late positions they agree to KL 0.0035. Changing the quantisation of the sink-token path (for example keeping layer 0 or the first positions in higher precision) was not explored.

Practical reading: for evaluation at the recommended sampling settings, expect the mixed 4-bit model to behave like Kolibri 1 with a small amount of added noise, strongest on the first tokens of a context. If a benchmark is sensitive to that, compare against an 8-bit run on a larger machine.

### What vLLM does differently from the fp32 reference

* The residual stream is rounded to bf16 after each fused add. The torch reference keeps fp32 throughout.
* FP8 dynamic activation quantisation on the FP8 linear layers. The reference uses dequantised weights and unquantised activations.
* The router gate multiplies bf16 input and bf16 weights with fp32 accumulation and fp32 output. The MLX model feeds an fp32 input to the gate so the logits are not rounded to bf16 before top-k.
* `head_dtype: float32` means fp32 accumulation over bf16 operands for `lm_head` on GPU. MLX does the same accumulation, so there is no special handling.

So even a perfect port will not be bit-exact with the production stack. A bf16-level difference is expected.

## 5. Eval readiness

The tables in this section were measured on the plain 4-bit model. The shipped mixed model behaves the same for speed and memory within 3 %: 50.3 tok/s generation, 45.4 GB peak, all smoke cases pass (section 5c-2).

### Server smoke test

`scripts/smoke_server.py` against `models/Kolibri-1-4bit` through `serve.py`, sampling `temperature 1.0, top_p 0.97, top_k 128`, max 400 tokens. Model load took 5.0 s. Source: `verify/results/server-smoke.json`. All three cases passed.

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

* 4-bit quality. The routed experts (75.5 of 78 B parameters) must be 4-bit to fit in 64 GB, and that costs a measurable amount against the bf16 floor (section 5c, 5c-2). The shipped mixed variant halves the gap of plain 4-bit. Whether the remaining loss is acceptable is a question for the evaluation suites. An 8-bit model (about 83 GB) is lossless relative to bf16 but needs a larger machine.
* Early-position sensitivity. Positions 0 to 15 are the worst in bf16 and in 4-bit, in prefill and in decode, while fp32 is exact everywhere. Massive activations at the sink token (residual max 6393 at layer 35 versus a median of 77) and near-tie expert selection are the measured correlates; the causal story is a hypothesis. Keeping the first layers or the sink path in higher precision was not tried.
* Even bf16 drifts when chained, most on short prompts. Short prompts are a large share of typical eval items. Check this in your own suites.
* No reference with the real production stack (vLLM on CUDA, bf16 residual, FP8 activations) was available. Everything is compared with the fp32 CPU reference built from the plugin's math; the production stack has its own bf16-level deviation from that.
* The dequantisation check covers shard 1 of the BF16 repo only.
* The transformers warning about an "incorrect regex pattern" (Mistral regex) is a false alarm. The raw `tokenizers` library, the transformers default and `fix_mistral_regex=True` give identical token ids on all verification prompts and on edge cases. The warning about model type `kolibri1` is harmless too.
* The upstream mlx-lm PR needs your own description. mlx-lm policy: AI use must be disclosed and PR text must not be AI-written. `drafts/mlx-lm-pr.md` lists facts and tasks only.
* The model is on the Hub under the owner's account ([`eins78/Kolibri-1-mlx-mixed-4-8-bit`](https://huggingface.co/eins78/Kolibri-1-mlx-mixed-4-8-bit)); `drafts/model-card.md` is the card as uploaded. No `mlx-community` upload has been made.
* Serving leaves only 8 to 12 % of system memory free next to the Docker stack. Long evaluation runs should not share the machine with other large jobs.

## 7. Repository layout

| Path | Purpose |
|---|---|
| `docs/BRIEF-phase1.md` | phase 1 task brief and constraints |
| `docs/BRIEF-phase2.md` | phase 2 brief (evals, licences, publication) |
| `docs/BRIEF-phase3.md` | phase 3 brief (Hugging Face upload) |
| `LICENSE` | CC0 1.0 Universal, default licence |
| `LICENSES/` | full texts of the Apache-2.0 and MIT licences used by derived files |
| `NOTICE` | derived files, upstreams, copyrights and changes |
| `evals/` | evaluation suites run against the model; results in `evals/RESULTS.md` |
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
| `verify/results/` | result JSON files of the runs quoted in this README (tracked) |
| `verify/out/` | fresh run output and reference activations (git-ignored) |
| `drafts/mlx-lm-pr.md` | notes for the upstream PR |
| `drafts/model-card.md` | draft model card |
| `models/` | converted models (git-ignored) |
| `run/` | run output files (git-ignored) |

## 8. Licences

The aim is the most permissive arrangement the upstream licences allow. Wholly original code keeps no rights. Derived code keeps its upstream licence and carries the upstream notice.

* Default: CC0 1.0 Universal (`LICENSE`). This covers the converter, checkpoint reader, FP8 dequantisation, verification scripts, server wrappers, docs and notes.
* MIT: `kolibri_mlx/models/kolibri1.py`, derived from mlx-lm's `qwen3_moe.py`, `cohere2.py` and `deepseek_v3.py` (Apple Inc.). It is meant to be contributed to mlx-lm.
* Apache 2.0: `kolibri_mlx/reference_torch.py`, a transcription of the math of Aleph Alpha's `aleph_alpha_inference/kolibri1.py`. `tests/test_kolibri1.py` is Apache 2.0 AND MIT: the routing test is ported from the plugin, the model checks follow mlx-lm's test harness.
* Weights: `Aleph-Alpha/Kolibri-1` and the converted models stay under Apache 2.0 by Aleph Alpha. They are not distributed here.

Every source file has an SPDX identifier. `NOTICE` lists the derived files and what was changed. Full texts are in `LICENSES/`.

## 9. Reproduction checklist

Run from the repo root. Steps 3 to 6 need the 4-bit model from step 2. Steps 4 and 5 need the reference from step 3.

```bash
# 0. environment and unit tests (tiny random model, no downloads)
uv sync
uv run pytest tests -q

# 1. download weights (FP8 for conversion; BF16 only for the dequant check)
hf download Aleph-Alpha/Kolibri-1
hf download Aleph-Alpha/Kolibri-1-BF16 model-00001-of-00032.safetensors config.json model.safetensors.index.json   # optional, for check_dequant
uv run python scripts/check_dequant.py --shard 1                  # optional

# 2. convert (shipped variant; drop the *-bits flags for plain uniform 4-bit)
uv run python scripts/convert.py --attn-bits 8 --embed-bits 8 --lm-head-bits 8 --mlx-path models/Kolibri-1-4bit-mixed --log

# 3. fp32 torch reference (about 2.5 min per prompt set; peak RSS about 26 GiB)
uv run python scripts/reference_run.py

# 4. layer-by-layer checks
uv run python scripts/verify_layerwise.py --dtype float32  --out verify/results/layerwise-float32.json
uv run python scripts/verify_layerwise.py --dtype bfloat16 --out verify/results/layerwise-bfloat16.json
uv run python scripts/verify_layerwise.py --dtype bfloat16 --chain --out verify/results/chain-bf16.json
uv run python scripts/verify_layerwise.py --dtype bfloat16 --chain --bits 4 --out verify/results/chain-4bit.json

# 5. end to end, position bands, decode consistency (4-bit model)
uv run python scripts/verify_e2e.py --out verify/results/e2e-Kolibri-1-4bit.json
uv run python scripts/check_position_bands.py
uv run python scripts/check_decode_consistency.py

# 6. serve, smoke test, speed
uv run python scripts/smoke_server.py --model models/Kolibri-1-4bit-mixed
uv run python generate.py --model models/Kolibri-1-4bit-mixed -p "Schreibe einen kurzen Absatz über die Geschichte der Stadt Zürich." -m 256 --temp 1.0 --top-p 0.97 --top-k 128
```

The exact flags used for the shipped result files are in the `opts` field of each JSON file. Check them before comparing numbers. Keep at least 15 GB of free memory when running steps 5 and 6, because the model alone takes 44 to 45 GB.
