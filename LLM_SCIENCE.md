# Chat Summary — LLM Inference Optimization: KV Caching & Prompt Caching

## Topics Covered
Two conceptual deep-dives into LLM inference optimization, both initiated as understanding/learning questions (no code or tasks produced).

---

## Topic 1: Why KV Cache Remains Valid Across Decoding Steps

### Original Question
Why are cached K/V vectors still valid after each new output token is generated, given that intuitively the K/V of each token seems like it should depend on all other tokens in the sequence (including newly generated ones)?

### Key Points Established

- **K and V vectors are not functions of other tokens** in causal (decoder-only) LLMs.
  - Computed as: `K_i = W_K · x_i` and `V_i = W_V · x_i`
  - `x_i` is the hidden state of token `i` *before* the attention operation — a per-token projection, independent of other tokens at the same layer.

- **What does depend on other tokens**: the *output* of the attention operation (via `softmax(Q · Kᵀ / √d) · V`), which feeds into the next layer — but this does not feed back into the K/V computation of previous tokens.

- **Across layers**: yes, `x_i^(L+1)` encodes information from all other tokens (because attention output is used to update hidden states). So K/V at layer L+1 *indirectly* carries information from all tokens — but this was computed once during prefill and is fixed for prompt tokens.

- **New output token generation**: only appends its own K/V to the cache. Previous tokens' K/V are never invalidated or recomputed. This is valid because causal attention is unidirectional — token `i` never attends to tokens `j > i`.

- **Why your intuition would be correct for BERT**: bidirectional models have every token attending to every other, so adding a new token would require full recomputation. This is why BERT cannot use KV caching and is not autoregressive.

### Architecture Comparison
| Property | Causal LLM (GPT/LLaMA) | Bidirectional (BERT) |
|---|---|---|
| K/V depend on other tokens? | No | Yes |
| KV Cache valid? | ✅ Yes | ❌ No |
| Autoregressive generation? | ✅ Yes | ❌ No |

---

## Topic 2: KV Caching vs. Prompt Caching

### Original Question
What is the difference between prompt caching and KV caching as inference optimization techniques?

### Key Points Established

**KV Caching**
- Per-request, in-memory optimization within a single generation pass
- Stores K/V for all tokens processed so far to avoid O(n²) recomputation during decoding
- Lives in GPU VRAM, discarded after the request completes
- Always on, invisible to the user — standard plumbing in every modern inference stack

**Prompt Caching**
- Cross-request optimization that reuses K/V activations for a shared prefix across multiple separate API calls
- Targets repeated large prefixes: system prompts, documents, few-shot examples
- Persists for a defined time window (e.g., minutes); has its own pricing model
- Exposed as an explicit feature at the API/serving layer (e.g., Anthropic, OpenAI) — user opts in and structures prompts around it

**Relationship**
- Prompt caching is built *on top of* KV caching: what's being reused *is* the KV cache (prefill activations) for a shared prefix, but the scope is extended from within-request to cross-request, requiring higher-level cache management (storage, prefix matching, eviction).

### Summary Table
| | KV Caching | Prompt Caching |
|---|---|---|
| Scope | Single request | Across multiple requests |
| What's cached | K/V for all tokens so far | K/V for a shared prefix |
| Lifetime | One request | Minutes+ (configurable) |
| Who manages it | Inference engine (automatic) | Application/API layer (opt-in) |
| Visibility | Invisible | Explicit, pricing implications |

---

## Open Questions / Unresolved Points
- None explicitly left open — both questions reached clean resolution.

## Current State
Both topics fully addressed. No follow-up questions pending at time of export.