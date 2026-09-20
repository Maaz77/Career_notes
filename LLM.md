# Tokenizer

- If the tokenizer tokenize the words into bigger tokens, it is better because the attention mechanism can attend to more previous tokens given the limited context window. 

- strings in python are encoded 

- Unicode code point: The character’s unique number
- UTF-8 : A method to convert that number into bytes


# RLHF

Large language models (LLMs) are first pretrained as next-token predictors on massive text corpora.  
To make them helpful and safe for conversational use, modern chat models undergo **Reinforcement Learning from Human Feedback (RLHF)**.

The canonical training pipeline consists of three stages:

1. Supervised Fine-Tuning (SFT)  
2. Reward Modeling  
3. Reinforcement Learning (typically PPO-based)

---

## 1. Supervised Fine-Tuning (SFT)

The base model is trained on instruction–response pairs written by humans.

Given:

- Prompt $x$  
- Assistant response tokens $y_1, \dots, y_T$

The objective is standard next-token prediction:

$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\sum_{t=1}^{T}
\log p_\theta(y_t \mid x, y_{<t})
$$

Key details:

- The full conversation (system + user + assistant) is tokenized into one sequence.
- Training uses **teacher forcing**.
- Loss is computed only on assistant tokens (others masked).
- The labels are the same sequence shifted by one position (next-token prediction).

This produces the supervised policy:

$$
\pi_{\text{SFT}}(y \mid x)
$$

---

## 2. Reward Model (RM)

The SFT model generates multiple candidate responses for each prompt.  
Humans rank them from best to worst.

A reward model $r_\phi(x, y)$ is trained to predict these rankings.

Typical pairwise ranking loss:

$$
\mathcal{L}_{\text{RM}}(\phi)
=
- \log \sigma\big(
r_\phi(x, y_{\text{w}})
-
r_\phi(x, y_{\text{l}})
\big)
$$

Where:

- $y_{\text{w}}$ = preferred response  
- $y_{\text{l}}$ = rejected response  
- $\sigma(z) = \frac{1}{1 + e^{-z}}$

The reward model outputs:

$$
R(x,y) = r_\phi(x,y)
$$

---

## 3. Reinforcement Learning (PPO)

The goal is to improve the policy while staying close to the SFT model.

### Policy

The LLM defines a policy:

$$
\pi_\theta(y \mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t \mid x, y_{<t})
$$

### Advantage

We compute:

$$
A = R - V
$$

Where:

- $R$ = reward from reward model  
- $V$ = baseline (value function estimate)  

Advantage tells whether a response is better or worse than expected.

---

### PPO Probability Ratio

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\text{old}}(a_t \mid s_t)}
$$

Measures how much the probability changed relative to the previous policy.

---

### PPO Clipped Objective

$$
L_{\text{PPO}}(\theta)
=
\mathbb{E}_t
\Big[
\min(
r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
)
\Big]
$$

- $\epsilon \approx 0.1 - 0.2$
- Prevents overly large policy updates
- Stabilizes training

---

### KL Regularization

To avoid drifting too far from SFT:

$$
R'(x,y)
=
r_\phi(x,y)
-
\beta \,
\mathrm{KL}(
\pi_\theta
\;\|\;
\pi_{\text{SFT}}
)
$$

Final objective:

$$
\max
\mathbb{E}[R']
$$

This enforces:

- Improve reward
- Stay close to original supervised model

---

### Intuition

- SFT teaches format and instruction-following.
- The reward model learns human preferences.
- PPO reshapes token probabilities gradually.
- The KL term prevents catastrophic policy shifts.
- Advantage reduces gradient variance and stabilizes updates.

RLHF adjusts the probability distribution so that:

> Responses humans prefer become more likely.

---

## Alternatives to RLHF

Because RLHF is expensive and complex, several alternatives exist.

## 1. Rejection Sampling Fine-Tuning

Generate multiple responses, keep only high-reward ones, train via supervised learning.

## 2. Direct Preference Optimization (DPO)

Eliminates PPO and the explicit reward model.

$$
\mathcal{L}_{\text{DPO}}(\theta)
=
- \log \sigma
\Big(
\lambda[
\log \pi_\theta(y_w \mid x)
-
\log \pi_\theta(y_l \mid x)
]
-
[
\log \pi_0(y_w \mid x)
-
\log \pi_0(y_l \mid x)
]
\Big)
$$

Encourages higher probability for preferred answers without RL.

## 3. ReST (Rejection Sampling + Self-Training)

Iterative generate → filter → fine-tune loop.

## 4. RLAIF

Replace human rankings with AI-generated rankings.

---

## Final Summary

RLHF consists of:

1. Supervised Fine-Tuning (cross-entropy objective)
2. Reward model trained via pairwise ranking
3. PPO-based policy optimization with clipping and KL regularization

Core mathematical components:

- $\pi_\theta$ : policy  
- $R$ : reward  
- $V$ : value baseline  
- $A = R - V$ : advantage  
- $r(\theta)$ : probability ratio  
- PPO clipped objective  
- KL regularization  

RLHF reshapes the LLM probability distribution to align with human preferences while preserving language capability learned during pretraining.


# Supervised Fine-Tuning (SFT) of LLMs 

This report consolidates the concepts and questions raised in this chat that relate to **Supervised Fine-Tuning (SFT)** and the *prerequisites needed to understand SFT end-to-end* (tokenization, embeddings, context window behavior, and inference mechanics). It is written to be directly usable as a reference.

---

## 1) Context window during inference

### What “context window = N tokens” means
A model’s **context window** is the **maximum number of tokens the model can condition on at a time** during generation. In typical deployments, this budget includes **both**:

- **Input tokens** (your prompt / conversation history), and
- **Output tokens** generated so far in the current response.

So, at generation step *t*, the model conditions on:

- prompt tokens + previously generated tokens.

### What happens if the input alone fills the context window?
If the input already reaches the maximum context length, there is **no room** for additional output tokens. Systems handle this by one of the following behaviors:

- **Reject / error**: “context length exceeded” (hard failure).
- **Truncation**: keep only the last *N* tokens (or keep system + last turns).
- **Sliding window**: drop older tokens as generation proceeds (implementation-dependent).

---

## 2) Inference mechanics (how an LLM produces outputs)

### One forward-pass per generated token
At a low level, transformers produce text by repeated **next-token prediction**:

1. Compute logits over vocabulary for the next token given current tokens.
2. Select a token (argmax, sampling, etc.).
3. Append it to the sequence.
4. Repeat until stop.

### “Reasoning” models (high-level explanation)
Regardless of whether the model is called a “reasoning model,” the underlying transformer still performs **next-token prediction**. The *serving system* may structure the generation so that:

- internal “reasoning” tokens are generated (possibly hidden), then
- final answer tokens are produced.

This is still a single token-generation stream; deployment may additionally apply decoding strategies (sampling multiple candidates, re-ranking, etc.), but the core inference is repeated forward passes.

---

## 3) Tokenization: how text becomes numbers

### 3.1 What the tokenizer is (and is not)
A tokenizer in typical LLM SDKs is **not neural-network-based**. It is a deterministic text processing component consisting of:

- vocabulary: token ↔ integer ID mappings,
- tokenization rules (e.g., BPE merges / unigram model),
- normalization rules,
- special tokens.

A tokenizer does **not** compute embeddings.

### 3.2 What the tokenizer input/output are
- **Input**: Unicode string (Python `str`).
- **Output**: list/array of integer token IDs (`List[int]`).

### 3.3 UTF-8 vs Unicode code points (what tokenizers operate on)
Tokenizer internals depend on the design:

- **GPT-style tokenizers**: commonly **byte-level BPE**  
  Unicode string → UTF-8 bytes → merges → token IDs.  
  This guarantees the tokenizer can represent any string (no true “unknown token” failure).
- **Other tokenizers (e.g., some SentencePiece setups)**: operate over Unicode characters / subwords with probabilistic segmentation.

### 3.4 Can a tokenizer “not know” how to tokenize some input?
For byte-level tokenizers: it will always produce tokens because any text can be represented as UTF-8 bytes.  
However, it can tokenize *inefficiently* (many tokens) for rare strings, hashes, mixed scripts, etc.

---

## 4) Python text basics used in tokenization

### Are Python strings UTF-8?
Python `str` is stored as Unicode code points internally (not UTF-8). UTF-8 appears when converting:

- `str.encode("utf-8")` → bytes
- `bytes.decode("utf-8")` → str

### Convert character to Unicode / UTF-8
```python
c = "é"

# Unicode code point
ord(c)          # 233
f"U+{ord(c):04X}"  # "U+00E9"

# UTF-8 bytes
c.encode("utf-8")     # b'\xc3\xa9'
list(c.encode("utf-8"))  # [195, 169]
```

---

## 5) Embeddings: how token IDs become vectors

### 5.1 Token embeddings are a lookup table (not an MLP)
After tokenization, token IDs are mapped to vectors via a learned **embedding matrix**:

- vocab size = `V`
- hidden size = `d`
- embedding matrix: `W_E ∈ R^(V × d)`

For token id `i`, the token embedding is the row:

- `e = W_E[i]`

This is implemented as `nn.Embedding` (lookup), not as an MLP.

### 5.2 Positional information
Before entering the transformer block, token embeddings are combined with positional information:

- **learned positional embeddings**: `z_t = e_t + p_t`, or
- **RoPE/rotary embeddings**: position is applied inside attention (no direct addition at input).

---

## 6) BERT contextual embeddings and the [CLS] token

### What BERT outputs
BERT outputs a vector per token:

- `H ∈ R^(T × d)` (sequence length `T`)

### Fixed-size “sentence embedding”
A fixed-size representation is obtained by pooling, commonly:

- **[CLS] vector**: use the hidden state at the CLS position, or
- **mean pooling**: average token vectors.

### How [CLS] “encodes the whole sentence”
[CLS] can attend to all tokens through self-attention. Over layers, its representation aggregates information. Training objectives and downstream fine-tuning often apply loss via [CLS], making it learn a global summary useful for tasks.

---

## 7) SFT for conversational chat models: dataset and loss

### 7.1 What the SFT dataset looks like
Chat SFT datasets are typically structured dialogues, e.g.:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Internally, this is serialized into one sequence with role markers, then tokenized.

### 7.2 What the model is trained to do
Even for chat, the model is trained as a **causal language model**:

> predict the **next token** at each position.

### 7.3 Cross-entropy loss with assistant-only masking
For a prompt `x` (system + user + prior turns) and assistant response tokens `y_1..y_T`:

$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\sum_{t=1}^{T}
\log p_\theta(y_t \mid x, y_{<t})
$$

Crucial implementation detail: loss is computed **only** on assistant tokens.
System/user tokens are masked (ignored) so the model learns “given this conversation history, produce the assistant continuation.”

### 7.4 “Shift” of inputs to form labels (why and how)
A causal LM produces logits at position `t` intended to predict token `t+1`.
So to compute loss:

- `shift_logits = logits[:, :-1, :]`
- `shift_labels = input_ids[:, 1:]`

This aligns each position’s prediction with the **next** token label.  
If we did not shift, the model would “see” the token it is supposed to predict (trivial and incorrect supervision).

### 7.5 What do the “true labels” look like in chat SFT?
Labels are generally the same as `input_ids`, but tokens not belonging to assistant spans are set to an ignore index (commonly `-100`) so the loss function skips them:

- `labels = input_ids.clone()`
- `labels[:assistant_start] = -100` (and any non-assistant tokens set to `-100`)

Then cross-entropy is computed over the remaining labels.

---

## 8) Parameter-efficient SFT (LoRA, QLoRA) and loss

### Is LoRA an SFT method?
No. **SFT** is a training setup/objective (supervised cross-entropy on instruction-response data).  
**LoRA** is a parameter update strategy (what weights you train).

You can do **SFT + LoRA**, **SFT + QLoRA**, or **SFT full fine-tuning**.

### What loss is used when fine-tuning with LoRA?
Typically the same SFT cross-entropy loss (assistant tokens only). LoRA does not define a new loss; it limits which parameters receive gradients.

### QLoRA: what is quantized?
In QLoRA, the **base model weights** are quantized (commonly to 4-bit, e.g., NF4) and kept frozen, while **LoRA adapter weights** are trained in higher precision (fp16/bf16). Quantization is primarily for memory efficiency; compute uses dequantized values during forward passes.

---

## 9) Minimal training loop summary for chat SFT

For each training sample:

1. Serialize messages → one text string with role markers.
2. Tokenize to `input_ids` (length `T`).
3. Build `labels`:
   - copy `input_ids`
   - mask non-assistant tokens as `-100`
4. Forward pass → logits `(batch, T, vocab)`.
5. Shift logits/labels for next-token prediction.
6. Compute cross-entropy on non-masked labels.
7. Backpropagate:
   - full fine-tuning: update all weights
   - LoRA: update only adapter weights
8. Repeat over dataset.

---

## 10) Quick glossary of symbols used in this report

- `V`: vocabulary size  
- `d`: hidden size / embedding dimension  
- `T`: sequence length (tokens)  
- `x`: prompt / context tokens  
- `y`: assistant response tokens  
- `W_E`: token embedding matrix  
- `p_\theta`: model probability distribution over next tokens  

---

## 11) Practical mental model

- **Tokenizer**: deterministic mapping `text → token IDs`.
- **Embedding lookup**: `token IDs → vectors`.
- **Transformer**: produces contextual hidden states and next-token logits.
- **SFT**: uses teacher forcing + shifted labels + assistant-only loss to teach “given a conversation, produce the assistant reply.”

