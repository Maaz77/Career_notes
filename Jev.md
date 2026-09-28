# Jev — Overview

## What Jev Is

Jev is not a generative model. It is a **decision model** from TypeSafe AI. It does not generate text. It reads a state and answers typed questions with structured, probabilistic answers.

TypeSafe calls this class of model a **"System One" model**. The name comes from Daniel Kahneman's idea of fast, intuitive thinking. Reasoning LLMs are the slower "System 2" by comparison.

## Launch Facts

- Launched September 15, 2026.
- Built by Diogo Almeida, who co-invented RLHF and InstructGPT at OpenAI.
- Raised $40M, led by DCVC.
- Response time: 70–500 milliseconds.
- Price: $0.042 per million input tokens. Output tokens are free.
- Trained with a method TypeSafe calls **RLCD** (Reinforcement Learning for Calibrated Decisions).

## The Three Primitives

| Type | Purpose | Returns |
|---|---|---|
| **Noul** | Answers a yes/no question | A probability of "yes", from 0 to 1 |
| **Choice** | Picks one option from a defined set | The top option, plus a probability for every option |
| **Score** | Rates the state on an ordered rubric | A probability-weighted score across the levels |

## `criteria` Field Shape by Type

- **Score** → an **array of strings**, ordered from lowest level to highest level.
- **Choice** → a **dictionary**, each option name mapped to a description.
- **Noul** → an optional dictionary with `true` and `false` descriptions.

## API Shape

**Request**

```json
POST https://api.typesafe.ai/v1/systemone
{
  "state": "...",
  "model": "jev-latest",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" }
  }
}
```

**Response**

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": { "type": "noul", "noul": 0.95 }
  },
  "usage": { "input_tokens": 296, "output_tokens": 20 }
}
```

- `answers` returns one entry per question, using the same keys you chose.
- Only **Choice** and **Score** answers include `confidence`. A **Noul** answer has no separate confidence field — the probability itself carries the uncertainty.
- `state` can be a string, object, or array. An array counts as **one** state, not a batch of separate items.
- You can put more than one Score (or Choice, or Noul) question in the same request. Each runs independently, in parallel, against the same state.

## Python SDK Example

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()

ticket = "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing."

response = client.system_one(
    state=ticket,
    questions={
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity",
        ),
    },
)

print(response.answers["department"].choice)
print(response.answers["frustration"].score)
print(response.answers["is_urgent"].noul)
```

## Open Source Status

Jev is **closed**, as of September 19, 2026:

- No open weights.
- No technical paper or model card.
- No way to run it locally.
- Every call goes through a hosted service: TypeSafe's own API, or a reseller (OpenRouter, Vercel AI Gateway, Cloudflare Workers AI).

**What TypeSafe has open-sourced:** the Python and JavaScript SDKs, an agent skill, and an adapter that runs the same typed questions on OpenAI or Anthropic models instead of Jev. These are clients only — they contain no model weights.

**Unofficial community replications** (not the real Jev, unverified benchmark claims):
- `kyegomez/open-jev` — a PyTorch reconstruction, explicitly built on guesses about the architecture.
- `Heman10x-NGU/Verdict-open-jev` — a smaller open model (151M params) inspired by Jev and RLCD, scoring lower than Jev on its own reported benchmark.

## Practical Notes

- `confidence` is derived from the probability distribution, but it is not simply the top probability — the docs don't disclose the exact formula.
- Add an "other" option to Choice questions when your option list might not cover every input.
- Pin an exact model version (e.g. `jev-1.13.0`) in production rather than `jev-latest`, since TypeSafe can update the model under you.
- Retry `429` and `529` errors with exponential backoff — the official SDKs do this automatically.

## Jev Providers
- Openrouter
- Vercel AI Gateway
- typeSafe.ai