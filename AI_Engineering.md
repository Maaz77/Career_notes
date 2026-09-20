Pure Vector Search is not enough for production RAG. Here is the fix. 👇

1️⃣ Hybrid Search is Mandatory
👉 Embeddings are great for concepts (“dog” ~ “puppy”), but terrible for exact matches (product IDs, specific legal clauses). You MUST combine BM25 (Keyword Search) with Vector Search.

2️⃣ Reciprocal Rank Fusion (RRF)
👉 Don’t just average the scores. Use RRF to normalize the ranking from your keyword search and your vector search to get the best of both worlds.

3️⃣ Re-Ranking (The Secret Sauce)
👉 Retrieve 50 documents instead of 5. Then, pass them through a Cross-Encoder (like BGE-Reranker) to sort them by true relevance before sending to the LLM.

4️⃣ Metadata Filtering
👉 Don’t search the whole database. Filter by year, category, or source before the vector search runs (Pre-filtering).

---

Most RAG cost comes from retrieved context, not generation.
So the real trick is:
👉 send fewer tokens, but higher-signal ones.

Here’s how teams do it in production:


1️⃣ Tighten Retrieval (quality > quantity)
• Reduce top-k — don’t send 10-15 chunks when 4-5 good ones work.
• Add a cross-encoder re-ranker to keep only the strongest matches.
• Improve chunking: bigger semantic chunks (sections, FAQs) give more meaning per token than tiny fragments.



2️⃣ Compress the Context

When the query touches many docs, run a cheap pre-step:
• remove boilerplate
• keep only query-relevant sentences
• or summarize long docs into a compact evidence pack
Cuts tokens without losing grounding.



3️⃣ Use a Clean Prompt Template + Smart Routing
• Use a tight prompt template → no duplicate instructions, no long examples.
• Cache answers for common queries.
• Route simple questions to a cheaper model, only use long context for hard ones.


---

You estimate with thumb rules and one baseline scenario. Cost is an important aspect in almost every company/teams.
Let’s assume a simple case and scale from there.
Assumed baseline (you can swap your own numbers):
➤ 10k queries per month
➤ ~2k tokens per query total (retrieved context + prompt + output)
1️⃣ Embeddings — cheap and mostly one-time
Thumb rule: Embeddings are not your monthly problem.
➤ You embed docs offline
➤ Re-embed only when docs change

Example:
If you embed ~5M tokens of docs at ~$0.0001 per 1k tokens
→ ~$0.50 one-time, maybe a few dollars monthly for updates

Most startups spend <$20/month here.

2️⃣ Vector search — fixed floor, not per query
Thumb rule: Vector DB cost is mostly baseline capacity.
➤ You pay for the cluster, not each lookup

Example:
Managed AWS OpenSearch / similar
→ ~$50–150/month for a small prod setup
This barely changes from 10k → 50k queries.

3️⃣ LLM calls — this is the big lever
Thumb rule:
Monthly LLM cost ≈ queries × tokens_per_query × $ per 1M tokens.
Example:
10k queries × 2k tokens = 20M tokens
At ~$3 per 1M tokens
→ ~$60/month

At 100k queries?
→ ~$600/month
This is what scales fastest.

4️⃣ Infra — linear (good thing)
Thumb rule: Infra cost scales gently with traffic.
➤ Lambdas, Functions, small services
Example:
10k queries → ~$20–50/month
Rarely the surprise.
5️⃣ The hidden multiplier: caching
Thumb rule: Every cache hit avoids an LLM call.
➤ Semantic cache for similar questions
➤ FAQ-style queries repeat a lot

Example:
If 50% of queries hit cache
→ LLM cost drops from $60 to ~$30

At scale, caching can be the difference between “fine” and “too expensive”.
✅ Put it all together (10k queries example)
➤ Embeddings: ~$10
➤ Vector DB: ~$100
➤ LLM: ~$60
➤ Infra: ~$30
Total ≈ $200/month
At 100k queries, expect roughly 10×, unless caching improves.

BOTTOM LINE:
RAG cost isn’t mysterious.
Pick a query volume, assume tokens per query, and multiply.
Most early-stage RAG systems land in the low hundreds per month and scale predictably if you design caching early.


---

here is not one definite universal approach that works for this but possible options to explore here based on usecase.
Most RAG systems optimize for relevance, not repeatability.
Queries get rewritten, vector search is non-deterministic, and chunks drift as data evolves.
So ‘What does Section 42 say about liability?’ can surface different sources on each run.

⸻

The Deterministic Pattern ( Our possible options )
You don’t fight randomness in the LLM—you remove it from retrieval.
1. Normalize the query
Every variation is collapsed into one canonical form using lowercasing, lemmatization, and entity extraction.
“Section 42 liability” → section_42_liability
2. Hybrid retrieval with fixed rules
Combine exact and semantic search using a fixed formula:
FUSED_SCORE = 0.6 × BM25 + 0.3 × Vector + 0.1 × Metadata
BM25 anchors exact matches like “Section 42.1”.
Fixed weights mean the same math every time.
3. Stable Top-K
The same scoring formula produces the same ordering, so the same K sources are selected on repeat runs.
4. Canonical context
Generate a fingerprint for the retrieved set, for example:
context_id = SHA256(query_norm + sorted_doc_ids)
That hash represents this exact context.
5. Semantic caching
Wire the pipeline as: query → context → answer.
If the query repeats, the system reuses the same context and answer path.
6. Source-first citations
Cite by document and section, like IFRS_9#2.1, not chunk_47.
Chunks drift. Document IDs don’t.

---

When someone says “RAG over millions of PDFs”, they’re really asking about a search system + LLM, not a vector DB demo.

I’d break it into five parts: ingestion, embeddings, retrieval, generation, and monitoring.

1️⃣ Ingestion is offline, not request-time
At this scale, ingestion must be async.

➤ Stream PDFs from object storage (S3/GCS)
➤ OCR only when needed
➤ Normalize to clean text
➤ Chunk meaningfully (≈512–2k tokens with overlap)
➤ Attach rich metadata: doc ID, page, section, product, timestamp

None of this runs in the user path.

2️⃣ Embeddings + index built for scale
You don’t embed on demand.

➤ Dedicated embedding jobs (GPU or batched)
➤ Distributed ANN index (Milvus, Qdrant, Vespa, Elastic)
➤ Sharding + HNSW / IVF / PQ for tens of millions of vectors
➤ Metadata stored alongside vectors

Metadata filtering is the first gate — vector search is the fallback.

3️⃣ Tight online retrieval + generation
The request path stays minimal:

query → metadata filter → cache → ANN search → rerank → LLM

➤ Most queries are narrowed by metadata before vector search
➤ Many never hit the vector DB at all
➤ Rerank only a small candidate set
➤ Send just 5–10 chunks to the LLM

More chunks usually hurt.

4️⃣ Where caching actually fits
Caching controls cost and p95 latency.

➤ Query → answer cache (FAQs, repeats)
➤ Query → retrieval cache (top-k chunk IDs)
➤ Index / data cache (hot vectors + parsed PDFs)

Real path becomes:
query → cache → (miss) retrieve + LLM → write back

5️⃣ Monitoring closes the loop
Without this, drift is silent.

➤ Retrieval quality (recall@k)
➤ Answer quality / feedback
➤ Latency + cache hit rates
➤ Re-embed and re-shard as data or models change

BOTTOM LINE:
RAG at this scale isn’t an LLM problem.
It’s a metadata-first search + caching architecture problem.

---

Response time under 1 second is achievable if you do the right tweaks.

1️⃣ Stream Output Tokens

👉 Stream tokens as they’re generated instead of waiting for complete responses. This reduces Time To First Token (TTFT) to under 500ms and dramatically improves perceived performance even when total generation time stays the same. Users see content immediately rather than waiting 8+ seconds for full responses.

2️⃣ Add Semantic Caching

👉 Cache responses for similar queries to cut response times by 50% for repeated patterns. Semantic caching is particularly effective for FAQs and common questions in RAG systems.

3️⃣ Prompt Caching

👉 Place static content (system prompts, instructions) at the beginning and dynamic content (user queries) at the end to leverage KV cache.

4️⃣ Use Smaller Models

👉 Obvious point, but does your task needs the largest LLMs? Maybe a smaller one does the job too.

🏷️ LLM, AI, Python, Large Language Models, AI Engineering, Coding Interview, AI Interview


--- 

# MCP Best practices

## Error Handling

- Always wrap tool calls in try-catch blocks.
- Provide meaningful error messages.
- Gracefully handle connection issues.

## Resource Management

- Use `AsyncExitStack` for proper cleanup.
- Close connections when done.
- Handle server disconnections.

## Security

- Store API keys securely in `.env`.
- Validate server responses.
- Be cautious with tool permissions.

## Tool Names

- Tool names can be validated according to the format specified here.
- If a tool name conforms to the specified format, it should not fail validation by an MCP client.