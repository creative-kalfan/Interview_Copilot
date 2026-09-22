# Interview Copilot V2.1 — Walkthrough

Surgical latency improvements on top of the validated V2 architecture. No rewrite.

## Changes

### 1. Speculative partial-STT preparation (Phase 1)
- Both audio listeners (`AudioChannelListener`, `SoundcardLoopbackListener`) now emit a **full-buffer speculative partial** every `SPEC_PARTIAL_INTERVAL` (1.0s) while the interviewer is still speaking, in addition to the existing ~350ms tail preview.
- `TranscriptionWorker` accepts an optional `spec_queue`; full-buffer partial transcripts are forwarded to it after the UI preview.
- New `SpeculativeWorker` thread: runs cheap local `IntentClassifier` + BM25 (`DocumentChunker.get_top_k`) on partials with >= `SPEC_MIN_WORDS` (6) words. No LLM calls. No embeddings.
- `LLMStreamingWorker.run` reconciles: reuses speculative retrieval only when the final intent matches AND token-Jaccard similarity >= `SPEC_SIMILARITY_THRESHOLD` (0.6) via `spec_matches()`. Final transcript stays authoritative; state is cleared after each question.

### 2. Persistent clients + optional warmup (Phase 2)
- `LLMStreamingWorker` builds `client_groq` / `client_nvidia` once at init (kept alive, TLS/HTTP reuse) instead of per-request client creation.
- `LLMStreamingWorker.warmup()`: one 1-token `ping` per provider at startup, in a daemon thread. Config: `WARMUP_PROVIDERS` (default on), `WARMUP_TIMEOUT` (6s). Non-fatal on failure; failover path untouched.

### 3. P95 instrumentation + 429 retry (Phase 3)
- `PerformanceTelemetry`: new fields `t_llm_stream_created`, `provider_events` (with EOS-relative offsets), `spec_reused`, `spec_prep_ms`; events surface in `[PERF]` logs and benchmark output.
- Root cause of the ~10.4s P95, reproduced live: **Groq free-tier ITPM 429 -> NVIDIA fallback** (slow streaming; one observed fallback run took 44.7s end-to-end).
- Fix: on Groq 429, parse provider-suggested `try again in N.NNs`; if delay <= 3.0s, wait and retry Groq once before falling back. Verified live: a 2.01s retry completed in 3076ms vs ~12-15s via NVIDIA.

### 4. Stale-generation cancellation (Phase 4)
- Unchanged: `gen_id` guard in stream loop + `clear_text` on new question. Regression test 2 PASSED.

### 5. Final STT (Phase 5)
- Investigated: faster-whisper `transcribe()` is stateless per call; partial decode state cannot be reused without replacing the framework. Intentionally not changed. CUDA float16 / CPU int8 fallback and model kept. Measured final STT is now ~110-160ms median warm, so it is not the bottleneck.

### 6-7. Prompts and output budget
- GCA / Googliness system prompts and `LLM_MAX_TOKENS` unchanged.

### 8. Benchmark
- `python copilot.py --benchmark`: 10 realistic prompts (3 GCA, 3 behavioral, 2 follow-ups, 2 technical) x `BENCHMARK_REPS` (default 2) = 20 runs, SAPI-synthesized audio through the real pipeline. Reports median/P95 for all four EOS milestones, speculative prep latency, cold vs warm Groq stream-create, fallback frequency, NVIDIA fallback latency.

## Measured results (V2.1, 20-run benchmark)

| Metric | V2 baseline | V2.1 |
|---|---|---|
| EOS -> STT complete (median / P95) | 430.3 / 694.8 ms | 157.8 / 294.3 ms |
| EOS -> LLM request (median / P95) | 430.3 / 695.1 ms | 157.8 / 294.3 ms |
| EOS -> first useful token (median / P95) | 990.7 / 1916.9 ms | 598.5 / 1743.9 ms |
| EOS -> complete (median / P95) | 1161.9 / 10401.2 ms | 887.3 / 14885.3 ms |
| Groq stream create cold / warm | ~2300 ms cold | 1280.0 / 339.5 ms |
| Groq fallback frequency | n/a | 4/20 (429 retry saved 1) |

Note: V2.1 STT medians are lower partly because the benchmark process keeps Whisper warm; relative pipeline gains (persistent clients, speculative prep) are independent of that.

## V2.1.1 — Follow-up runtime bug fix

Diagnosed from the user's real session log (`logs/session_20260921_142513.log`).

**Failure points (two, both on the follow-up path):**
1. `DialogueManager.run` — `if speaker == "Candidate": continue`. All utterances spoken into the microphone arrive on the `Candidate` channel and were logged then discarded, so no LLM request was ever made for mic follow-ups. The initial GCA question only worked because it arrived on the `Interviewer` loopback channel (system audio).
2. `IntentClassifier.classify` — short questions either fell through to `("OTHER", False)` ("How?", "What?", "Then what?", "What changed?" → dropped in `DialogueManager`) or were classified as a brand-new `GCA` question ("What would you check first?" → previous context overwritten). Short wh-questions now classify as `FOLLOW_UP`; short challenge continuations ("What if ...", "And the trade-off?") too.

**Fixes:**
- Mic (`Candidate`) turns that are question-shaped or interviewer-style prompts now run through the same intent/retrieval/LLM path; spoken candidate narration ("I built a Django REST API ...") is still ignored. Guard: `_looks_like_question()`.
- `DialogueManager` inherits previous question + intent for `FOLLOW_UP`, and falls back to a fresh `GCA` question when a short question arrives with no prior context (cold start).
- No word-count minimum for follow-ups; VAD thresholds, models, prompts, retrieval, and latency architecture unchanged.

**Diagnostics added (INFO, greppable):** `[TRANSCRIPT]`, `[QUESTION_DETECTION]`, `[INTENT]`, `[FOLLOW_UP]`, `[PREVIOUS_INTENT]`, `[RETRIEVAL]`, `[LLM_REQUEST]`, `[LLM_RESPONSE]`, including explicit `DROPPED ...` lines at every stage that can discard an utterance.

**Regression test:** `test_follow_up_regression()` (TEST 7) — classifier/routing table, mic guard, context inheritance, and a real audio pass through `AudioChannelListener` → `TranscriptionWorker` → `DialogueManager` → `LLMStreamingWorker` for "Why?" and "What would you check first?" via the mic channel, plus a new unrelated GCA question and an ignored spoken answer.

Validation suite status: 7/7 PASSED (first-token median unchanged at 526ms).


## V2.2 — Interview Separation & Routing Engine (ISRE)

ISRE classifies the interview TYPE before GCA-specific reasoning. GCA is now one mode among several.

**Taxonomy (InterviewType enum):** `GCA, TECHNICAL, QUANTITATIVE, PERSONAL, PROJECT, BEHAVIORAL, FOLLOW_UP, OTHER`.

**Pipeline:** transcript -> question detection (`_looks_like_question` + small talk) -> `ISRE.classify(text, prev_type, prev_question)` -> interview type -> intent/answer strategy. Local heuristics only (structure + semantic cues + tech vocabulary + context), median **0.040 ms** per classification, max 3.8 ms. Logged as `[SEPARATION] type=... confidence=... latency_ms=...` per question.

**Routing rules (ordered fallback ladder):**
1. Small talk / greetings -> `OTHER` (`should_trigger=False`).
2. Deterministic math (`quant_answer`) & quantitative shape -> `QUANTITATIVE` (sub-millisecond, zero LLM latency).
3. Personal / background (`_PERSONAL_RE`) -> `PERSONAL` (ensures "Why did you move..." & "Why Google?" route to PERSONAL).
4. Project deep-dive (`_PROJECT_RE`) -> `PROJECT` (with phonetic aliases for STT mismatches like "gear arrows", "Careyros").
5. Direct technical definition/troubleshooting (`_TECH_DEFN`, `_TECH_TROUBLE`) -> `TECHNICAL` (never GCA-scaffolded).
6. Behavioral / STAR (`BEHAVIORAL_INDICATORS`) -> `BEHAVIORAL`.
7. Follow-up (`FOLLOW_UP`):
   - With parent context: inherits parent `interview_type` and intent (`FOLLOW_UP (parent=...)`). Tech justifications after PROJECT inherit PROJECT.
   - Without parent context: standalone follow-up shapes route to `FOLLOW_UP` (defaulting to GCA).
   - Standalone architecture and tool justifications route to `PROJECT` and `TECHNICAL`.
8. GCA cues & analytical questions -> `GCA`.
9. Candidate speech vs Question guard -> non-questions (`_looks_like_question` == False) ignored; unclassified questions fallback to `OTHER` (generic direct answer, **never defaulted to GCA**).

**Deterministic quantitative:** `quant_answer()` solves unambiguous arithmetic (percent-of, what-percent-of-X, percent increase/decrease, multiply/add/subtract, churn-remaining) with exact output and zero LLM latency (measured: 0.0ms TTFT, 21.3ms complete). Ambiguous quant falls back to the LLM quant prompt.

**Answer formats:** every route outputs natural spoken language starting with `ANSWER:`. Non-GCA routes avoid frameworks or labels. GCA outputs `ANSWER:` + `FOLLOW-UP CUE:`. Behavioral strictly enforces verified STAR stories with multi-provider hardened `NO VERIFIED STORY FOUND` guardrail.

**Preemption & Cancellation:** `LLMStreamingWorker` checks `not self.prompt_queue.empty()` directly inside the stream generation loop to immediately abort superseded generations and discard stale chunks.

## Validation & Regression Suite Results

All 8 comprehensive test suites ran against live audio and provider endpoints:

```
===========================================================================
======================== OVERALL VALIDATION SUMMARY =======================
===========================================================================
1. Real Audio Pipeline (10 Prompts): PASSED (Median E2E First UI Token: 1704.6ms)
2. Stale-Generation Cancellation:    PASSED
3. Provider Failover (5 scenarios):   PASSED
4. No-Hallucination Guardrail (5/5): PASSED
5. GCA & Googliness Formatting:      PASSED
6. Speculative Partial-STT Prep:     PASSED
7. Follow-Up Routing + Audio Path:   PASSED
8. ISRE Separation & Routing:        PASSED
===========================================================================
```

