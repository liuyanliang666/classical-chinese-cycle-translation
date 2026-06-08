# Lu Xun OpenAI API Modernization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable script that converts each Lu Xun source record into modern daily Chinese with an OpenAI-compatible chat completions API.

**Architecture:** Add a focused CLI script under `scripts/` with pure helper functions for loading input, building chat payloads, parsing API responses, retrying failures, appending JSONL records, and skipping existing ids. Keep HTTP implementation in standard library `urllib` so the project does not need another dependency.

**Tech Stack:** Python standard library, pytest, OpenAI-compatible `/chat/completions` HTTP API.

---

## Chunk 1: Script And Tests

### Task 1: Test batch conversion behavior

**Files:**
- Create: `tests/test_modernize_luxun_with_api.py`
- Create: `scripts/modernize_luxun_with_api.py`

- [ ] **Step 1: Write failing tests**

Create tests that verify:
- a JSON list of strings is loaded and invalid JSON shapes are rejected;
- existing ids in the JSONL output are skipped during resume;
- successes write `target_modern`, while repeated API failures write `error`;
- chat completion response content is extracted and stripped.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_modernize_luxun_with_api.py -q
```

Expected: fail because `scripts.modernize_luxun_with_api` does not exist yet.

- [ ] **Step 3: Implement minimal script**

Add `scripts/modernize_luxun_with_api.py` with:
- `load_texts(path)`;
- `record_id(index)`;
- `load_seen_ids(path)`;
- `build_messages(text)`;
- `build_payload(text, model, temperature)`;
- `extract_chat_content(response)`;
- `OpenAIChatClient`;
- `modernize_with_retries(client, text, retries)`;
- `run_batch(...)`;
- `build_parser()` and `main()`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_modernize_luxun_with_api.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Run syntax verification**

Run:

```bash
python -m compileall scripts/modernize_luxun_with_api.py tests/test_modernize_luxun_with_api.py
```

Expected: both files compile.
