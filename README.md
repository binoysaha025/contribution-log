# Contribution 1: server : add "token healing" support

**Contribution Number:** 1
**Student:** Binoy Saha
**Issue:** https://github.com/ggml-org/llama.cpp/issues/5765
**Status:** Phase I Complete

**Why I chose this:**
Token healing sits at the intersection of tokenizer internals and inference server architecture - exactly the kind of systems-level ML engineering I want to develop depth in. I have built an xLSTM from scratch and a Go AI agent gateway with an ensemble critic, so I understand both the model side and the server infrastructure side. 
This issue lets me go deeper on how sampling pipelines actually work in production inference engines.

The existing draft PR (#19238) uses a retry-loop approach which is architecturally incorrect for a performance-critical server — it makes multiple inference calls instead of one. I want to implement this the right way by masking logits at sampling time, constrained to tokens whose byte strings start with the prefix of the rewound 
token. This is a roadmap feature that the maintainer (ggerganov) has explicitly flagged as good first issue, and it has direct applications in code completion and autocomplete tooling.

**Problem Description:**
When a completion prompt ends mid-token (e.g., "Five, Four, Thr"), the tokenizer greedily splits "Thr" as its own token. The model then 
sees a complete token boundary and often ignores the partial word, producing completions like ", Two" instead of "ee, Two, One". Token healing fixes this by rewinding one token and constraining the sampler to only consider tokens whose byte representation starts with the rewound token's bytes.

Expected: prompt "Five, Four, Thr" -> completion "ee, Two, One"
Current:  prompt "Five, Four, Thr" -> completion ", Two" (ignores partial token)

- examples/server/server.cpp (server endpoint logic)
- common/sampling.cpp (llama_sampling_sample — where logit 
  filtering should be implemented per maintainer guidance)
- common/sampling.h

## Understanding the Issue

The server has no mechanism to handle mid-token prompts. The existing 
draft PR (#19238) attempts a retry-loop workaround but requires 
multiple inference passes — architecturally incorrect for a 
production inference server.

### Affected Components

- `examples/server/server.cpp` — completion endpoint request handling
- `common/sampling.cpp` — llama_sampling_sample, the correct hook 
  per maintainer guidance in the issue thread
- `common/sampling.h` — API surface for new parameter
- `src/llama.cpp` — tokenizer byte-string lookup for prefix matching

---

## Reproduction Process

### Environment Setup

```bash
# Clone fork
git clone https://github.com/<your-username>/llama.cpp.git
cd llama.cpp

# Build with CMake
cmake -B build
cmake --build build --config Release -j$(nproc)

# Download a small model (e.g. Qwen2-0.5B-Instruct-GGUF)
# Start the server
./build/bin/llama-server -m your_model.gguf --port 8080
```

Key challenge: llama.cpp requires CMake 3.14+ and a C++17 compiler. 
On Mac, ensure Xcode Command Line Tools are installed. Build time 
on M-series chips is approximately 5-10 minutes.

### Steps to Reproduce

1. Start llama-server with any instruction-tuned model
2. Send a completion request with a mid-token prompt:

```bash
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Five, Four, Thr",
    "n_predict": 5,
    "stop": ["\n"]
  }'
```

3. Observe output ignores "Thr" prefix — produces ", Two" or similar
4. Expected output: "ee, Two, One"

### Reproduction Evidence

- **Commit showing reproduction:** [Link to be added after setup]
- **My findings:** The sampling pipeline has no step that identifies 
  a partial final token, rewinds it, or constrains the logit 
  distribution at sampling time. The existing PR #19238 adds a retry 
  loop in server.cpp but requires multiple forward passes. The 
  correct fix is a single forward pass with logit masking inside 
  llama_sampling_sample, which ggerganov confirmed in the original 
  issue thread as the right location.

---

## Solution Approach

### Analysis

The root cause is that llama_sampling_sample applies no constraint 
on which tokens can be sampled as the first completion token. When 
a prompt ends mid-token, the model has no way of knowing it should 
continue the partial word — it simply samples from the full 
unconstrained distribution.

### Proposed Solution

Implement logit masking at the sampling layer. Before sampling the 
first completion token, zero out all logits for vocabulary tokens 
whose byte string does not start with the rewound prefix bytes. 
Sample normally from the constrained distribution. This keeps the 
operation to a single inference pass.

### Implementation Plan (UMPIRE)

**Understand:**
Mid-token prompts cause incorrect completions because the sampling 
pipeline has no mechanism to constrain next-token predictions to 
prefix-matching candidates. The model sees a complete token boundary 
and samples freely from the full vocabulary.

**Match:**
The existing sampling pipeline in `common/sampling.cpp` already 
manipulates the `llama_token_data_array` logit distribution for 
temperature, top-k, and top-p sampling. Token healing follows 
the exact same pattern — iterate the array, zero out logits for 
tokens that don't meet the constraint, then sample normally. 
The JVP for this pattern already exists in the codebase.

**Plan:**
1. `common/common.h` — add `token_healing` bool to `gpt_params`
2. `common/sampling.h` — add `healing_prefix` string field to 
   sampling context struct
3. `examples/server/server.cpp` — on receiving a completion 
   request with token_healing=true, tokenize the prompt, extract 
   the last token's byte string, strip it from the prompt, store 
   bytes in sampling context
4. `common/sampling.cpp` in `llama_sampling_sample` — add prefix 
   filtering step before sampling: iterate vocabulary, zero logits 
   for tokens whose decoded bytes don't start with healing_prefix, 
   clear prefix after first token sampled
5. Update server API docs to include token_healing parameter
6. Write tests validating correct prefix continuation

**Implement:** [Branch link — to be added]

**Review:**
- [ ] Follows llama.cpp C++ style (snake_case, existing patterns)
- [ ] Single inference pass only — no retry logic
- [ ] token_healing defaults to false — no regression on existing tests
- [ ] Edge cases handled: empty prefix, unicode multi-byte, 
      special tokens, single character prefix
- [ ] Checked CONTRIBUTING.md — PR requires passing CI and 
      at least one manual test example in the PR description

**Evaluate:**
Manual curl tests against the three example prompts from issue 
#5765. Verify single inference pass via server timing logs. 
Confirm token_healing=false preserves all existing behavior. 
Run existing server test suite to check for regressions.

---

## Testing Strategy

### Unit Tests

- [ ] Test 1: Mid-word prompt produces prefix-matching first token
- [ ] Test 2: Full-word prompt behavior unchanged when healing=false
- [ ] Test 3: Unicode multi-byte prefix handled correctly
- [ ] Test 4: Single character prefix constrains correctly
- [ ] Test 5: Empty prompt edge case handled gracefully

### Integration Tests

- [ ] Server endpoint accepts token_healing parameter
- [ ] Healing disabled by default — no regression on existing tests
- [ ] Works across multiple model architectures

### Manual Testing

```bash
# Test 1: number sequence
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Five, Four, Thr", "token_healing": true,
       "n_predict": 5}' | jq '.content'
# Expected: "ee, Two, One"

# Test 2: word completion
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My favorite band is Green Da",
       "token_healing": true, "n_predict": 3}' | jq '.content'
# Expected: "y"

# Test 3: baseline preserved
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Five, Four, Thr", "token_healing": false,
       "n_predict": 5}' | jq '.content'
# Expected: ", Two" (existing behavior unchanged)
```

**Status:** Phase II Complete


## Implementation Notes

### Week 3 Progress

Implemented token healing in the llama.cpp server via logit masking 
at sampling time — a single-pass approach rather than the retry-loop 
used by the existing stalled PR (#19238).

Work completed:
- Added `token_healing` bool to `gpt_params` (default false, no 
  regression when disabled)
- Added `healing_prefix` byte-string field to the sampling context
- In `server.cpp`: on a completion request with `token_healing=true`, 
  tokenize the prompt, extract the final token's byte string, strip 
  it from the prompt before inference, and store the prefix in the 
  sampling context
- In `llama_sampling_sample` (`common/sampling.cpp`): added a prefix 
  filtering pass that zeroes logits for all vocabulary tokens whose 
  decoded bytes do not start with `healing_prefix`, then samples 
  normally from the constrained distribution; prefix is cleared after 
  the first token so generation continues unconstrained

### Challenges

- Mapping vocabulary token IDs back to their byte strings for the 
  prefix comparison required using the model's token-to-piece 
  conversion rather than assuming UTF-8 directly — needed to handle 
  byte-fallback tokens correctly
- Edge case: multi-byte UTF-8 prefixes where the rewound token splits 
  a codepoint; handled by comparing at the raw byte level, not 
  character level
- Ensuring the constraint applies only to the first generated token, 
  not the whole sequence

### Approach Decisions

- Chose logit masking over retry generation because retrying requires 
  multiple inference passes — unacceptable for a production server. 
  Single-pass masking keeps it O(1) in forward passes.
- Located the logic in the sampling layer per maintainer guidance 
  (ggerganov pointed to `llama_sampling_sample` in the issue thread)

---

## Testing Strategy

### Manual Testing

Validated against the example prompts from issue #5765:

```bash
# Mid-word prompt — healing on
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Five, Four, Thr", "token_healing": true, 
       "n_predict": 5}' | jq '.content'
# Produces prefix-matching continuation ("ee, Two, One")

# Same prompt — healing off (baseline preserved)
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Five, Four, Thr", "token_healing": false,
       "n_predict": 5}' | jq '.content'
# Produces original unconstrained behavior
```

### Validation Steps

- [x] Mid-word prompt produces prefix-matching first token
- [x] Full-word behavior unchanged when `token_healing=false`
- [x] Single-character prefix constrains correctly
- [x] Unicode multi-byte prefix handled at byte level
- [ ] Confirmed no regression on existing server test suite
- [ ] Verified single inference pass via server timing logs

### Next Steps (Phase IV)

- Open PR against ggml-org/llama.cpp
- Address maintainer feedback
- Add automated test case to server test suite if requested

**Status:** Phase III Complete
