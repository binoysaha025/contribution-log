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
