# Potential models to add

Hardware: 1x or 2x NVIDIA A6000 (48GB each, 96GB total)

Constraint: ~32B dense or ~30B MoE in FP8 on 1x; up to ~48B dense in BF16 on 2x.
Q4 quantisation extends range but with small accuracy penalty (MedMarks Section 3.10).

## Priority list

Based on MedMarks (Warner et al., 2026) findings and relevance to multi-turn
epidemic policy decision-making.

### 1. Baichuan-M2 32B (added, running Q4_K_M via Ollama)

- 32B dense, medical reasoning model
- MedMarks-V: 0.552, MedMarks-OE: 0.476
- +5.2 over base Qwen 2.5 32B on V, even larger gap on OE
- Rationale: strongest evidence that medical fine-tuning helps on open-ended
  reasoning tasks. Tests whether domain-specific training aids epidemic policy.

### 2. Qwen3 30B-A3B Thinking

- 30B MoE (3B active), fits easily on 1x A6000 FP8
- MedMarks-V: 0.559, MedMarks-OE: 0.413
- Punches above weight class (outperforms avg Large model per MedMarks Table 3)
- Rationale: direct within-family scale comparison vs our existing Qwen3 14B.
  Fast inference due to low active params.

### 3. MedGemma 27B

- 27B dense, medical fine-tune of Gemma 3 27B
- MedMarks-V: 0.502
- +4.1 over Gemma 3 27B base
- Rationale: second medical-specific model. Lower priority than Baichuan-M2
  (smaller OE gains, not evaluated on MedMarks-OE in the paper).

### 4. Olmo 3.1 32B Think

- 32B dense reasoning model
- MedMarks-V: 0.507
- Rationale: fully open (weights + data + training code). Useful as a
  reproducibility-friendly baseline. Lower medical performance than above picks.

## Does not fit

- Llama 3.3 70B Instruct: 70B dense, ~70GB FP8. Too tight on 2x A6000 with KV cache.
- gpt-oss 120b: 120B MoE total params. Does not fit.
- MiniMax M2/M2.1: 230B MoE. Does not fit.
- GLM 4.5 Air / 4.7 FP8: 106B MoE. Does not fit.
- Baichuan M3 235B, Qwen3 235B-A22B: far too large.
