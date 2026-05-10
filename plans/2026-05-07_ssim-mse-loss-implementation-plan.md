# SSIM-MSE Loss Implementation Plan (2026-05-07)

## Scope
Implement an SSIM-based training loss option for synthoseis-pre-train, designed for 3D seismic data with zero-mean assumptions, float-valued amplitudes, and masked invalid regions. Add a mixed SSIM+MSE mode and expose all relevant controls in both train entry points.

## Success Criteria
- New loss option is available in `train.py` and `train_multi_datasets.sh`.
- SSIM implementation is 3D and memory-aware.
- SSIM uses zero-mean assumption (does not estimate local mean terms).
- Loss output semantics match existing losses: lower is better, near 0 is best.
- Tests cover shape/math/CLI behavior.
- Session summary is written in `.agentic-docs/`.

---

## Phase 1 - Algorithm and Interface Design
### Checklist
- [ ] Planning step: define SSIM variant for zero-mean seismic data and masked supervision.
- [ ] Plan review step: verify constraints against user requirements and memory limits.
- [ ] Plan execution step: finalize formulas, defaults, and CLI schema before coding.
- [ ] Code review step (constructive critic / anti-AI-slop): challenge assumptions, remove hand-wavy terms, confirm naming consistency.
- [ ] Testing step: design focused unit tests for math and argument validation.
- [ ] Documentation step: record decisions and tradeoffs in session summary draft notes.

### Deliverables
- Loss math spec and parameter list.
- CLI option spec in both shell and Python entry points.

---

## Phase 2 - Core Loss Implementation
### Checklist
- [ ] Planning step: choose module location and API shape for loss function/class.
- [ ] Plan review step: review memory behavior of 3D Gaussian-window SSIM implementation.
- [ ] Plan execution step: implement masked zero-mean 3D SSIM and mixed SSIM+MSE criterion.
- [ ] Code review step (constructive critic / anti-AI-slop): inspect tensor allocations, avoid unnecessary temporaries, simplify hotspots.
- [ ] Testing step: add unit tests for numerical behavior and edge cases.
- [ ] Documentation step: write implementation notes and rationale in session summary draft.

### Deliverables
- New loss module in `src/synthoseis_pre_train/`.
- Clear interfaces for use inside training loop.

---

## Phase 3 - Training Integration and CLI Exposure
### Checklist
- [ ] Planning step: identify all touchpoints in `train.py` and `train_multi_datasets.sh`.
- [ ] Plan review step: verify backward compatibility for existing `mse`/`huber` options.
- [ ] Plan execution step: integrate criterion selection and pass-through flags end-to-end.
- [ ] Code review step (constructive critic / anti-AI-slop): check user-facing messages, defaults, and validation clarity.
- [ ] Testing step: run syntax checks and targeted tests for CLI and runtime wiring.
- [ ] Documentation step: update session summary with integration details and usage examples.

### Deliverables
- `--loss-type` includes SSIM-MSE option in launcher.
- `--loss_type` includes SSIM-MSE option in trainer.
- Additional SSIM tuning flags surfaced and validated.

---

## Phase 4 - End-to-End Validation and Finalization
### Checklist
- [ ] Planning step: define end-to-end validation matrix (small run + resume + monitor enabled).
- [ ] Plan review step: review expected failure modes (OOM, NaN, mask sparsity, monitor thread issues).
- [ ] Plan execution step: run E2E checks and fix defects.
- [ ] Code review step (constructive critic / anti-AI-slop): final pass for readability, maintainability, and no brittle hacks.
- [ ] Testing step: confirm unit + smoke coverage and summarize any residual risks.
- [ ] Documentation step: finalize `.agentic-docs` session summary with results and follow-ups.

### Deliverables
- Verified E2E behavior.
- Final session summary document in `.agentic-docs/`.
