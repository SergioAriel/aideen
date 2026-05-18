# Associative Memory Status

Last updated: 2026-04-18

## Validated Default

- `ASSOC_BANKS=1` remains the default stable path.
- The corrected `AR_PAIRS_PER_SEQ` benchmark now queries a random stored pair
  instead of always querying the first pair. This is a measurement fix, not a
  model behavior change.
- Current baseline sanity run:
  `AR_N_TRAIN=250 AR_N_EVAL=20 AR_GAPS=0,64,256 AR_PAIRS_PER_SEQ=1 AIDEEN_CTX_LEN=32 cargo run -p aideen-training --features wgpu --bin assoc_recall --release`
  produced finite learning below random baseline on all tested gaps:
  `gap0=0.7355`, `gap64=0.6088`, `gap256=0.5935` versus random `3.9120`.

## Experimental Useful Pieces

- `ASSOC_READ_BETA=4.0`: useful for multi-bank content addressing. With
  `ASSOC_BANKS=1`, it is behaviorally inactive because the read softmax has a
  single candidate.
- `AIDEEN_ASSOC_BANKS`: experimental override for the number of associative
  banks. Default remains `1`; supported experimental range is `1..=4`.
- `AIDEEN_ASSOC_SLOT_ANCHOR=1`: experimental slot-identity source for associative
  write/read reconstruction. It improved slot/source diversity but did not
  recover the single-binding baseline when paired with 4 banks.
- `AIDEEN_ASSOC_TRANSITION_GATE=1`: experimental learned transition gate using
  existing associative projections. The first validation was worse than default;
  keep it out of default behavior.
- `AIDEEN_ASSOC_REUSE_MATCH=1`: experimental allocation policy that reuses a
  matching bank before filling an empty one. This exposed a regression for
  `ASSOC_BANKS=1`, so it is disabled by default.
- `AIDEEN_ASSOC_SLOT_STRIPE=1`: experimental slot/head routing. It makes each
  slot write only one local token phase, so the effective associative capacity is
  closer to `h_slots * ASSOC_BANKS` instead of four duplicated copies of the same
  early transitions. Short validation improved the multi-binding case but did
  not solve it.
- `AIDEEN_ASSOC_TIE_QK=1`: experimental shared address encoder. It routes query
  gradients into `W_k_assoc`, fixing the observed `Wq_assoc moves / Wk_assoc
  nearly static` asymmetry. It is not validated as stable default behavior.
- Enlarged associative backward workgroup storage is required before any
  `ASSOC_BANKS>1` variant can be valid. For `d=512`, `rank=32`, and 4 banks,
  the slot stride is `4 * (32 + 512 + 1) = 2180` floats, so `2048` is too small.
- Shared/tied query-key addressing remains plausible, but it was not sufficient
  by itself in the tested multi-bank variant.

## Rejected As Default

- Directly switching to `ASSOC_BANKS=4` is rejected as a default behavior.
- Reason: it degrades the single-binding baseline because the current write
  policy fills available banks with early adjacent transitions, not necessarily
  useful bindings.
- `AIDEEN_ASSOC_BANKS=4 AIDEEN_ASSOC_TRANSITION_GATE=1` is rejected as a default
  behavior after a short run stayed near random.
- `AIDEEN_ASSOC_BANKS=4 AIDEEN_ASSOC_SLOT_ANCHOR=1` is rejected as a default
  behavior: it improved over the transition-gate variant but still regressed
  against the stable one-bank baseline.
- Reusing matched banks by default is rejected because one-bank memory relies on
  protecting the first durable write rather than repeatedly overwriting it.
- `AIDEEN_ASSOC_BANKS=4 AIDEEN_ASSOC_SLOT_STRIPE=1 AIDEEN_ASSOC_TIE_QK=1
  AIDEEN_ASSOC_SLOT_ANCHOR=1` is rejected as a recommended training setting
  after a longer `AR_N_TRAIN=300` run collapsed to `loss=23.0258` on all tested
  gaps. Short-run improvement was not enough to validate the dynamics.
- Replacing the associative write gate with the mean FPM dimension-level write
  budget is rejected. It preserved the coupling intent but regressed the
  single-binding baseline toward random, because FPM dimension plasticity is not
  the same semantic event as associative binding.

## Known Limitation

- Multi-binding recall is not solved in the default path. With corrected random
  queried pair selection, this run:
  `AR_N_TRAIN=120 AR_N_EVAL=8 AR_GAPS=0,64,256 AR_PAIRS_PER_SEQ=4 AIDEEN_CTX_LEN=32 cargo run -p aideen-training --features wgpu --bin assoc_recall --release`
  remained near random: `gap0=3.6579`, `gap64=3.6253`, `gap256=3.6356` versus
  random `3.9120`.
- This is a controlled capability limit, not a numerical instability: no NaN,
  no `loss=23`, and no crash were observed in that run.
- The best short experimental run so far used
  `AIDEEN_ASSOC_BANKS=4 AIDEEN_ASSOC_SLOT_STRIPE=1 AIDEEN_ASSOC_TIE_QK=1
  AIDEEN_ASSOC_SLOT_ANCHOR=1` with
  `AR_N_TRAIN=120 AR_N_EVAL=8 AR_GAPS=0,64,256 AR_PAIRS_PER_SEQ=4
  AIDEEN_CTX_LEN=32`, producing `gap0=3.3405`, `gap64=3.2916`,
  `gap256=3.3858` versus random `3.9120`. This is better than the unstriped
  multi-bank path but still not a solved recall mechanism.

## Structural Cause

The current associative write policy is capacity-driven, not selectivity-driven.
With one bank, the first write often captures `KEY -> VAL`, so the benchmark can
work. With multiple banks, additional banks capture transitions such as
`VAL -> KEY`, filler transitions, or `KEY -> QUERY`, and the read path averages
or retrieves from polluted banks.

Additional findings:

- Slot routing matters. Without explicit routing, slots duplicate the same writes,
  so nominal capacity is not actual usable capacity.
- Address learning matters. `Wq_assoc` moved while `Wk_assoc` stayed nearly
  static; tying Q/K made `Wk_assoc` move, but did not by itself solve recall.
- Long-run stability is still unresolved for the stronger experimental
  combination. The next fix should target a stable write controller, not raw
  capacity or address sharpness alone.
- Directly substituting FPM write budget into the associative gate is too blunt:
  it starves valid bindings. The bridge should use FPM/H state as contextual
  input to a binding decision, not as a scalar replacement for that decision.

## Architecture Correction

The target is not two independent memories.

FPM and associative memory must remain different in role but coupled in control:

- FPM: state continuity, slow plastic persistence, and stability signal.
- Associative memory: explicit key/value binding and retrieval.
- Shared bridge: read/write decisions must be conditioned on the common H/FPM
  fixed-point dynamics and associative confidence/usage.

Changes that make associative write/read independent from FPM/H are rejected as
model direction, even if they are useful diagnostic probes. Capacity-only changes
such as adding banks are insufficient unless they improve this coupled
selectivity.

## Next Validation Target

A multi-binding change is acceptable only if all are true:

- `pairs_per_seq=1` does not regress versus the current default.
- `pairs_per_seq=4` with random queried pair improves materially below random
  baseline on `gap=0,64,256`.
- Debug shows distinct write allocation/routing rather than identical writes
  across all slots/banks.
- No NaN, no `loss=23`, no DEQ invalid escalation.
