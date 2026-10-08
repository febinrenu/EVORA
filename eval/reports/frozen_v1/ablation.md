| Configuration | n | Hit@1 | Hit@5 | MRR | Neg prec | TTFA p50 (ms) | Contribution metric (change) |
|---|---|---|---|---|---|---|---|
| full system | 19 | 0.38 | 0.46 | 0.44 | 0.83 | 172 | - |
| no Track-centric multi-granular retrieval (C1) | 19 | 0.38 | 0.54 | 0.45 | 0.83 | 88 | hit@1 0.38 (+0.00) |
| no Attribute grounding (C2) | 19 | 0.38 | 0.46 | 0.44 | 0.83 | 180 | hit@1 0.38 (+0.00) |
| no Plan-then-verify: LLM planner (C3a) | 19 | 0.15 | 0.31 | 0.20 | 0.00 | 268 | hit@1 0.15 (-0.23) |
| no Plan-then-verify: verification (C3b) | 19 | 0.38 | 0.46 | 0.44 | 0.83 | 166 | negative_precision 0.83 (+0.00) |
| no Clarify-once referent memory (C4) | 19 | 0.38 | 0.46 | 0.44 | 0.83 | 174 | reask_count 1.00 (+0.00) |
| no Topology-aware cross-camera linking (C5) | - | - | - | - | - | - | skipped: needs the footage re-indexed with the switch off |
| no Motion-gated adaptive sampling (C6) | - | - | - | - | - | - | skipped: needs the footage re-indexed with the switch off |
