System: ours

| Capability | n | Hit@1 | Hit@5 | MRR | Cam acc | Ts err (s) | Neg prec | Count acc | Count MAE | Within 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| object | 8 | 0.62 | 0.75 | 0.69 | 0.75 | 6.9 | n/a | n/a | n/a | n/a |
| negative | 4 | n/a | n/a | n/a | n/a | n/a | 1.00 | n/a | n/a | n/a |
| activity (diagnostic) | 7 | 0.00 | 0.00 | 0.03 | 0.20 | 210.4 | 0.50 | n/a | n/a | n/a |

Not evaluated (no reliable ground truth in the available data):
- count: MEVA labels only actors in annotated activities (parked cars and bystanders are unlabelled), so annotated counts are lower bounds and exact counts cannot be scored
- colour: no colour labels in MEVA; needs the blind human labels from scripts/colour_label_tool.py
- carrying: no bag-carrying ground truth: MEVA's carried objects are class 'other', one bag actor in 6 cameras
- path: no cross-camera identity ground truth (MEVA actor ids are per clip and camera)

System: b0

| Capability | n | Hit@1 | Hit@5 | MRR | Cam acc | Ts err (s) | Neg prec | Count acc | Count MAE | Within 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| object | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 8.6 | n/a | n/a | n/a | n/a |
| negative | 4 | n/a | n/a | n/a | n/a | n/a | 0.00 | n/a | n/a | n/a |
| activity (diagnostic) | 7 | 0.00 | 0.20 | 0.17 | 0.40 | 119.9 | 0.50 | n/a | n/a | n/a |

Not evaluated (no reliable ground truth in the available data):
- count: MEVA labels only actors in annotated activities (parked cars and bystanders are unlabelled), so annotated counts are lower bounds and exact counts cannot be scored
- colour: no colour labels in MEVA; needs the blind human labels from scripts/colour_label_tool.py
- carrying: no bag-carrying ground truth: MEVA's carried objects are class 'other', one bag actor in 6 cameras
- path: no cross-camera identity ground truth (MEVA actor ids are per clip and camera)
