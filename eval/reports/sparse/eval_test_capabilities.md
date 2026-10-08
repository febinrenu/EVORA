System: ours

| Capability | n | Hit@1 | Strict Hit@1 | Hit@5 | Strict Hit@5 | MRR | Cam acc | Ts err (s) | Neg prec | Count acc | Count MAE | Within 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object | 6 | 0.33 | 0.33 | 0.50 | 0.50 | 0.42 | 0.50 | 4.9 | n/a | n/a | n/a | n/a |

Not evaluated (no reliable ground truth in the available data):
- count: MEVA labels only actors in annotated activities (parked cars and bystanders are unlabelled), so annotated counts are lower bounds and exact counts cannot be scored
- colour: no colour labels in MEVA; needs the blind human labels from scripts/colour_label_tool.py
- carrying: no bag-carrying ground truth: MEVA's carried objects are class 'other', one bag actor in 6 cameras
- path: no cross-camera identity ground truth (MEVA actor ids are per clip and camera)

System: b0

| Capability | n | Hit@1 | Strict Hit@1 | Hit@5 | Strict Hit@5 | MRR | Cam acc | Ts err (s) | Neg prec | Count acc | Count MAE | Within 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object | 6 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 | 1.00 | 3.1 | n/a | n/a | n/a | n/a |

Not evaluated (no reliable ground truth in the available data):
- count: MEVA labels only actors in annotated activities (parked cars and bystanders are unlabelled), so annotated counts are lower bounds and exact counts cannot be scored
- colour: no colour labels in MEVA; needs the blind human labels from scripts/colour_label_tool.py
- carrying: no bag-carrying ground truth: MEVA's carried objects are class 'other', one bag actor in 6 cameras
- path: no cross-camera identity ground truth (MEVA actor ids are per clip and camera)

System: null

| Capability | n | Hit@1 | Strict Hit@1 | Hit@5 | Strict Hit@5 | MRR | Cam acc | Ts err (s) | Neg prec | Count acc | Count MAE | Within 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object | 6 | 0.33 | 0.33 | 1.00 | 1.00 | 0.60 | 1.00 | 18.1 | n/a | n/a | n/a | n/a |

Not evaluated (no reliable ground truth in the available data):
- count: MEVA labels only actors in annotated activities (parked cars and bystanders are unlabelled), so annotated counts are lower bounds and exact counts cannot be scored
- colour: no colour labels in MEVA; needs the blind human labels from scripts/colour_label_tool.py
- carrying: no bag-carrying ground truth: MEVA's carried objects are class 'other', one bag actor in 6 cameras
- path: no cross-camera identity ground truth (MEVA actor ids are per clip and camera)
