# Ingest throughput (M2)

Measured on the development laptop (RTX 4060 8 GB, 16 GB RAM, Windows 11), GPU profile, models in half precision,
MEVA 1080p clips of 300 s unless stated. `video-s/s` is seconds of video indexed per second of wall time. Numbers vary
by up to 20% between runs; they are measurements of this machine, not guarantees.

## Whole ingest (L0 scenes + L1 tracks + L2 attributes, ReID and events)

| Camera (300 s, 1080p) | people tracks | wall time | video-s/s |
|---|---|---|---|
| G299 (crowd) | 175 | 72-76 s | 4.0-4.2 |
| G330 (crowd) | 207 | 71-76 s | 4.0-4.2 |
| G423 | 57 | 45-47 s | 6.4-6.7 |
| G419 | 29 | 41-42 s | 7.2-7.3 |
| G300, G328 (parking lot) | 12 and 30 | 40 s and 51 s | 7.5 and 5.9 |

Eight cameras of 300 s one after the other: 440 s of wall time, 5.5 video-s/s overall.
Four cameras, two workers: 188 s wall (6.4 video-s/s) against 251 s with one worker (4.8), about 25% faster.

## Where the time goes (G330, 71 s)

| Stage | Time | Note |
|---|---|---|
| Decode 1080p H.264 | 2.3 s | 9000 frames, not the bottleneck; L0 and L1 share one decode |
| Image preprocessing for the scene and crop embeddings | about 25 s | HF processor 8 ms per image; now 5.5 ms with the same output (cosine 0.9999) |
| SigLIP2 forward pass | about 9 s | 2 ms per image in half precision, batches of 32 |
| Detection and tracking (yolo26n, 1280 px, about 1260 sampled frames) | about 15 s | 4 fps sampling floor; the motion gate raises it on busy scenes |
| L2 (colours, ReID, events) | 21 s | 201 tracks; ReID with osnet_ain about 28 s per 1000 tracks |

## Choices that were measured

| Choice | Result |
|---|---|
| Detector size 640 vs 1280 on 1080p | 1280 finds 3 to 4 times more small people and cars; costs about 3x per frame |
| Sampling floor 1 fps vs 4 fps | 4 fps: G423 12 -> 57 people, G300 1 -> 12; ingest 40-75 s instead of 30-50 s per clip |
| yolo26n vs yolo26s at 1280 | same wall time (the GPU is not the limit), more detections (G299 244 vs 175 tracks); not adopted: more fragments |
| One detector shared under a lock vs one per thread | 4 cameras with 2 workers: 251 s -> 188 s |
| Tracker start thresholds 0.25 vs 0.2/0.15 | same people, 58% more car tracks (parked cars split): kept 0.25 |
| Re-ID model | osnet_x0_25 same-camera AUC 0.80, cross-camera 0.63; osnet_ain_x1_0 0.87 and 0.78 (WILDTRACK) |

## Other layers

* L3 captions (local qwen3-vl 4b, deferred until every camera is searchable): about 3 s per track, 84 captions for 87 people tracks of two 156 s clips in about 4 minutes.
* WILDTRACK, 7 cameras of 200 s at 2 fps (crowd, about 400 tracks per camera): 58-132 s per camera.
* CPU profile (earlier spike, same laptop): yolo26n 34 fps on the CPU, detector input 640; use for fall-back only.

## Not done

ONNX / OpenVINO export of the detector for CPU-only machines: not measured, the GPU path is the target.
