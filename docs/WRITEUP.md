# evora: conversational search over multi-camera footage

Problem HNX26EPS05. Numbers below are copied from `eval/reports/report.json` (generated 2026-10-08, code commit 4688ade, thresholds frozen at 9325b7e). Re-create them with `make eval`. Where a result does not support a claim, this document says so.

## Abstract

evora indexes recorded or live camera footage so that a person can ask questions in plain language ("did a red car pass through the main gate in the last hour?") and get back a camera, a time, and a clip with the object marked. The system indexes tracks (things that move) rather than frames, remembers places it has been told about so that it asks only once, links people and vehicles across cameras, watches for standing questions, and keeps faces blurred and data on the machine unless told otherwise. On the one real multi-camera set we could evaluate (MEVA, eight school cameras, one five-minute window) the honest result is mixed. The system answers "nothing like that here" correctly where the baselines never do (negative precision 1.00 against 0.00, n = 16), but on questions about objects that are present it is not better than a random moment inside the same camera and time window, and it is clearly worse than a plain frame-similarity baseline. We report both, with the sample sizes, and describe what we believe causes it.

## 1. Problem and constraints

Input is several cameras whose clocks are unknown, whose files may use odd codecs, and which the team has never seen. Questions are in natural language and mention places ("main gate") that no model knows. Scoring covers retrieval accuracy, localisation of camera and time, whether the system re-asks for something it was already told, and latency. We added four requirements of our own: it must work with the network off, faces must be hidden by default, every answer must be checkable against the footage, and nothing may be asked twice.

## 2. System

Ingestion runs one worker per camera and builds the index in layers so that queries work early: L0 scene embeddings, L1 detection and tracking with the best few crops per track, L2 attributes, events and re-identification features, L3 captions. The clock is read from the filename, the container metadata, the on-screen timestamp (local vision model) or entered by hand, and every answer shows two timestamps: wall clock and seconds into the camera's own file.

Storage is one folder per site (a workspace): a SQLite database, a vector store and a media cache. Memory facts live in the workspace, so a place learned on one site never leaks into another.

The query path parses the question (a fast path with no network call, otherwise a planner model with a local fallback), resolves places against memory, retrieves with structured filters plus image and text similarity, applies time and zone logic, optionally verifies the top candidates with a small local vision model, and composes the answer from templates over the evidence. The language model plans; it does not write the evidence.

Platform features built around that path:

- **Ask once.** An unknown place triggers one clarification (pick the camera, optionally draw the gate line). The binding is stored with aliases and survives a restart; a test kills the server mid-session and checks that the original question and two paraphrases then ask nothing. Corrections replace a fact rather than adding one.
- **Retroactive zones.** Drawing a line after indexing recomputes the crossing events from stored track points in seconds, with no re-ingest.
- **Standing questions and alerts.** A question such as "notify me if anyone enters the parking zone after 8 pm" becomes a rule over the event stream. Alerts appear on screen and, if configured, on a phone; phone pushes are capped per watch so a burst cannot flood the phone, while every alert is still stored.
- **Replay as live and real cameras.** Recorded files can be replayed as RTSP streams and analysed as if live; real RTSP cameras are analysed the same way and recorded into a rolling buffer so that their alerts have playable clips and thumbnails.
- **Evidence.** Each answer's clip and frames can be exported as a pack with provenance and a SHA-256 manifest. The manifest is signed with a per-workspace Ed25519 key; `python -m evora.evidence.pack verify` checks hashes and signature offline. A pack carries its own public key, so the signature shows the manifest is unchanged since export; tying a pack to one installation needs the fingerprint that installation published (`/api/evidence/signer`).
- **Privacy.** On-prem mode sends every model call to the local runtime and blocks non-loopback network connections (tested, including DNS and the asynchronous event loop). Faces are blurred in everything served unless a viewer asks for them with a stated reason, which is written to an audit log. There is no face recognition: identity is appearance only and local to the workspace.
- **Operations.** `make doctor` checks that a machine is ready (and `--fix` runs the listed repairs after asking); `make up` starts everything.

## 3. Contributions and what the evidence says

| | Contribution | Evidence in this repository |
|---|---|---|
| C1 | Track-centric retrieval | Implemented. In our ablation, frame-level retrieval beat the full system (Hit@1 0.46 against 0.38, n = 19). We do not claim a benefit. Detector recall on small, distant people limits the track layer. |
| C2 | Attribute grounding (colour, carrying) | Implemented and unit-tested. No query in the evaluated set exercises it (MEVA has no colour or carrying labels), so no effect is measured. |
| C3 | Plan then verify | Parsing the question into constraints is essential: replacing the parsed plan with raw-text search drops Hit@1 from 0.38 to 0.15 and negative precision from 0.83 to 0.00. The verification step has no measured effect on the evaluated queries. |
| C4 | Ask once memory | The restart test passes and re-asks are 0 across 28 queries. Ask precision and recall were not measured (no evaluated query involves an unknown place). |
| C5 | Topology-aware linking | Implemented with strict thresholds. The ablation row was not run, and MEVA has no cross-camera identity labels, so path accuracy is not evaluated. |
| C6 | Motion-gated layered indexing | Throughput is measured (below). The ablation row needs a re-index with the gate off and was not run. |

## 4. Evaluation setup

Data: MEVA, school site, one five-minute window (2018-03-09 10:10), eight cameras, split by camera into dev (3), test (2) and judge-sim (1). The judge-sim split shares the recording day with dev and test, so it checks unseen cameras, not unseen footage. Queries come from MEVA's activity annotations (objects present on a camera in a window, and negatives for classes absent from a camera's whole annotation). Ground truth labels only actors that take part in annotated activities, so a correct but unlabelled object counts as a miss and object scores are conservative.

Systems: ours; a frame-similarity baseline (B0: whole-frame embeddings, text to image similarity); and a null baseline that returns a random moment inside the same camera filter and time window with no pixels. All three receive the same parsed constraints. Thresholds and weights were frozen before test and judge-sim were run; nothing was tuned on them. One boundary bug in time-of-day ranges was fixed after the freeze; the original results are kept in `eval/reports/frozen_v1` and the fix changed no object or negative result of ours.

Hardware: a laptop with an RTX 4060 (8 GB) for indexing; queries were answered with no language-model call (the no-LLM share is 1.00).

## 5. Results

Pooled over dev, test and judge-sim (object n = 18, negative n = 16):

| System | Object Hit@1 | Object Hit@5 | Camera accuracy | Negative precision |
|---|---|---|---|---|
| ours | 0.50 | 0.67 | 0.78 | 1.00 |
| null (random moment, same filter) | 0.56 | 0.94 | 1.00 | 0.00 |
| frame baseline (B0) | 1.00 | 1.00 | 1.00 | 0.00 |

Read this table as follows. The null baseline scores high because MEVA's annotated actors are dense in time, so chance is already strong; every object number must be read next to that row. The camera filter is given to all systems, which is why the baselines reach 1.00 camera accuracy. Our system is not better than chance on object Hit@1 and is worse on Hit@5; only the frame baseline is clearly above chance. Timestamp error after the boundary fix is of the same order for all systems (ours 11.7 s dev and 6.9 s test, null 9.5 and 2.9, B0 12.0 and 5.9), and we make no localisation claim. The one supported difference is honest negatives: ours answers "no" correctly in 16 of 16 queries where the others cannot answer "nothing there" at all.

How a hit is counted matters for that comparison. A returned window counts as correct when it overlaps a labelled window (widened by 2 s), however long it is. The frame baseline joins frames that are 2 s apart into windows that cover most of the asked minute, so on these short windows it is right almost by construction. Its own localisation shows this: on the low-chance windows of the dev split (1-minute windows where labelled people fill at most half the time, n = 11) the frame baseline's mean temporal overlap with the truth is 0.07, below the random baseline's 0.08 and below ours at 0.23, and its median timestamp error is 5.3 s against our 1.1 s (random 12.3 s). We read this as: the frame baseline finds the right minute, ours finds the right second where the detector saw the person, and neither number alone says which system is better. A stricter Hit@1 that requires the returned peak moment to fall inside the labelled window is being added to the harness and will be reported beside the current one.

Latency (dev, 28 queries, all without a language model): time to first answer p50 195 ms, p95 348 ms; time to verified answer is within 2 ms of that because verification did not run on these queries. The frame baseline answers in about 43 ms.

Ablation (19 queries, full system Hit@1 0.38, Hit@5 0.46, MRR 0.44, negative precision 0.83):

| Switch off | Hit@1 | Hit@5 | Neg. precision |
|---|---|---|---|
| Track-centric retrieval (frames instead) | 0.46 | 0.62 | 0.83 |
| Attribute scoring | 0.38 | 0.46 | 0.83 |
| Planner (raw text instead) | 0.15 | 0.31 | 0.00 |
| Verification | 0.38 | 0.46 | 0.83 |
| Ask-once memory | 0.38 | 0.46 | 0.83 |

Indexing throughput on the RTX 4060 laptop after profiling: about 30 video-seconds per second per camera for L0 and about 17 for L1; detector plus tracker about 85 frames per second on the GPU and 34 on the CPU for the small model; two parallel cameras was the best setting we measured.

Not evaluated, with the reason: counting (annotated counts are lower bounds), colour (no colour labels; needs human labels), carrying (no ground truth), cross-camera paths (no identity ground truth), ask precision and recall.

## 6. Qualitative notes

What works: a question about something that is not on a camera returns "no" with the nearest miss and its weak score, and the clip, thumbnail and frame at the moment can be opened from the answer. Restarting the server keeps the learned places. A replayed file raises an alert whose clip plays.

What fails: object questions on small, distant people (the detector misses them, so no track exists); questions on activities (put something down, get out of a vehicle), because there is no activity recogniser; and any colour or carrying question on this data, which we cannot score.

## 7. Limitations

- One site, one five-minute window, eight cameras. Every number carries its n and should be read with it.
- The null baseline matches or beats us on object presence. The system's retrieval quality on present objects is not demonstrated by this evaluation.
- Colour, carrying, counting, paths and ask metrics are implemented but not measured on independent ground truth.
- Live analysis was verified with loopback streams and test doubles; it has not been run on a physical camera.
- The indexing and verification models are small and local by design; recall on distant people is the main weakness.

## 8. Privacy and ethics

Faces are blurred by default in every served frame and clip; showing them requires a typed reason that is audited. There is no face recognition and no identity beyond appearance matching inside one workspace. On-prem mode keeps all processing on the machine and is enforced, not only requested. Recorded rolling buffers of real cameras stay on the local disk and are served only through the blurred routes. The public datasets are used under their stated research terms (MEVA, EPFL multi-camera sequences). Evidence packs are signed so alterations are detectable.

## 9. Reproducing

```
make setup && make setup-perception     # environment (perception adds the model stack)
make models                              # local model weights
make doctor                              # is this machine ready
make up                                  # backend and interface
make eval && make ablate                 # the numbers in this document
make offline-test                        # the suite with network rules on
python -m evora.evidence.pack verify <pack.zip>
```

The demo script is `docs/DEMO.md`; the held-out rehearsal protocol is `docs/JUDGE_SIM.md`; the architecture map is `docs/ARCHITECTURE.md`.

## 10. Licences

evora is released under AGPL-3.0 (`LICENSE`) because Ultralytics and BoxMOT, which it uses for detection, tracking and re-identification, are AGPL-3.0. Other components keep their own licences: SigLIP2 and the Qwen models (Apache-2.0), bge-small (MIT), OpenCV's YuNet face detector, MediaMTX, FFmpeg. Datasets and model weights are not committed.
