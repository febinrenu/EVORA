# Judge simulation

A 45-minute dress rehearsal on footage nobody tuned on, with 20 questions nobody on the query side has seen. It is run at
about T+12:00, before the feature freeze. It exists to find crashes, confusing moments and clarify mistakes, not to tune.

## Rules

1. **The questions stay out of git.** The person running the simulation writes them, and the query and evaluation side
   must not see them before the run. Keep the file at `data/judge_sim/queries.yaml` (the `data/` folder is ignored by
   git). Only this protocol and the empty skeleton are committed; the skeleton lives in `docs/judge_sim_skeleton.yaml`.
2. **The footage is held out.** Use own-footage scenarios 6, 9 and 10 from the recording script (umbrella, the late
   parking entrance, two people walking together) or a separate five-minute segment recorded for this purpose, or a MEVA
   window nobody used for tuning.
3. **No tuning afterwards.** Fix crashes and blockers only. Do not adjust thresholds, prompts or weights using these
   queries.
4. **A fresh workspace**: `evora_WORKSPACE=judge-sim make up ARGS="--open"`.

## The 20 questions

Write them the way a judge would, in plain sentences, using the names a judge would use for places, not names the system
has seen.

| Slots | Purpose | Example shape |
|---|---|---|
| 4 | Negatives: nothing matches | "Did a blue truck enter the loading bay?" (it did not) |
| 4 | First time a place is named (must ask once) | four different places, each asked about once |
| 1 | A paraphrase of an earlier place (must not ask) | "the front entrance" after "main gate" |
| 1 | Asked after a server restart (must not ask) | repeat an earlier place after `Ctrl-C` and `make up` |
| 3 | Path across cameras | "Where did the person with the red top go?" |
| 2 | Counting | "How many people walked past the lobby together?" |
| 2 | Attributes (colour, carried object) | "Anyone carrying an umbrella?" |
| 1 | Time of day or a relative time | "after 8 pm" or "in the last ten minutes" |
| 1 | Standing question | "Tell me when someone enters the parking area." |
| 1 | Describe | "What happened near the back door?" |

At least four of the 20 must be negatives and at least ten must name a place.

## Ground truth

Write the answer for each question before you ask it, from the director's log of the recording. Use the same format as
the evaluation query files (see `eval/queries/_example.yaml`): the verdict, and for positives the camera and a time
window with a UTC offset. Keep a margin of two seconds on windows. For a path question list the hops in order.

## Running it

1. Start the stopwatch. Upload the footage, read each clock aloud, start ingest. Note the time until the first question
   can be answered ("Searchable now").
2. Ask all 20 questions through the interface exactly as a judge would, including every clarification. Restart the
   server once, midway, and ask the post-restart question.
3. For each question fill the scoring sheet below as you go.

## Scoring sheet

One row per question.

| Column | Meaning |
|---|---|
| id, text | the question |
| clarify | asked / not asked, and whether that was expected |
| ttfa | seconds from pressing Enter to the first answer |
| verdict | correct or wrong |
| camera | the first evidence is on a correct camera |
| time error | seconds between the evidence peak and the ground truth window centre |
| confusing | anything that made the operator hesitate |
| notes | crashes, odd copy, slow moments |

## What to record at the end

- Time until the first question could be answered.
- Share of correct verdicts and of first evidence on the right camera, split into positives and negatives.
- Number of clarification questions asked when none was expected (target: zero) and missed ones when one was expected.
- Crashes and any state that needed a restart.
- A short list of confusing moments for the interface owner.

Then fix only what is broken, and file the rest as known issues.
