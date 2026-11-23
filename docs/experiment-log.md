# ARC-AGI-3 Experiment Log

Team: Alton Maseko, Elias Sepuru, Paballo, Banele Mkhize.
Moodle due: 27 Oct 2026, 23:59 SAST.

## How to use this file

- Each member gets one section. All four sections have the same skeleton.
- One run (or one debug session that taught you something) = one entry.
  Copy the blank entry below, paste it at the bottom of your own section,
  and fill it in straight after the run, not later.
- Entry IDs: your initials plus a running number. AM-001, ES-001, PB-001, BM-001.
- Version names: `RAND`, `DQN-v0` (baseline), `DQN-v1`, `DQN-v2`,
  `PPO-v0`, `PPO-v1`, `PPO-v2`. Ablations: `DQN-v2-minus-<thing>`.
- Never put shaped training return in the "official score" field. The report
  must keep the two apart.
- Write "What it shows" in your own words before looking at anything else.
  That field is what the report's analysis gets built from.
- Leave a field as `n/a` rather than deleting it, so every entry keeps the same shape.

## Blank entry (copy this)

```
### <ID> | <date> | <version> | <type: sanity / baseline / improvement / ablation / debug>

**Why this run**
- Limitation seen (what earlier evidence pointed here, cite entry IDs):
- Hypothesis:
- Intervention (the actual change, not just a hyperparameter):

**Setup**
- Train / dev games:
- Held-out games:
- Seeds:
- Interaction budget (train steps, eval action limit, eval episodes):
- Checkpoint used for eval:
- Key hyperparameters:
- Package versions (arc-agi, sb3, sb3-contrib, python):
- Notebook section / commit / file:

**Results**
- Official ARC-AGI-3 score (local or Kaggle, eval set, scoring version):
- Levels / games completed:
- Actions used (include failed runs):
- Per-seed numbers:
- Shaped training return (training only, NOT the score):
- Runtime + hardware (training and eval kept separate):

**Evidence**
- Raw log path:
- Figure / table path:
- Success case (replay or trajectory):
- Failure case (replay or trajectory):

**Reading it**
- What it shows:
- What is still uncertain:
- Next step:
```

---

## Alton Maseko (2544308)

### Owned tasks
-

### Environments I used during development
| Game ID | Used as (train / dev / held-out) | First used (entry ID) |
|---|---|---|
| | | |

### My contribution (one paragraph for the report)
-

### Entries

<!-- paste entries below this line, newest at the bottom -->

---

## Elias Sepuru (1427726)

### Owned tasks
-

### Environments I used during development
| Game ID | Used as (train / dev / held-out) | First used (entry ID) |
|---|---|---|
| | | |

### My contribution (one paragraph for the report)
-

### Entries

<!-- paste entries below this line, newest at the bottom -->

---

## Paballo (3134698)

### Owned tasks
-

### Environments I used during development
| Game ID | Used as (train / dev / held-out) | First used (entry ID) |
|---|---|---|
| | | |

### My contribution (one paragraph for the report)
-

### Entries

<!-- paste entries below this line, newest at the bottom -->

---

## Banele Mkhize (2663814)

### Owned tasks
-

### Environments I used during development
| Game ID | Used as (train / dev / held-out) | First used (entry ID) |
|---|---|---|
| | | |

### My contribution (one paragraph for the report)
-

### Entries

<!-- paste entries below this line, newest at the bottom -->
