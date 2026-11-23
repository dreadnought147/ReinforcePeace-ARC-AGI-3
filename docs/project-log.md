# ARC-AGI-3 Project Log

Benny's rough running log: raw outputs, decisions, and findings as they happen, in
chronological order. This is the scratch material the report gets written from.
(Per-member experiment entries for individual runs still go in `experiment-log.md`.)

---

## 2026-10-25

### Part 2 setup ran locally (not Colab)

- Confirmed: running fully local now, not in Colab, not using shared Drive.
- `pip install` warnings about `ipython`/`pandas`/`pillow`/`packaging` version conflicts with
  Spyder/Streamlit were noise, unrelated to `arc-agi`/`stable-baselines3`. Install succeeded.
- `PROJECT_DIR` fell back to `./ARC_RL_Project` (the `except ImportError` branch), relative to
  wherever Jupyter was launched from. Confirmed `results/` and `models/` created there.
- Cell output: `Actions per step: 70. Results go to: ARC_RL_Project\results` — correct, matches
  `N_ACTIONS = 70` from the shared settings cell.

### Game list loaded (Part 2, game split cell)

25 public games fetched, anonymous API key issued automatically
(`3f35fb8e-021e-4b85-b1e9-397b9e781b04`, can register a real one at three.arcprize.org).

Split: every 4th game held out (`HELD_OUT_GAME_IDS = ALL_GAME_IDS[3::4]`), as fixed in Part 1.

**Training games (19):**
`ar25-0c556536, bp35-0a0ad940, cd82-fb555c5d, dc22-fdcac232, ft09-0d8bbf25, g50t-5849a774,
lf52-271a04aa, lp85-305b61c3, ls20-9607627b, r11l-495a7899, re86-8af5384d, s5i5-18d95033,
sc25-635fd71a, sk48-d8078629, sp80-589a99af, tn36-ef4dde99, tr87-cd924810, tu93-0768757b,
wa30-ee6fef47`

**Held-out games (6):**
`cn04-2fe56bfb, ka59-38d34dbb, m0r0-492f87ba, sb26-7fbdac44, su15-1944f8ab, vc33-5430563c`

This exact list is what the brief means by "disclose every environment used during development" —
copy it into the report as-is, do not re-run the split later unless this cell's code changes
(the split is deterministic given game order, but note the API order isn't guaranteed stable
over time, so if we re-fetch on another day, re-paste the new list here rather than assuming
it matches).

### Task 0: real game field probe (ran on `ar25-0c556536`, the first training game)

Confirms the exact field names Task 1 (`ArcGymEnv`) needs. Anonymous API key and scorecard were
created automatically; game files cached to `environment_files\ar25\0c556536\`.

- `reset()` returns a `FrameDataRaw` object (pydantic model), not a plain dict.
- Confirmed fields: `state` (a `GameState` enum, e.g. `GameState.NOT_FINISHED`), `levels_completed`
  (int, starts at 0), `win_levels` (int, this game needs 8 levels to win), `available_actions`
  (list of ints, here `[1,2,3,4,5,6,7]` — all 7 actions offered), `frame` (list of grids; one grid
  here, shape `(64, 64)` — already full-size, no padding needed for this game), `game_id`, `guid`.
- `env.action_space` is a plain Python list of `GameAction` enum members (`ACTION1`..`ACTION7`),
  not a Gymnasium space — confirms the brief's warning that the official interface isn't Gymnasium.
- After one `step()`: state still `NOT_FINISHED`, `levels_completed` still 0 — one action wasn't
  enough to progress on this game, as expected.

**For Task 1:** `action_masks()` should read `available_actions` (ints 1-7, matching `GameAction`
values) rather than anything else; `frame_to_observation` should take `frame[-1]` (last grid) and
only pad when a game's grid is smaller than 64x64 — this game's wasn't.

### What the MDP actually looks like in `ArcGymEnv` (Task 1)

So basically working through what our state/action/reward setup actually is in this adapter:

- state is just the 64x64 picture `frame_to_observation` spits out, padded and scaled up
- actions is the 70 options, 6 simple ones plus the 64 click spots on the 8x8 grid, `index_to_game_action` does the mapping
- the transitions (how the game actually responds) we don't control or see, that's inside the real ARC engine, it's a black box to us
- reward is just levels_completed now minus before, nothing else added
- discount factor isn't even part of this adapter, that's set later in DQN/PPO's own settings
- starting state is whatever reset() gives back once a game id gets picked at random from our list
- and it ends either because terminated (game said WIN or GAME_OVER) or truncated (we hit our own step cap), kept as two separate things on purpose

### Task 1 self-check: PASS

Ran on games `bp35` (downloaded fresh to `environment_files\bp35\0a0ad940`) and repeatedly
re-fetched metadata for `ar25` during the random-action loop. Output ended `Task 1 PASS`.

Alias (Elias), writing up what each function does, as said:

I went over the function breakdown. I took the current game screen and resized it into a neat,
standardised picture for the bot to look at (that's `frame_to_observation`). For
`index_to_game_action`, I translated the bot's numbers, its button press, into something that can
actually happen as a direction on the game board. For `action_masks`, I checked which controller
buttons are available, so the bot doesn't waste moves trying something impossible. For `info`, I
gathered the current score, the progress, and the game-over status into a tiny card of info. For
`reset`, it just restarts the game and puts the bot at a random level, handing it the starting
screen. And `step` takes the bot's chosen move, plays it, and reports back the new screen, plus
whether the player died or won.

(Raw for now — gets cleaned up and tightened into report language only once we draw up the report.)

### Task 2 results schema, updated, with why each column is there

Original columns weren't enough to show whether the model was actually learning (no training
progress info, no official score). Added 3 columns. Full schema, with the reasoning as a team:

- `version` — version control, so we know which agent/stage (e.g. `dqn_v1_memory`) a row belongs to
- `algorithm` — which algorithm produced it (random / dqn / ppo), independent of version naming
- `seed` — so a result isn't attributed to luck, we need it to repeat across seeds before trusting it
- `game_id` — which game this row is about, standard
- `episode` — which episode/attempt, standard
- `levels_completed` — how much the agent actually got done in that episode
- `win_levels` — how many levels that game needs total to count as beaten, so completion can be read
  as a fraction, not just a raw number
- `won` — whether the episode ended in a win
- `actions_used` — how many actions it took, ties directly into the official score and efficiency
- `invalid_actions` — how many of those actions weren't even offered by the game (wasted moves);
  this especially matters for DQN since it can't use action masking the way PPO does, so this
  column is part of how we'll argue PPO's masking advantage
- `wall_seconds` — timing, for the compute/efficiency side of the comparison
- **NEW** `timesteps_at_checkpoint` — how much training the model had seen when this eval ran, so we
  can plot levels_completed / score against training progress and actually see whether it's
  learning or just flat
- **NEW** `official_score` — the brief's actual scoring formula, `(human_actions / agent_actions)^2`
  capped a little above 1, computed per won episode; levels_completed/won alone are not the real
  metric the course cares about, this is
- **NEW** `checkpoint_path` — which saved model file produced this row, so any number in the report
  traces back to one reproducible model file

Elias is implementing this next; output to follow once run.

### Task 2 self-check: PASS (on FakeArcEnv)

2 fake games x 2 episodes = 4 rows, all 14 columns present. Every row: 0 levels completed (of 3),
not won, 30 actions used (hit the max_steps cap), 0 invalid actions, ~0.001-0.002 s per episode,
`timesteps_at_checkpoint` = 0, `official_score` = NaN, `checkpoint_path` = None.

Why these numbers make sense:
- 0 invalid actions is expected: FakeArcEnv offers every action, so random can never pick an illegal one.
  On real games this column will start showing numbers, especially for DQN.
- 0 levels in all 4 runs is plausible: the fake target is 1 of 64 click blocks, so random has roughly
  a 1-in-3 chance of hitting it in 30 steps per episode. Random failing is the sanity-check floor doing its job.
- official_score NaN is correct: no human baseline exists for fake games, and nothing was won anyway.

Open protocol question for the protocol cell: every episode of a game reuses the same seed, so a
deterministic learned agent will play identical episodes. Either use 1 episode per game for learned
agents, or vary the seed per episode (e.g. seed + episode). Decide before Round 2.
**Resolved:** 1 episode per game (see protocol below).

### Task 2 evaluation protocol (Benny, took over Task 2 from Alton), as written

- **Games used for evaluation:** the held-out games, which is every 4th game on the list, the ones
  the bot never got to practise on.
- **Episodes per game:** just 1 round per game by default.
- **Action cap:** 500 moves max per game before it gets cut off.
- **Seeds:** we test it 3 times, using seeds 0, 1 and 2, so we know the results aren't just luck.
- **How a learned agent picks actions during evaluation:** it looks at the game screen, checks which
  buttons are actually allowed right now, and uses its trained brain to pick one.
- **What is recorded:** a big list of stuff for every single game: whether it won, how many levels it
  beat, how many moves it wasted on invalid buttons, how long it took, and its official score.
- **Official ARC score vs our local numbers:** the real official score is a formula comparing the
  bot's move count to a human's, and it only counts if the bot actually wins. Our local numbers
  (like points for finishing a level) are just rewards we made up to help it learn, and don't count
  for the real score.

**Corrections to fix before this goes in the report (Claude review):**
1. "Checks which buttons are allowed": only true for PPO. DQN can't use the mask, so it can pick
   invalid buttons (that's what the invalid_actions column catches). Also add that during
   evaluation the agent acts deterministically: always its best guess, no random exploring.
2. "Points for finishing a level are made up": not right. +1 per level is the game's own signal,
   not invented. The only made-up reward is the novelty bonus, which is training-only. And our
   local numbers (levels completed, actions used, etc.) are real measurements, not rewards. Three
   separate things: official score (ARC formula), our local measurements, shaped training reward.
3. Scoring granularity: the official score is per LEVEL (human vs agent actions on each level),
   but our `official_score` column only scores whole-game wins. Whole games (e.g. 8 levels) will
   almost never be won, so the column will be NaN nearly everywhere. Either track actions per
   level later, or state this as a known limitation in the report.

### Task 3 self-check: PASS (DQN baseline, on FakeArcEnv)

**Log 1, as said:**
I created the code that makes our first AI bot, which was quite cool. We make the bot play the game
for a few thousand turns to learn how to play, then it saves the model (the brain) into a file so we
can test it as a baseline. `train_dqn` spawns the AI and saves its progress so we don't lose what it
learned, which is something I missed when I first implemented it. `dqn_policy_function` acts as the
controller during the real test: it takes the game screen and presses whichever button the trained
brain thinks is the absolute best move. I didn't really understand this part, so I used AI to help
me implement it.

**Log 2, as written:**
My task was to train a DQN model as our first real baseline. The issue was making Stable-Baselines3
work with our custom Gymnasium adapter. I wrapped the environment with Monitor before feeding it to
the algorithm, otherwise it wouldn't log episode lengths and rewards during training.
Standard DQN ignores our action_mask: it outputs whatever action its network thinks is best, even if
that button is locked. Hacking SB3's predict function to force the mask was too risky for a baseline,
so I let the bot press invalid buttons: the evaluate loop catches it, counts it as an invalid_action,
and the game returns zero reward. I kept the action_mask argument purely so the function matches the
contract for the evaluation loop.

**Review (Claude, checked against the actual cell 38 code): CHANGES REQUESTED**
1. **The agreed DQN settings are missing.** The code calls `DQN("CnnPolicy", environment, seed=seed,
   verbose=0)` with SB3 defaults. The task card's settings (buffer_size=20_000, learning_starts=1_000,
   exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=1_000, train_freq=4)
   aren't there. This matters: SB3's default only explores for the first 10% of training, and only
   updates its target network every 10,000 steps, which is about 5 times in a 50k run. For a
   sparse-reward game that's a weak baseline, and the report has to list the settings actually used.
   Neither log mentions this.
2. "Saves its progress so we don't lose what it learned": it only saves once, at the very end. If a
   run dies halfway, everything is lost. Fine for a baseline, but don't claim mid-run saving.
3. "The evaluate loop catches it and the game returns zero reward": not quite. The ADAPTER (Task 1)
   catches the invalid action and never sends it to the game; it returns the same screen and 0
   reward itself. evaluate only counts it. Also, during TRAINING these invalid presses are wasted
   steps, so it's a training cost too, not just an eval number.
4. "Wrap with Monitor, otherwise it wouldn't work": overstated. DQN runs without Monitor; Monitor is
   only what records episode rewards and lengths for the learning curves. Still the right call.
5. "A few thousand turns": that's the self-check (2,000 steps). Real runs use 50,000 per seed.
6. PASS only proves it runs and saves, not that it learns anything.
7. **AI use:** Log 1 says AI helped write `dqn_policy_function`. That must go in the AI disclaimer,
   which currently says AI was only used for setup. Log 2 reads AI-drafted; if it was, declare that too.

### Task 3 fixes after review

- Added the task card settings to `DQN(...)`: buffer_size=20_000, learning_starts=1_000,
  exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=1_000, train_freq=4.
- Added SB3 `CheckpointCallback`, saving every 10,000 steps (prefix `{version}_seed{seed}_cp`), so a
  crashed run doesn't lose everything. Note: nothing loads these yet; resuming is a Task 7 job.
- Tried masking invalid actions at evaluation (set their Q-values to -infinity before argmax), then
  **reverted it**. Reasons: it made DQN train without the mask but test with it (a mismatch), it would
  force invalid_actions to 0 and hide the "DQN can't mask" evidence we want to report, and it would
  break with Adaptive Stacking's 70×k action space later. Decision: keep the most basic DQN as the
  baseline. Invalid presses are no-ops in the adapter and get counted, and they still eat into
  max_steps during training. That waste is unmeasured so far; it's a ready-made hypothesis for later.

### Task 3 extended self-check (smoke run, FakeArcEnv, 2,000 steps): PASS

- Settings confirmed in the model: buffer 20,000, learning_starts 1,000, train_freq 4, target update
  1,000, exploration 1.0 -> 0.05 over 30% of training, gamma 0.99, lr 1e-4, batch 32.
- 40 episodes, 9.4 s, ~250 gradient updates. Mean reward first half of episodes 0.6, second half 0.0.
- Greedy policy collapsed onto one action (10, every step). DQN 0.0 levels vs random 0.6 on 5 fake seeds.
- Reading: the first-half reward came from random button presses while exploration was still high
  (it decays to 0.05 by step 600). Once the agent acts on its own estimates, it presses one button
  forever. After only 250 updates its Q-values are basically noise, so this is expected at smoke
  scale and does NOT mean the setup is broken. Pipeline confirmed working end to end.
- Worth keeping: "greedy DQN collapses to a single action under sparse reward" is a real failure mode
  to watch for in the 50k real runs, and a good failure-case example for the report if it persists.

### Task 4: PPO baseline (MaskablePPO), Benny

Code matches the task card: `MaskablePPO("CnnPolicy", n_steps=512, batch_size=64, learning_rate=3e-4,
ent_coef=0.01)`, Monitor-wrapped, same CheckpointCallback as Task 3 (every 10,000 steps).

**Self-check (smoke run, FakeArcEnv, 2,000 steps): PASS**
- Settings confirmed: n_steps 512, batch 64, lr 3e-4, ent_coef 0.01.
- 42 episodes. Mean reward first half 0.86, second half 0.76 (flat, within noise).
- Greedy actions on 5 fresh screens: [22, 54, 9, 39, 22]. Not collapsed, unlike DQN in Task 3:
  it picks different buttons for different screens.
- Eval over 5 fake seeds: PPO 0.0 levels, random 1.0.
- Reading: only ~4 learning rounds at this budget (2,048 steps / 512), so no learning is expected.
  The training reward (0.8ish) comes from PPO sampling actions randomly-ish while it practises. At
  test time, `deterministic=True` makes it press its single top button every step; with a
  near-untrained policy that means repeating the same miss, which is why it does worse than random.
  Expected at smoke scale, not a bug.
- Worth keeping for the report: deterministic evaluation of an under-trained policy can score below
  random, because random at least varies its presses. Check whether this still holds in the 50k runs.

### Task 5: novelty bonus wrapper

Code matches the contract: md5 hash of the observation bytes, count per hash, bonus = 0.1 / sqrt(count),
`info["true_reward"]` and `info["novelty_bonus"]` recorded, counts not cleared on reset, mask passed through.

**Self-check (FakeArcEnv): PASS**
- Same screen twice: bonus 0.100 -> 0.071. Reward = true + bonus: True. Counts survive reset: yes.
- 200 random steps: 13 distinct screens, total bonus 7.89, total true reward 3.

**Log, as written:**
My task was to build the NoveltyBonus wrapper to fix the sparse reward problem. Since the game only
gives 1 point when you beat an entire level, the bot wiggles around randomly, gets zero points, and
never learns anything; it basically never finishes a level by accident.
Fix: a count-based bonus as a trail of breadcrumbs. Hashing the raw bytes of the observation gives a
unique string for every screen. The bot gets paid bonus_scale / sqrt(count), so fresh screens pay a
lot and old ones pay almost nothing.
I also noticed a risk with the evaluation: if we train on these fake points (shaped rewards), we
cannot report them as the official score, or Kaggle will reject the project.
Fix: I save the raw, unshaped reward in info["true_reward"], so the bot learns from the shaped points
but evaluation can still pull the real numbers.

**My understanding of each function:**
`__init__` sets up a dictionary that keeps count of every screen the bot has ever seen in its whole
life. `step` plays the move, hashes the new screen, checks the dictionary to see how boring that screen
is, calculates a small bonus, and hands back the boosted score while secretly logging the real sparse
score. `action_masks` passes the allowed-buttons list straight through, so MaskablePPO from Task 4
doesn't break. AI helped along the way.

**Review notes (Claude):**
1. "It never finishes a level by accident": not shown yet. On the fake game random gets ~1 level per
   episode. On real games it's likely rare, but that's the hypothesis to measure (random_v0 on real
   games), not a fact yet.
2. "A unique string for every screen STATE": it's per SCREEN, not per state. Two different situations
   that look identical (hidden key, pressed switch) share one fingerprint. And if a game shows anything
   that changes every step (a timer, a move counter), every screen counts as brand new and the bonus
   never fades. Check this on real games.
3. "Kaggle will reject the project": wrong. Kaggle never sees our rewards; it runs its own scoring.
   The rule is the brief's: don't present shaped training return as the ARC score.
4. "Evaluation pulls true_reward": evaluate never reads reward at all, it records levels_completed
   and actions from info. true_reward is for honest training curves (true vs bonus) in the report.
5. From the self-check: bonus 7.89 vs true reward 3 in 200 steps, so the bonus outweighs the real
   signal ~2.6x even on a 64-screen fake game. Real games have far more distinct screens, so the bonus
   will dominate harder. Risk (Part 0 warned about it): the bot learns to chase new screens instead of
   levels. bonus_scale is a knob to report and possibly test.

### Task 6: memory wrappers (FrameStackImage, AdaptiveStackImage), Benny

**Self-check (FakeArcEnv): PASS**
- Frame stack: newest screen lands in the last slot, the other 3 shift along unchanged. Correct.
- Adaptive: only the chosen slot (2) changed. Correct.
- Flat decode: option 41 -> (move 10, slot 1); flat mask length 280. Correct.

**Fix decided for the upstream break (point 2 below):** in `ppo_policy_function`, return
`int(action) if np.ndim(action) == 0 else action`, so baseline PPO still returns one number and
PPO + Adaptive Stacking returns the [move, slot] pair. The 280-action DQN point is not a bug: no fix,
just state it in the report as a fair-comparison caveat.

**Update: fix APPLIED** to the PPO controller in the Task 4 cell. ~~Status: pending~~ Done.
Re-ran Task 4 self-check after the fix: PASS, identical numbers to before (same seed, and baseline PPO
still returns a single int), so the hotfix didn't change baseline behaviour.

### Task 7: experiment runner (`run_version` + `VERSIONS`), Benny

**Log, as said:**
This is the experiment runner. I think of it as the master: it automates our testing, spinning up the
different versions of our bots, and writes the report card so we don't have to do it manually.
`run_version` takes one bot configuration, loops through our three seeds, trains the bot, tests it
(including on the held-out games), and saves the scorecards. Most importantly, it checks the save file
first and skips runs that are already finished, so we don't waste time repeating work. `VERSIONS` is a
dictionary mapping every bot experiment we need to run, from the dumb baselines to the fully upgraded,
novelty-seeking bots.
My thinking: we needed a runner that finally puts everything together. It needs our training
environment (the lambdas) and the held-out game IDs, so I instantiated the training environment with
all its memory and bonus wrappers. I used AI to help build this script, and I spent a lot of time
understanding what it's doing. From that I realised that if the run disconnects, we need to use the
CSV: reading RESULTS_CSV at the very top of run_version lets us skip the seeds we already ran safely,
so a rerun picks up where it left off.

**Review (Claude), checked against the code: CHANGES REQUESTED**
1. The resume logic, training dispatch and eval-environment trick (rebuild the same wrappers, swap
   `unwrapped.game_ids` to the held-out list) are correct.
2. **Version names don't match Part 1.** `dqn_v2_exploration` etc. should be `*_v2_memory_novelty`, and
   the **ablation versions `*_v2_novelty_only` are missing**. The brief asks for ablations (rigour, 25
   marks), so they're required, not optional.
3. **v1 to v2 changes two things at once.** v1 uses frame stacking, v2 uses Adaptive Stacking AND novelty,
   so any difference between them can't be pinned on novelty. v2 must use the same memory as v1.
4. **Wrapper order:** `NoveltyBonus(FrameStackImage(...))` hashes the stack of 4 screens, so it counts
   4-screen sequences, not screens. Far more sequences look new, so the bonus barely fades. Put novelty
   inside: `FrameStackImage(NoveltyBonus(ArcGymEnv(...)), 4)`.
5. **random_v0 is never run.** The sanity-check policy needs evaluating on the held-out games too.
6. **Learning curves:** every row gets `timesteps_at_checkpoint = total_timesteps`, so there's one eval
   point per run, no curve. The brief wants curves against interaction budget. Optional fix: also
   evaluate the 10k checkpoints CheckpointCallback already saves.
7. For novelty versions, Monitor sits outside NoveltyBonus, so its training curves show reward + bonus.
   Label those curves as shaped, or plot levels from eval instead.
8. "It's the master": careful, "master" is our name for the shared notebook. This is the runner.
   "Tests including the held-out games": it tests ONLY on held-out games.
9. Reminder: `HUMAN_ACTIONS_PER_GAME` is still empty, so official_score will be NaN everywhere.
10. AI use: declared.

### Crash in the first real run (random_v0 on held-out game cn04) -> 3 bugs found in Task 1's adapter

`IndexError: list index out of range` in `frame_to_observation` at `frame_object.frame[-1]`: a step
returned a frame object with an EMPTY grid list. Reading the adapter code (cell 29) to find out why
turned up three bugs. **Task 1 passed its self-check with all three in it.**

1. **Game over is never detected.** `state` comes back as the string "GameState.GAME_OVER", not
   "GAME_OVER" (Task 0 already printed "GameState.NOT_FINISHED"). The check
   `current_state in ["WIN", "GAME_OVER"]` compares whole strings, so it's always False: episodes
   never terminate, they only get truncated at 500 steps. The bot keeps pressing buttons on a dead
   game, which is very likely what produced the empty frame. Also: WIN would never end an episode.
2. **Empty frames crash the adapter.** Even with 1 fixed, guard against an empty grid list: keep
   showing the last good screen.
3. **Clicks all land on the top row.** The mapping used `x = idx % 64, y = idx // 64`, so for indices
   0-63 y is always 0 and x runs 0-63 along the top edge. The contract says the centres of an 8x8 grid:
   `row, column = divmod(idx, 8)`, `x = 8*column + 4`, `y = 8*row + 4`. Every click game would have
   been nearly unplayable.

Why the self-check missed them: it only checked shapes, info keys, and that an episode ended within
50 steps (truncation satisfies that). It never checked termination, click positions, or game over.
Nothing was saved from the crashed run (save happens after each seed's evaluate finishes).
~~Task 1: PASS~~ -> Task 1: PASS on shapes, but logic bugs found on first real run; fixed below.

**Fixes APPLIED to cell 29 (all three), checked against the pasted code:**
- `last_observation` fallback added in `__init__`; `frame_to_observation` returns it when the grid
  list is empty, and stores each good observation.
- Clicks: `row, column = divmod(action_index - 6, COARSE_GRID)`, `x = 8*column + 4`, `y = 8*row + 4`.
- `terminated = ("WIN" in current_state) or ("GAME_OVER" in current_state)`.
Task 1 self-check re-run: PASS (games bp35, ar25). Next: re-run random_v0, then the runner smoke test.

### FIRST REAL RESULT: random_v0 on the 6 held-out games, seeds 0-2 (25 Oct, ~11:05)

| game | win_levels | actions used (seed 0 / 1 / 2) | levels |
|---|---|---|---|
| cn04-2fe56bfb | 6 | 75 / 75 / 75 | 0 |
| ka59-38d34dbb | 7 | 100 / 100 / 100 | 0 |
| m0r0-492f87ba | 6 | 151 / 151 / 151 | 0 |
| sb26-7fbdac44 | 8 | 500 / 500 / 500 (hit our cap) | 0 |
| su15-1944f8ab | 9 | 37 / 40 / 35 | 0 |
| vc33-5430563c | 7 | 50 / 50 / 50 | 0 |

0 invalid actions everywhere (random only picks from the mask), 0 wins, 0 levels: the floor.

**Readings:**
1. Most episodes end well before our 500 cap, so the termination fix works: the games end
   themselves (GAME_OVER, most likely). Before the fix every episode would have run to 500.
2. On 4 games the episode length is IDENTICAL across seeds (75, 100, 151, 50). That suggests those
   games have a fixed move budget: you get N moves, then game over, whatever you press. su15 varies
   (35-40), so there something random presses do (dying?) ends it early. sb26 never ends by itself.
3. RL consequence: running out of moves gives no penalty, just an end with 0 reward. A no-reward ending
   looks the same to the agent as "nothing happened", so it can't learn that dying/wasting moves is bad
   except through missing future level rewards. Worth a sentence in the problem formulation.
4. Reproducibility gap: `evaluate` never seeds the policy's randomness (random_policy uses numpy's global
   generator), so the `seed` column labels the run but doesn't make random_v0 repeatable. Fix: call
   `set_global_seed(seed)` at the top of `evaluate`. Cheap, and it's what "reproducible protocol" means.
   (The runner already seeds training via train_dqn/train_ppo.)
5. official_score: NaN (nothing won, and no human baselines filled in).

**Decision (Benny):** add `set_global_seed(seed)` at the top of `evaluate` (Task 2 cell), so every
seeded evaluation is repeatable. Consequence: the random_v0 rows above were made BEFORE seeding, so
remove them from RESULTS_CSV and re-run the random loop once, so all reported rows follow one protocol.

**Seeding APPLIED** (`set_global_seed(seed)` is the first line of `evaluate`). Task 2 self-check re-run:
PASS. One fake row changed (fake-b episode 1 now completes 1 level instead of 0), which is expected:
the random choices now come from the seed instead of whatever state numpy was in. Same seed will now
always give these same rows.

### Task 7 runner smoke test (tiny DQN, 2,000 steps on ar25 only, 50-step episodes, 1 seed): PASS

- First run trained, evaluated on the 6 held-out games, saved 6 rows. Second run printed
  "Skipping selfcheck_runner seed 0, already found in results": the resume logic works.
- **Took ~2 minutes (11:14:04 to 11:15:54).** Nearly all of it is the ~2 s "fetched metadata" API call
  that happens on EVERY game reset (NORMAL mode = online). 40 resets x ~2 s = ~80 s of waiting.
  For real runs (50k steps, episodes ~75-150 steps, so hundreds of resets) that's 10-20 min of pure
  waiting per run, x 24 runs. Too slow for today. Fix: download all games once, then switch ARCADE to
  OFFLINE mode (the setup cell's own comment says to do this once games are cached).
- **Results (smoke DQN, not reportable):**

| game | actions used | invalid actions |
|---|---|---|
| cn04 | 75 | 0 |
| ka59 | 500 | 500 |
| m0r0 | 151 | 0 |
| sb26 | 500 | 0 |
| su15 | 500 | 0 |
| vc33 | 500 | 500 |

- **Reading:** on ka59 and vc33, EVERY press was invalid (500/500). The collapsed DQN (presses one
  button forever) picked a button those games don't offer, so nothing ever reached the game, it never
  hit game over, and it ran to our 500 cap. That's "DQN can't mask" shown concretely, the evidence the
  plan wanted to keep. Watch if it persists in the real dqn_v0 runs: if so, it's a key failure case.
  su15 also ran to 500 with 0 invalid: it survived on a harmless button, where random died in ~37.
- Cleanup: remove the selfcheck_runner rows from RESULTS_CSV before real runs.

### Runner smoke test re-timed after OFFLINE + Monitor filename fix: PASS
- 14 s, down from ~110 s: about 8x faster, so offline mode removed the per-restart server wait.
- Same 6 rows as the first smoke run (same seed, same results), monitor file saved: True, second run skipped.
- Estimate for real runs: ~4-6 min per 50k-step run x 24 runs (8 versions x 3 seeds) = ~2-2.5 hours.

### Real runs started (~11:50, 25 Oct)
dqn_v0_baseline seed 0: 45 training episodes in ~115 s (episodes mostly 500 or 100 steps), so roughly
7 min per 50k-step run. Notebook prints nothing during training (verbose change not picked up); progress
is visible in `ARC_RL_Project/results/<version>_seed<n>.monitor.csv` (one line per episode).

### Trained-agent Kaggle prep + report draft (Claude, while training ran, 25 Oct afternoon)
- `YearProject/kaggle-dataset/`: `pylibs/` (stable_baselines3 2.9.0, sb3_contrib 2.9.0, gymnasium 1.4.0,
  farama_notifications, copied from the local Anaconda install, no internet used) + `dataset-metadata.json`
  (`banelemkhize/arc-rl-agent-files`) + empty `model/` (chosen model goes in as `agent_model.zip`).
- `ARC-AGI-3-Kaggle-Starter/agent/my_agent_trained.py`: trained MaskablePPO agent in the kit's format
  (subclasses Agent, RESET on start/game over, is_done on WIN only, same preprocessing/mask/click mapping
  as the fixed adapter, STACK_FRAMES setting for memory models). `my_agent.py` left as the random agent.
- Tested: the copied libraries load `ppo_v0_baseline_seed0.zip` and pick legal actions under a mask.
- Report: `08 - Report (RLC template, 8 pages)/report.tex` + `references.bib` + `rlj.sty/.bst` (copied
  from RLC2026, template folder untouched) + `latexmkrc` (BibTeX path). Compiles via wsl-build.sh to
  `build/report.pdf`. Main text currently ends on page 5 with results still TODO; supplementary holds
  games list, reuse statement, AI declaration draft, contributions, evidence.
- Open items in the report: Paballo's surname, email format check, Adaptive Stacking citation details,
  CPU model, all results.

### 26 Oct: training finished, results + trained Kaggle agent
- All 24 runs + random done (162 eval rows). **Held-out: 0 levels for every version, every seed.**
- Training logs (true reward for non-novelty runs): levels completed during 50k training steps per seed:
  DQN v0 3/7/10, DQN v1 5/0/1, PPO v0 6/15/17, PPO v1 11/8/19. Novelty runs' Monitor reward includes the
  bonus (~10 per episode), so their true training levels were not logged: a limitation.
- Held-out invalid actions: 0 for all PPO (mask) and DQN v0; DQN v1 167 on vc33; DQN v2 167 on sb26/su15/vc33;
  DQN novelty-only 333 on sb26/su15/vc33. DQN with memory/novelty collapses onto unavailable actions.
- Random agent on Kaggle scored 0.07 (Benny).
- Kaggle trained agent: `ppo_v1_memory_seed2` chosen (most TRAINING levels, 19; no held-out selection),
  STACK_FRAMES=4, sampling (deterministic=False) because Kaggle RESETs and retries after game over.
  my_agent.py = trained agent; random kept as my_agent_random.py; kernel-metadata attaches
  `banelemkhize/arc-rl-agent-files`.
- Offline test of the agent caught a bug: `GameAction(3)` fails (enum value not a plain int); fixed by
  name lookup (`GameAction["ACTION3"]`) and a name-based action_number() for the mask. Re-test: all
  actions legal, clicks at block centres (e.g. x=20, y=12), empty frame handled, GAME_OVER -> RESET with
  memory cleared, is_done True only on WIN.

### Kaggle dataset uploaded (26 Oct): Kaggle unzipped the model
- `banelemkhize/arc-rl-agent-files` Version 1, 10.28 MB, 358 files. Kaggle auto-extracted
  `model/agent_model.zip` into a folder `model/agent_model/` (6 files: data, policy.pth,
  policy.optimizer.pth, pytorch_variables.pth, _stable_baselines3_version, system_info.txt).
- SB3 loads from a .zip, so the agent would have failed. Fix (no re-upload): if the zip is missing but
  the folder exists, the agent re-packs the folder into a zip in the temp directory, then loads it.
- Random agent (kit default) on Kaggle: public score 0.07, "ARC Prize 2026 — ARC-AGI-3 Starter - Version 1".

### Final ZIP built (26 Oct)
`10 - Submission Evidence and Final ZIP/ReinforcePeace.zip` (1.11 MB, 52 files): report PDF; Kaggle notebook
Version 2 (submission 57654331) + my_agent.py; training/experiments notebook (outputs cleared, former member's
name removed); figure script + results CSV and monitor logs; evidence (submission IDs/scores/timestamps,
nominated entry 57654331, Paper Track link + saved copy, contributions, 4 screenshots); README.
Checked: no credentials, no logged API key, no "Alton" in any file. Staging copy kept in `ReinforcePeace/`.

### Kaggle submissions (competition evidence for the ZIP)
| Submission ID | Submitted (UTC) | Description | Status | Public score |
|---|---|---|---|---|
| 57627136 | 2026-10-25 10:52:03 | Random starter agent (kit default), notebook Version 1 | COMPLETE | 0.07 |
| 57654331 | 2026-10-26 05:01:44 | Trained PPO v1 memory seed 2 (sampling), notebook Version 2 | COMPLETE | 0.00 |
Nominated entry: the trained one (57654331), Benny's decision, regardless of score.
Paper Track write-up posted (26 Oct): https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/memory-and-count-based-exploration-for-arc-agi-3

### Report TODOs cleared (26 Oct)
- Paballo listed by first name (Benny's call). CPU given as Intel Core, 6 cores / 12 threads. Code line in
  abstract points to the submitted notebook (57654331). AI declaration finalised from the logs (incl. code
  Claude supplied in review fixes, Kaggle agent, figure script, report drafting). Contributions written from
  the notebook's task assignment (Benny: 2, 4, 6, 8 + integration/report; Elias: 0, 1, 7; Paballo: 3, 5),
  to be confirmed by the team. Adaptive Stacking citation dropped (reference unconfirmed), mentioned as
  introduced in the course by G. Nangue Tasse.
- Only 3 red markers left, all waiting on Benny: trained Kaggle score (x2) and Paper Track link.
- Paper Track write-up drafted: `08 - Report/paper-track-writeup.md`, 487 words (limit 1500), [SCORE] to fill.

### Report edits from Benny's direction (26 Oct)
- AI declaration reworded: "Claude" (not Claude Code); members attempted their code first, then reviewed with
  Claude against the decided workflow; Claude used to diagnose/fix/refactor bugs; packaging and writing kept.
- Contributions: Benny configured the whole workflow and orchestrated (design, set-up, reviews, integration)
  plus Tasks 2, 4, 6, 8, experiments, Kaggle, report; Elias and Paballo implemented their tasks and submitted logs.
- Adaptive Stacking now cited: Nangue Tasse, Riemer, Rosman, Klinger, "Beyond Sliding Windows: Learning to
  Manage Memory in Non-Markovian Environments" (code github.com/geraudnt/adaptive-stacking); year shown as n.d.
  until the venue/year is confirmed.

### Report results written (26 Oct)
- `08 - Report/make_report_figures.py` builds `figures/` (training_levels, episode_length, invalid_actions;
  PDF + PNG preview) and `figures/summary.txt` from the results CSV and monitor logs.
- report.tex now has: real abstract, contributions, improvement motivation (honest: improvements were
  fixed in advance), Kaggle agent description, compute (174 min training, 71 s eval), results table,
  3 figures, iterative story, ablation/failure case (DQN invalid-action collapse: 1 / 3 / 6 seed-game
  pairs for memory / memory+novelty / novelty-only), discussion, limitations, conclusion.
- Compiles clean, no overfull boxes; references start on page 8, so main text is within 8 pages.
- Remaining TODOs: Paballo's surname, Kaggle code link + trained score + nominated entry, CPU model,
  member contributions, final AI declaration check, Adaptive Stacking citation.

### Kaggle random submission: checklist (to do on return)
1. kaggle.com: join "ARC Prize 2026 - ARC-AGI-3" AND the Paper Track, accept the rules, same team name on
   both (the Moodle ZIP is named after it).
2. Kaggle > Settings > API > create a token.
3. In WSL Ubuntu (the kit uses `make`, Python 3.12): `git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter`,
   put the token in the repo's `.kaggle/` folder as the README says, set username in `notebooks/kernel-metadata.json`.
4. `make setup`, then `make submit`, then `make status` until "complete". Agent left unchanged = random.
5. kaggle.com > your notebook > "Submit to Competition" > Output File `submission.parquet`.
6. Evidence for the ZIP (brief requires it): screenshot + note submission ID, score, timestamp, here in this log.

### Kaggle submission facts (from the official starter kit README, github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- Submission is a Kaggle notebook auto-built from `agent/my_agent.py` (`make submit`), validated by
  Kaggle, then on kaggle.com: open the kernel, "Submit to Competition", choose `submission.parquet`.
  5 submissions per day. Set your username in `notebooks/kernel-metadata.json` first.
- `MyAgent` must subclass `agents.agent.Agent`, implement `is_done(frames, latest_frame) -> bool` and
  `choose_action(frames, latest_frame) -> GameAction`.
- The kit's default agent IS a random agent: RESET on first call and after GAME_OVER (retries, doesn't
  stop), `is_done` only on WIN, random (x, y) for ACTION6 on the 64x64 grid, MAX_ACTIONS cap of 80 per game.
- Internet off in accelerated sessions. Model files / extra packages: not documented in the README.
- Consequences for our MyAgent: don't stop on GAME_OVER (press RESET instead), return GameAction objects,
  and note in the report that Kaggle's rules (retry after game over, 80-action cap) differ from our local
  protocol (episode ends at game over, 500 cap).
Old random rows cleared (an accidental 4x duplicate, 72 rows, was found and removed), loop run once:
18 rows. Same as before except su15: 40 / 39 / 39 actions (seeds 0/1/2). All 0 levels, 0 wins.

### Offline check (25 Oct)
All 25 games are downloaded in `environment_files` (19 training + 6 held-out). Saved notebook still had
`OperationMode.NORMAL`; switching to OFFLINE removes the ~2 s server call per game restart, removes the
internet dependency mid-run, and matches Kaggle (no internet during scoring). Not scope creep.

### Task 8: figures + Kaggle agent, Benny

**Log, as written:**
For Task 8, I had to build the data visualisations for the report and format the final Kaggle agent. I
had a big realisation while drafting MyAgent: the official Kaggle starter kit ignores all the Gymnasium
wrappers we built. It feeds raw game frames straight into choose_action. If I passed that raw frame into
model.predict(), the bot would crash because the image size and format wouldn't match training. The
action mask was missing too, because ArcGymEnv isn't there to provide it.
Fix: I made MyAgent self-contained. I copied the frame_to_observation code from Task 1 into the class as a
helper, and inside choose_action I rebuilt the 70-button mask from the latest frame's available actions
before feeding it to the model. This tricks the loaded model into thinking it's still inside our wrapper.

**Review (Claude): CHANGES REQUESTED**
1. **MUST DO BEFORE REAL RUNS:** `Monitor(make_environment())` has no filename, so episode logs live only
   in memory and vanish. No learning curves possible afterwards (brief requires them). Fix in train_dqn
   and train_ppo: `Monitor(make_environment(), filename=str(RESULTS_DIR / f"{version}_seed{seed}"))`.
2. Figure 3 (learning curves) missing entirely. Code given: reads the `*.monitor.csv` files, plots
   rolling-mean episode reward vs cumulative steps per seed; novelty versions titled "includes bonus".
3. Filter `selfcheck*` versions out of all figures.
4. Stats table std mixes games and seeds. Variability across SEEDS = mean per (version, seed) first,
   then std across seeds.
5. MyAgent copied the PRE-FIX adapter: click mapping % 64 / // 64 (top-row bug), no empty-frame guard,
   is_done compares exact strings (same bug as Task 1). Use the fixed versions.
6. DQN branch passes `action_masks` to `DQN.predict`, which doesn't accept it: crash.
7. Starter-kit format unverified (couldn't inspect the kit): constructor, return type of choose_action
   (likely a GameAction with set_data for clicks, not a (6, dict) tuple), and RESET-after-game-over
   behaviour must follow the kit's own template.
8. Kaggle scores offline: a trained agent needs sb3/sb3-contrib and the model zip as Kaggle datasets.
   Plan: submit the kit's random agent first (secures leaderboard eligibility), trained agent if time.
   Benny is doing the Kaggle submission himself on the website.
9. Log corrections: no longer an "exact copy" (it's the old version); not "tricking" the model, it's
   train/serve consistency, a standard requirement. AI use: not stated.

**Log, as written:**
My task was to implement the memory wrappers so the bot stops having the memory of a goldfish. I
realised I could reuse the Adaptive Stacking logic I studied in May for my research project on
time-series prediction. Porting it was mostly smooth; the big difference was that instead of a stack of
1D financial numbers, the memory slots now hold whole 2D game grids.
Another issue was how PPO and DQN read button presses differently once memory choices are added. PPO
can handle pressing two buttons at once (one game move, one memory slot), but standard DQN gets
confused unless it gets a single list of options.
Fix: the flatten_for_dqn flag. For PPO I used MultiDiscrete and concatenated the game's action mask with
k True values, so it picks a game move and a save slot independently. Since DQN only understands one
big list of buttons, I multiplied the action space (70 * k), stretched the mask with np.repeat, and used
divmod to decode the single choice back into a game move and a memory slot.

**Review (Claude), checked against the code:**
1. Code is correct for both wrappers: frame stack drops the oldest screen first; AS always writes the
   new screen somewhere (the choice is which old one to drop); divmod decoding and np.repeat mask line
   up (index = move * k + slot); PPO mask is 70 game values then k slot values, which is what
   MaskablePPO expects for MultiDiscrete.
2. **Upstream break:** for PPO with Adaptive Stacking, `model.predict` returns TWO numbers [move, slot],
   but `ppo_policy_function` does `int(action)`, which crashes on a 2-number array. And if it didn't,
   the wrapper's `step` expects action[0] and action[1]. So `ppo_v1/v2` with AS can train but can't be
   evaluated as-is. Needs a policy function that returns the [move, slot] pair for AS (Task 7 concern;
   evaluate's "returns an int" contract assumed one action).
3. "PPO presses two buttons at once": it makes two choices per step, one game move and one memory
   slot. Only the game move reaches the game.
4. "DQN gets confused": SB3's DQN only supports a single list of actions (Discrete); given MultiDiscrete
   it refuses with an error. Flattening is the standard workaround.
5. "Stretched the mask with np.repeat" as a fix for DQN: DQN ignores masks entirely, so the stretched
   mask isn't used by DQN itself. Harmless and consistent, but don't present it as making DQN legal.
6. Cost to state in the report: flattened AS gives DQN 70 x k actions (280 for k=4) to learn from the
   same sparse reward, so it's a harder problem than baseline DQN. That's a fair-comparison caveat.
7. Task 8: the Kaggle agent must rebuild the same memory stack at play time if the submitted model uses it.
8. AI use for this task: not stated in the log. Add a line either way.

**Log, as written:**
My task was to train a PPO baseline model. I came across a familiar problem: standard PPO ignores the
changing rules about which moves are legal, so it would waste time mashing locked buttons just like
the early DQN did. The same risk of losing all training to a crash was also still there.
Fix: I used MaskablePPO from sb3_contrib, which is built to read our action_mask during both
practice runs and the final evaluation, so it won't waste steps on dead buttons. I also brought over
the same CheckpointCallback from Task 3 so it saves progress every 10,000 steps.

**My understanding while implementing:**
`train_ppo` spawns the bot and feeds it a set of rules for how to update its brain, in things we call
batches, then makes it practise the game and saves its progress every so often, so a crash doesn't
delete its hard work. `ppo_policy_function` is the controller: during the final test it takes the
game screen and the list of currently unlocked buttons and presses the best legal move.
I used AI while implementing, learning as I went.

**Review notes (Claude):**
1. "I swapped out standard PPO for MaskablePPO" reads as a fix made during Task 4. It was a decision
   fixed in Part 1 before any code. Report it as a design choice, with the reason.
2. "Batches": PPO plays 512 steps first (`n_steps`), then learns from that pile in chunks of 64
   (`batch_size`), several passes over it. Collect, then learn, then repeat. That's different from
   DQN, which learns a little every 4 steps from its stored memory.
3. "Presses the best legal move": PPO doesn't score moves the way DQN does. It learns how likely it
   should be to press each button. `deterministic=True` presses the button it currently rates most
   likely, among the unlocked ones. "Most likely", not "best".
4. AI use: declared here, so the AI disclaimer needs updating (same as Task 3).

**What "controller" means here (game speak):** the trained model is the brain. The policy function is
the hands on the controller. Every turn, the test (`evaluate`) shows the hands the screen and which
buttons are lit up, and asks "what do you press?". The hands ask the brain and press one button.
Random, DQN and PPO all plug into the same controller slot (same inputs, same output), which is why
one `evaluate` can test all three fairly.
