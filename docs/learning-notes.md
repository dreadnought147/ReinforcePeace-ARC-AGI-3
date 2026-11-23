# ARC-AGI-3: Learning Notes

Things to go back over once the build is done. Each note is self-contained: it starts with where we
were in the journey, then teaches the idea as simply as possible. Familiar RL ideas are kept short;
new ones get more room. Notes are added as we work, in the order they came up.

---

## The journey so far, in one paragraph

We're building a robot that plays games nobody explained to it. First we played a toy game ourselves
(Part 0) to feel the problem. Then we built the plug between the real ARC games and our learning
library (Task 1, the adapter), a fair way to test any robot (Task 2), and two robots: DQN (Task 3) and
PPO (Task 4). Now we're giving them help: a reason to explore (Task 5, the novelty bonus) and a memory
(Task 6). Then we run everything (Task 7) and turn the numbers into figures and a Kaggle entry (Task 8).

---

## 1. Observation vs reward (Part 0.4)

**Where we were:** playing the mystery game, you said "everything in between was up in the air".

It wasn't. After every button press you saw the board change. That's the **observation**: what the
robot sees after each move. The **reward** is the score signal, and in our games it only arrives when
a level is completed. You used the observations constantly (you saw which block moved). The robot gets
the same split: a picture every step, a point almost never.

**Remember:** observation = every step, information. Reward = rare, the only thing that says "good".

---

## 2. A policy, not a list of moves (Part 0.4)

**Where we were:** you described the robot as "figuring out what the moves array should be".

A robot doesn't learn a fixed list. It learns a **policy**: a rule that looks at the current screen and
picks one action, then looks again and picks the next. If the game started differently, a fixed list
would fail; a policy adapts. That's also why we test on games it never trained on: a memorised list
would be useless there.

---

## 3. Episodes, and terminated vs truncated (Part 0.4, Task 1)

An **episode** is one go at a game: reset, play, end. It can end two different ways:
- **terminated:** the game itself ended it (won, or game over). Nothing comes after.
- **truncated:** we stopped it because it hit our 500-step cap. The game could have carried on.

**Why keep them separate:** learning works by estimating "how good is the future from here". If a
cut-off from our cap were treated as a real ending, the robot would learn "the future here is worth
nothing", which is false: we just ran out of time. Mixing them up teaches the robot wrong values.

---

## 4. Hidden information (Part 0.7, and the trick question)

**Where we were:** I asked what hidden information there was in the mystery game. There wasn't any,
because the key disappearing shows on screen that you hold it.

Some games hide things: a light flashed at the start, a switch was pressed off-screen. Two moments can
then look identical but need different actions. A robot that only sees the current screen can't tell
them apart, however long it trains. The fix is **memory** (Task 6). The technical name is a
**partially observable** game: the screen is only part of the truth.

---

## 5. The adapter IS the MDP (Task 1)

**Where we were:** Elias built `ArcGymEnv`, the plug between the ARC game and the learning library.

It's the normal adapter pattern from software engineering, but here every translation choice decides
what problem the robot is solving:
- **State:** the 64x64 picture we hand it.
- **Actions:** the 70 buttons we offer.
- **Reward:** +1 per level, nothing else.
- **How the world responds:** the ARC engine, a black box we don't see.
- **Discount:** not in the adapter; it's a setting inside DQN/PPO (0.99 here).

**Why 70 actions:** clicking can land on any of 4,096 cells. That's far too many to explore, so we offer
6 simple buttons plus clicks on the centres of an 8x8 grid (64 spots). The cost: a game that needs a
precise click between grid centres becomes impossible. That's a design decision to state in the report.

---

## 6. Action masking (Tasks 1, 3, 4)

**Where we were:** DQN kept pressing locked buttons; PPO didn't.

Each game only offers some buttons at a time. The **mask** is a list of 70 yes/no values: which buttons
work right now.
- **DQN can't use it.** It scores every button and presses the highest score, locked or not. Our adapter
  turns a locked press into "nothing happens" and counts it in `invalid_actions`. Those presses still
  use up steps, which is wasted practice.
- **MaskablePPO uses it.** PPO turns its preferences into probabilities. Masking sets the probability of
  every locked button to zero before it picks, both while practising and while being tested. It can
  never waste a press.

We tried adding a mask to DQN at test time only, then removed it: it trained without the mask but was
tested with it, which is a mismatch, and it would have hidden the evidence for "DQN can't mask".

---

## 7. Fair testing: the evaluation protocol (Task 2)

**Where we were:** you wrote the rules for how every robot gets tested.

- **Held-out games:** every 4th game is never used for practice, so a good score there means it learned
  something general, not those particular games.
- **3 seeds:** a seed fixes the randomness. Running 3 and comparing shows whether a result is real or luck.
- **Deterministic testing:** a trained robot always takes its top choice when tested, no random presses.
- **1 episode per game:** with a fixed seed and no randomness, a second episode would be an exact repeat.

**Correction:** the official score is worked out **per level**. Our `official_score` column only scores
whole-game wins, which will be rare, so it'll be mostly empty. Either track actions per level, or say so.

---

## 8. Three different kinds of number (Tasks 2 and 5)

These keep getting mixed up, so keep them in three boxes:
1. **Official ARC score:** the competition's formula, (human actions / robot actions) squared, per
   level. Only this counts as "the score".
2. **Our measurements:** levels completed, actions used, invalid actions, time. Real facts, not rewards.
3. **Training reward:** what the robot learns from. +1 per level is the game's own signal. The novelty
   bonus is the only thing we invented, and it's used for training only. It's never reported as a score.

Kaggle never sees any of our rewards; it runs its own scoring. The rule about not reporting shaped
reward comes from the assignment brief.

---

## 9. DQN, and what its settings do (Task 3)

You know DQN from Lab 3. What it learns: a score for every button on every screen ("Q-values"). When:
a little every 4 steps, from a memory of past moves. How it acts: presses the highest-scoring button,
plus random presses while exploring.

Our settings, in plain words:
- `buffer_size=20_000`: remembers the last 20,000 moves to learn from.
- `learning_starts=1_000`: just collects moves for the first 1,000 steps before learning anything.
- `exploration 1.0 -> 0.05 over 30%`: starts fully random, becomes mostly greedy by 30% of training,
  then keeps 5% random presses.
- `target_update_interval=1_000`: DQN learns towards a frozen copy of itself, refreshed every 1,000
  steps, so it isn't chasing a target that moves every step.
- `train_freq=4`: one learning update every 4 moves.

**The failure we saw:** after the smoke run it pressed button 10 every single time (**collapse**). With
only ~250 updates its scores are noise, and it just obeys the biggest one. Watch for this in real runs.

---

## 10. PPO, from the ground up (Task 4)

**Where we were:** you used AI to help implement it, and asked what a "controller" is.

PPO doesn't score buttons. It learns **how likely it should be to press each button** on each screen
(the policy), plus a separate guess of how good the current situation is (the value).

**How it learns:** play 512 steps (`n_steps`), then stop and learn from that pile in chunks of 64
(`batch_size`), several passes, then throw the pile away and play again. Collect, learn, repeat.
That's **on-policy**: it only learns from its own most recent play. DQN is **off-policy**: it reuses old
memories.

**The clipped objective (the "P" in PPO):** after each pile, PPO makes button presses that worked more
likely and ones that didn't less likely, but it refuses to change any probability by more than about
20% in one round. Big jumps can wreck a policy that was working; small careful steps are safer.

**Entropy bonus (`ent_coef=0.01`):** a small reward for staying a bit unsure, so it keeps trying
different buttons instead of locking onto one too early. Useful when reward is this rare.

**Deterministic = "most likely", not "best":** at test time it presses the button it rates most likely.
For an untrained robot that's an arbitrary button, pressed every step. That's why it scored below
random in the smoke test: random at least varies its presses.

**The controller:** the trained model is the brain; the policy function is the hands on the
controller. Each turn the test shows the hands the screen and which buttons are lit, and asks "what
do you press?" Random, DQN and PPO all plug into the same slot, so one test treats all three fairly.

---

## 11. Sparse reward and the novelty bonus (Task 5)

**Where we were:** the robot gets a point only per level, so for a long time it learns nothing. You
built the fix, and asked whether the robot even sees the same screen more than once.

**Yes, constantly.** Pressing a button that does nothing gives the same screen again. Walking left then
right brings you back. Every episode starts on the same opening screen. That's exactly why we count.

**What exactly is counted:** whole screens. Each 64x64 picture is turned into a short fingerprint
(an md5 **hash**: the same picture always gives the same fingerprint, a different picture almost
always gives a different one). A dictionary maps fingerprint to "times seen".

**How it decides how boring a screen is:** it looks the fingerprint up and pays `0.1 / sqrt(count)`.
First visit 0.100, second 0.071, fourth 0.050, hundredth 0.010. It fades but never hits zero. The square
root makes it fade gently, so a screen stays a little interesting for a while.

**This is the course's optimistic-values idea, scaled up.** In Part 0 a table assumed untried actions
were great. Here we can't keep a table of every screen's value, so we pay directly for newness.

**Three traps to check on real games:**
1. **It counts screens, not situations.** Hidden information (a held key) doesn't change the
   fingerprint, so two different situations count as one.
2. **Anything that changes every step** (a timer, a move counter on screen) makes every screen "new"
   forever, so the bonus never fades and the robot is paid for doing nothing useful.
3. **The bonus can drown the real goal.** In the self-check, 200 random steps earned 7.89 in bonus vs 3 in
   real reward. If chasing new screens pays more than finishing levels, the robot learns to chase
   screens. `bonus_scale` is the dial.

---

## 13. Memory: frame stacking vs Adaptive Stacking (Task 6)

**Where we were:** you ported Adaptive Stacking from your research project so the robot remembers more
than the current screen.

- **Frame stacking:** the robot sees the last k screens (k=4 here), like a short video clip. Every step
  the oldest drops out. Simple, but it only remembers the recent past: a light that flashed 50 steps
  ago is long gone.
- **Adaptive Stacking:** the robot still has k slots, but it **chooses which old screen to throw away**
  each step. The new screen always goes in. So it can keep the "light flashed LEFT" screen for as long
  as it's useful. It learns what to remember, which is your research project's title, just with whole
  pictures in the slots instead of single numbers.
- **The price:** choosing a slot is an extra decision every step, learned from the same rare reward.
  More to learn means it may learn slower.
- One thing the network can't tell: which slot is newest. Slots aren't in time order in Adaptive Stacking.

---

## 14. Two choices per step: MultiDiscrete vs flattening (Task 6)

With Adaptive Stacking each step needs two choices: a game move (70 options) and a memory slot (k).

- **PPO, MultiDiscrete:** PPO picks each part separately, two dials. Its mask is the 70 game values
  followed by k values for the slots (all allowed).
- **DQN, flattened:** SB3's DQN only accepts one dial. So every (move, slot) pair becomes one option:
  70 x 4 = 280 options. Option number n decodes with `divmod(n, k)`: n // k is the move, n % k is the
  slot. Example with k=4: option 41 gives move 10, slot 1.
- **Cost:** DQN now has 280 scores to learn per screen instead of 70, from the same rare reward.
- **Gotcha:** for PPO the "action" is now two numbers, so anything that assumed one number (like
  `int(action)` in the policy function) breaks.

---

## 15. Change one thing at a time (Task 7)

**Where we were:** the first VERSIONS draft went from frame stacking (v1) to Adaptive Stacking plus
novelty (v2).

If two things change between versions and the score moves, you can't say which one caused it. That's
called **confounding**. Each step in the chain must add exactly one thing:
- v0 baseline, then v1 = v0 + memory, then v2 = v1 + novelty.
- The **ablation** (novelty without memory) is the other half of the puzzle: comparing v2, v1 and
  novelty-only tells you what each part contributed on its own, and whether they help each other.
This is most of what the 25 "experimental rigour" marks are about.

---

## 16. Wrapper order matters (Task 7)

**Where we were:** the novelty bonus was wrapped around the frame stack.

Wrappers are layers, like an onion. The outer layer sees whatever the inner layers hand it.
- `NoveltyBonus(FrameStackImage(game))`: novelty sees the stack of 4 screens and fingerprints the
  whole stack. The same screen reached by a different route gives a different stack, so it counts as
  "new". Almost everything looks new, and the bonus barely fades.
- `FrameStackImage(NoveltyBonus(game))`: novelty sees single screens, which is what we mean by "a place
  the robot hasn't been".
Same pieces, different order, different experiment. The order goes in the report.

**Novelty, recap in one line:** the robot gets a small tip (0.1 / square root of times seen) for
reaching a screen, so new screens pay and familiar ones barely do. See note 11.

---

## 17. PASS is not "correct" (Task 1 bugs found in Task 7)

**Where we were:** the first real run crashed, and the cause was in Task 1, which had passed.

A self-check only proves the things it checks. Task 1's checked shapes and info keys, and that an
episode ended within 50 steps. Three real bugs slipped through:
- **Comparing the wrong text:** the game reports its state as "GameState.GAME_OVER". Checking whether
  that equals "GAME_OVER" is always False. Checking whether "GAME_OVER" appears *inside* it works.
  Task 0 had printed the real format; the lesson is to build from what the probe actually showed.
- **A wrong formula that still runs:** clicks were mapped with 64 instead of 8, so every click hit the
  top row. Nothing crashes, the bot just can't click most of the board. Silent bugs like this are the
  dangerous kind, because the results look like "the bot is bad" rather than "the code is wrong".
- **Unexpected empty data:** the game sometimes returns no picture. Code that assumes there's always
  one crashes.
The habit: when a test passes, ask "what would still be wrong even with this passing?"

---

## 18. Reading the first real result (random on held-out games)

**Where we were:** random played all 6 held-out games with 3 seeds and completed nothing.

- **Episodes ended by themselves** (75, 100, 151 moves...) instead of always hitting our 500 cap. So the
  games have their own endings, most likely a limited number of moves. That's also proof the
  game-over fix works.
- **Same length on every seed** for 4 games means the ending doesn't depend on what you press: you get
  a fixed number of moves, like a timer. Efficiency isn't just about the score, it's survival: waste
  moves and the level ends.
- **No punishment for running out:** the game just stops with 0 reward. To the robot, "ran out of moves"
  and "nothing happened yet" look identical. It only learns from the levels it would have won.
- **A seed must actually control the randomness.** Writing "seed 1" in the results means nothing unless
  the code that makes random choices was set with that seed. Otherwise rerunning won't reproduce it.

---

## 19. Where the time actually goes (Task 7 smoke test)

**Where we were:** a 2,000-step smoke test took 2 minutes, though training that size took 9 seconds on
the fake game.

The robot's learning wasn't slow. Every time a game restarts, our setup asked the ARC server online
for the game's details, about 2 seconds each. 40 restarts = 80 seconds of waiting on the internet.
This is the most common real-world surprise: the slow part is often loading and waiting, not the
clever maths. The fix is to download every game once and play them **offline** from disk, which is
also how Kaggle runs (no internet during scoring).

---

## 20. DQN pressing a dead button 500 times

**Where we were:** in the smoke test, on two games every single press was invalid.

The barely-trained DQN had collapsed onto one button (note 9). On games that don't offer that button,
every press is ignored: the screen never changes, the game never ends, and it burns all 500 moves.
PPO with masking can't do this, because locked buttons have zero chance of being picked. This is the
"DQN can't mask" cost made visible, and exactly the kind of failure case the report should show.

---

## 21. Same preparation in training and on Kaggle (Task 8)

**Where we were:** building the Kaggle agent, you realised Kaggle hands the robot raw screens, with none
of our wrappers.

The robot only knows screens prepared the way it saw them in training: padded to 64x64, multiplied by
16, one channel. On Kaggle, our agent has to do that exact preparation itself, and rebuild the button
mask itself. This is called **train/serve consistency**: whatever you did to the data in training, you
must do identically at play time. It's not tricking the model, it's speaking its language.

The trap we fell into: the copy was taken from the OLD adapter, before the bug fixes. Copies drift.
Whenever you copy logic, copy from the current, fixed version, or better, have both places call one
shared function.

---

## 22. Logs that disappear (Task 8)

**Where we were:** writing the learning-curve figure, which needs every episode's reward during training.

Monitor records each training episode's reward and length, but without a filename it keeps them only in
memory, and they vanish when training ends. Learning curves are required by the brief, so the file has to
be switched on BEFORE the real runs. You can't recover data you never saved.

---

## 23. Reading a negative result (the final runs)

**Where we were:** 24 runs finished and no robot completed a single held-out level.

- **"Nothing worked" is still a result.** The marks are for the investigation, not the score. The
  question becomes: what *did* differ, and why didn't the changes help?
- **What differed:** PPO with masking never wasted a press and finished about twice as many levels in
  training. DQN, once given memory or novelty, sometimes locked onto a button the game doesn't offer
  and pressed it for all 500 steps. The ablation (novelty without memory) showed novelty is the main
  cause of that.
- **Why the changes likely didn't help:** budget. 3 to 19 levels over 19 games in 50,000 steps means
  almost no reward per game, so there's little to learn from, with or without memory.
- **The logging lesson:** our training log recorded reward WITH the novelty bonus, so we can't tell if
  novelty made the robot finish more levels while training. Always log the true reward separately from
  any shaped reward. You only find out you needed a number after the run is over.

---

## 24. We're not alone: what others found on ARC-AGI-3

**Where we were:** checking whether the zero held-out score was our mistake or a known wall.

It's a known wall. Other groups report the same things we saw:
- **Sparse reward stops learning:** level completion is the only signal, which "severely constrains
  learning-based methods" (Graph-Based Exploration paper, arXiv 2512.24156). DreamerV3, a much stronger
  RL agent, got zero reward on 21 of 25 games over 500,000 steps, 10 times our budget.
- **No transfer between games:** pretraining across games gave "no consistent benefit" (NeurIPS 2026
  "Competence without Performance"), because every game has new mechanics.
- **What works instead:** a graph explorer with NO learning at all, which just remembers which states
  and actions it has tried, ranked 3rd on the private leaderboard. Systematic exploration beats learned
  reflexes here, which is what our novelty bonus was reaching for.
- **Even frontier LLMs** scored under 1% at launch; humans solve 100%.

---

## 12. Smaller ideas worth knowing

- **Smoke test vs real run:** a self-check proves the code runs. It doesn't prove the robot learns.
  2,000 steps is far too few; real runs are 50,000 per seed.
- **Checkpoints vs resuming:** saving every 10,000 steps means a crash doesn't lose everything, but only
  if something loads the latest save and carries on. Nothing does that yet (Task 7's job).
- **Ablation:** take the improved robot and remove one part to see how much that part alone mattered.
  Ours: novelty without memory (`v2_novelty_only`).
