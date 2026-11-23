# ARC-AGI-3: Team Workflow Guide

Who does what, in what order, and how to hand work in.
Read this once properly. Then use it as a checklist for every task.

---

## One-time setup: do this ONCE only (Elias and Paballo)

1. In Google Drive, click **Shared with me** and find `Assignment`.
   *This is the folder Benny shared.*
2. Right-click it, then **Organize > Add shortcut**, then choose **My Drive**.
   *Colab can only see folders in your My Drive, so the shortcut makes the shared folder show up there.*
3. **Don't rename the shortcut.**
   *The code looks for the exact name `Assignment`.*
4. Open the master notebook and click **File > Save a copy in Drive**. Work in your copy.
   *If two people edit the same notebook at once, one person's save can wipe the other's work.*

---

## Every time you open the notebook

1. Run Part 2. When Google asks for permissions, **tick all of them**.
   *If any box is unticked, Drive won't connect.*
2. Check that `Assignment/ARC_RL_Project/results` and `models` appear.
   *That shows you are saving into the shared folder.*

---

## The loop for every task

> **This is the part you repeat.** The setup above is done once. This loop is done again for EVERY task you own.

Every task, every time, goes through the same five steps:

1. **Build it** in your own copy of the notebook.
2. **Pass the test** for your task (usually the self-check printing `PASS`).
3. **Log it** as `#` comments in your task's NOTES cell.
4. **Paste it into the master** notebook (lock first, see below).
5. **Benny reviews it**: approved, or sent back with notes.

---

## Where to log

Right under your code and its self-check there is a **NOTES cell** for your task.
Write your log there as `#` comments. Benny writes the report from these, so be specific:
what you built, how it works in your own words, what broke, what you decided and why, and any AI help.

**Log as you go, not at the end.**

---

<h2 style="color:red; font-size:2em;">🔴 THE MASTER LOCK: READ THIS 🔴</h2>

<div style="color:red; font-size:1.3em; font-weight:bold;">

The MASTER is the one notebook inside the shared Assignment folder on Drive. It is NOT your own copy.

Only ONE person may have the master notebook open at a time.

1. Post <code>MASTER: mine</code> in the group chat. Wait until nobody else has it.
2. Open the master notebook in the Assignment folder.
3. Paste in your task's code cell and NOTES cell. Only yours.
4. Save, close the master tab, then post <code>MASTER: free</code>.

**Why:** Colab saves the whole notebook at once. If two people have it open and both save,
the later save wipes out the earlier person's work.

**Two people post "mine" at once?** The first message in the chat wins. The other person waits.

**Someone forgot to post "free"?** Message them. Do NOT open the master anyway.

**Your own copy is not locked.** Work in it whenever you like. The lock is only for the master.

</div>

---

## The tasks, in order

Each task waits on the ones before it. Tasks 2, 3 and 4 run at the same time because they're built
on the stand-in game (`FakeArcEnv`), so they don't wait for Task 1.

### 0. Shared setup (Part 2)
- **Owner:** Benny
- **Waits on:** nothing
- **Do:** run Part 2 top to bottom.
- **Test:** it prints `Actions per step: 70` and both game lists (training and held-out).
- **Log** as comments at the bottom of the settings cell:
  - how many games there are
  - which ones are held out
  - the package versions
- **Hand-off:** post "Setup done" in the chat.

### 1. Task 0: check what a real game returns
- **Owner:** Elias
- **Waits on:** Setup
- **Do:** run the cell. Nothing to write.
- **Test:** every field prints a value, and none of them say `<missing>`.
- **Log** as `#` comments in the Task 1 NOTES cell, under the heading `# Task 0`:
  - the real field names
  - anything that surprised you
  - anything that's different from the docs
- **Hand-off:** paste the whole output in the chat.

### 2. Task 1: ArcGymEnv, the game adapter
> Everyone waits on this one for real games.

- **Owner:** Elias
- **Waits on:** Task 0
- **Do:** fill the TODOs in the Task 1 cell.
- **Test:** the self-check prints **Task 1 PASS**.
- **Log** as `#` comments in the Task 1 NOTES cell:
  - how a game screen becomes the 64x64 observation, and why
  - how action numbers 0-69 map to real game actions
  - what happens with an invalid action
  - the difference between terminated and truncated
  - what broke and how you fixed it
  - any AI help you used
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 1 PASS".

### 3. Task 2: random agent and evaluation
- **Owner:** Benny
- **Waits on:** Setup
- **Do:** fill the TODOs, then write the protocol cell underneath.
- **Test:** the self-check prints **Task 2 PASS**.
- **Log** as `#` comments in the Task 2 NOTES cell:
  - the evaluation rules: which games, how many episodes, the action cap, the seeds
  - why each rule was chosen
  - what broke and how it was fixed
  - any AI help used
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 2 PASS".

### 4. Task 3: DQN baseline
- **Owner:** Paballo
- **Waits on:** Setup (real training also waits on Task 1)
- **Do:** fill the TODOs in the Task 3 cell.
- **Test:** the self-check prints **Task 3 PASS**.
- **Log** as `#` comments in the Task 3 NOTES cell:
  - in your own words: WHAT DQN learns, WHEN it learns, HOW that changes the actions it picks
  - any settings you changed, and why
  - what broke and how you fixed it
  - any AI help you used
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 3 PASS".

### 5. Task 4: PPO baseline with action masking
- **Owner:** Benny
- **Waits on:** Setup (real training also waits on Task 1)
- **Do:** fill the TODOs in the Task 4 cell.
- **Test:** the self-check prints **Task 4 PASS**.
- **Log** as `#` comments in the Task 4 NOTES cell:
  - WHAT PPO learns, WHEN, and HOW it changes decisions
  - what the clipped objective does
  - what masking stops the agent from doing
  - what broke
  - any AI help used
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 4 PASS".

---

> **Round 2 starts only after your Round 1 task is APPROVED.**

---

### 6. Task 5: novelty bonus
- **Owner:** Paballo
- **Waits on:** Task 3 approved
- **Do:** fill the TODOs in the Task 5 cell.
- **Test:** the self-check prints **Task 5 PASS**.
- **Log** as `#` comments in the Task 5 NOTES cell, answering these four:
  1. the problem we saw
  2. your guess for why the bonus helps
  3. what you built
  4. what the results showed

  Also log what broke, and any AI help.
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 5 PASS".

### 7. Task 6: memory (frame stacking, then Adaptive Stacking)
- **Owner:** Benny
- **Waits on:** Task 4 approved
- **Do:** fill in both wrapper classes.
- **Test:** the self-check prints **Task 6 PASS**.
- **Log** as `#` comments in the Task 6 NOTES cell: the same four (problem, guess, what was built,
  results), plus what broke and any AI help.
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 6 PASS".

### 8. Task 7: experiment runner
- **Owner:** Elias
- **Waits on:** Tasks 1-4 for the baselines, Tasks 5-6 for the improved versions
- **Do:** write `run_version` and fill in `VERSIONS`.
- **Test** (there's no self-check cell, so do both):
  - run `dqn_v0_baseline` with 1 seed on a tiny budget, and check that new rows appear in your results file
  - run it again, and check that it SKIPS that seed instead of retraining
- **Log** as `#` comments in the Task 7 NOTES cell:
  - run time per version
  - hardware (Colab CPU or GPU type)
  - how resuming works
  - what broke
  - any AI help
- **Hand-off:** paste into the master (🔴 lock first), then post "Task 7 works" with a screenshot of the rows.

### 9. Task 8: figures, tables, Kaggle agent
- **Owner:** Benny
- **Waits on:** nothing for the random-agent Kaggle submission (do it by 20 Oct); Task 7 results for the figures
- **Do:** submit the random agent to Kaggle, then build the figures, then swap in the trained agent.
- **Test:**
  - Kaggle shows a score for our submission
  - the figures build from the results files
- **Log** as `#` comments in the Task 8 NOTES cell:
  - the submission ID, score and timestamp
  - which figure shows what
  - what broke
- **Hand-off:** paste into the master (🔴 lock first), then post the Kaggle score in the chat.

---

## Rules

- Work in your **own copy** of the notebook.
- Only edit **your own section**.
- **Never rename functions**, or change their inputs or outputs. Other people's code calls them by name.
- **Log as you go**, not at the end. The report gets written from our logs.
- 🔴 **Only touch the master while you hold the lock.**
