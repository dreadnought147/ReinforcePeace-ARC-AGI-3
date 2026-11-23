# Memory and Count-Based Exploration for ARC-AGI-3: DQN versus Maskable PPO

*Making peace with sparse rewards*

**Team ReinforcePeace** (Banele Mkhize, Elias Sepuru, Paballo), University of the Witwatersrand.
Linked code submission: 57654331 (trained Maskable PPO). Full report attached.

## Problem

ARC-AGI-3 drops an agent into unfamiliar grid games with no instructions and no reward other than a
counter of completed levels. Until the first level is completed by chance, every reward is zero, and
the screen does not always show the full state. We asked whether two standard deep reinforcement
learning methods can learn in this setting, and whether memory and count-based exploration help.

## Approach

**Interface.** We wrapped the official toolkit in a Gymnasium environment. Observation: the latest
grid, padded to 64x64. Actions: 70 discrete actions (ACTION1-5, ACTION7, and ACTION6 clicks at the
centres of an 8x8 grid of blocks), with a mask of the actions each game currently offers. Reward: the
change in levels completed. Episodes end at WIN or GAME_OVER, or are truncated at 500 steps.

**Algorithms.** DQN (Stable-Baselines3), which cannot use the mask, and Maskable PPO (sb3-contrib),
which gives unavailable actions zero probability. Both use the standard Nature CNN.

**Improvements.** (1) Memory: the last 4 screens are stacked. (2) Count-based novelty: each screen is
hashed and the training reward gets a bonus of 0.1 / sqrt(times seen). An ablation applies novelty
without memory. All versions use 50,000 training steps per seed, 3 seeds, and are evaluated on 6
held-out public games never used in training, against a random policy.

**Submitted agent.** Maskable PPO with memory (seed 2, chosen by the most levels completed during
training). It reproduces the training preprocessing and mask, resets after game over, and samples from
the masked policy so that retries differ.

## Results

- No agent, learned or random, completed a level on the held-out games at this budget, so neither
  improvement produced a measurable gain.
- During training, Maskable PPO completed about twice as many levels as DQN (6-17 versus 3-10 per seed)
  and never pressed an unavailable action.
- DQN with memory or novelty sometimes locked onto an action the game does not offer and pressed it for
  all 500 steps: in 1, 3 and 6 seed-game pairs for memory, memory + novelty and novelty only. The
  ablation attributes this mainly to the novelty bonus.
- Several held-out games end each attempt after a fixed action budget whatever is pressed, so undirected
  exploration is cut off quickly.
- Kaggle public score: random starter agent 0.07; our trained agent 0.00. The random agent retries with
  clicks anywhere on the 64x64 grid, while ours clicks only 64 block centres and samples from a policy
  that learned little, so it explored less widely than random.

## Why our trained agent scored 0.00

Making peace with sparse rewards starts with admitting how little reward our agent ever saw. We think
the 0.00 has five causes, and each is backed by our own numbers or by other work on ARC-AGI-3.

1. **Too little reward to learn from.** In 50,000 training steps across 19 games, the submitted model
   completed 19 levels, the most of any run; most runs completed 3 to 10. That is about one success per
   game, far too few examples for a network to learn what completing a level looks like. Others hit the
   same wall at much larger scale: DreamerV3 received zero reward on 21 of 25 games over 500,000-step
   runs, ten times our budget (Ahmad et al., 2026).
2. **Every hidden game has new rules.** Kaggle scores on games the agent has never seen. Our agent learns
   reflexes for its training games ("on screens like this, press that"), and those do not carry over to
   new mechanics. Cross-game pretraining has been reported to give no consistent benefit (Ahmad et al., 2026).
   Locally, none of our eight versions completed a level on our six held-out games either.
3. **No learning while playing.** The weights are frozen at test time, and four remembered screens are
   too few to work out a game's rules. A non-learning graph explorer that simply records which states and
   actions it has already tried solved a median of 30 of 52 preview levels and ranked third on the private
   leaderboard (Rudakov et al., 2025). Within-game exploration matters more than learned reflexes here.
4. **Less coverage than random.** The random starter agent clicks anywhere on the 64x64 grid (4,096
   targets) and retries after every game over. Ours can click only 64 block centres, and it samples from a
   trained policy that likely favours a few actions without having learned which ones win, so its retries
   probably cover fewer distinct actions (we did not measure this directly). On games where the answer is a click outside our 64 centres, it cannot succeed at all. The same
   NeurIPS study found that changing the action interface alone made reward discovery possible on one game,
   so this design choice may matter as much as the algorithm.
5. **The score punishes slow wins.** A level scores (human actions / agent actions) squared. Even a lucky
   completion in ten times the human's actions is worth about 1%, so an agent that wins rarely and
   slowly rounds to 0.00.

The honest reading: reward-driven learning cannot start until reward arrives, and on ARC-AGI-3 it almost
never does. Making peace with sparse rewards means stopping waiting for them. The next step is an agent
that explores each game systematically and keeps track of what it has already tried, with learning
layered on top once there is something to learn from.

## What we learned

Invalid-action masking mattered more than either improvement. The likely reason the improvements did
not help is budget: a handful of completed levels across 19 games gives almost no reward per game to
learn from. Our main methodological lesson: under reward shaping, log the true reward separately. Our
training logs recorded reward including the novelty bonus, so we could not tell whether novelty led to
more completed levels during training.

## References

- Ahmad, H., Sun, Y., Zahin, A., Mehra, V., Banerjee, S. (2026). Competence without Performance:
  Representation Without Control on ARC-AGI-3. NeurIPS 2026 Workshop.
- Rudakov, E., Shock, J., Cowley, B. U. (2025). Graph-Based Exploration for ARC-AGI-3 Interactive
  Reasoning Tasks. arXiv:2512.24156.
- Nangue Tasse, G., Riemer, M., Rosman, B., Klinger, T. Beyond Sliding Windows: Learning to Manage Memory
  in Non-Markovian Environments.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Huang, S., Ontañón, S. (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.
