"""Our trained ARC-AGI-3 agent (MaskablePPO), ready to replace the random `my_agent.py`.

How it gets onto Kaggle:
  1. Kaggle scores offline, and the starter kit only installs `arc-agi`. So our extra libraries
     (stable-baselines3, sb3-contrib, gymnasium) and the trained model file travel in our own Kaggle
     dataset, `arc-rl-agent-files`, laid out as:
         pylibs/  -> the library folders, added to sys.path below
         model/agent_model.zip  -> the MaskablePPO model chosen from our experiments
  2. When it's time to submit, this file is copied over `agent/my_agent.py`, and the dataset is listed
     in `notebooks/kernel-metadata.json` under "dataset_sources".

Everything the model sees is prepared EXACTLY as our training adapter (ArcGymEnv, Task 1) prepared it:
last grid of the frame, padded to 64x64, multiplied by 16, one channel; the same 70-action layout
(6 simple actions + clicks on the centres of an 8x8 grid); the same action mask. If the submitted model
was trained with frame stacking, STACK_FRAMES must match (4), otherwise 1.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`, class named `MyAgent`.
  - `is_done(frames, latest_frame) -> bool` and `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

# ---------------------------------------------------------------------------------------------
# Settings that must match the model we submit
# ---------------------------------------------------------------------------------------------
STACK_FRAMES = 4                         # submitted model: ppo_v1_memory seed 2 (frame stacking, k=4); 1 for *_v0_baseline
# Sample from the policy instead of always taking its top action. On Kaggle the agent RESETs and retries
# after game over; a deterministic policy would replay the identical failed attempt every time, while
# sampling (still restricted to legal actions by the mask) varies each retry. Local evaluation used
# deterministic=True; this difference is stated in the report.
DETERMINISTIC = False
GRID_SIZE = 64                           # every screen is padded to 64x64, as in training
COARSE_GRID = 8                          # clicks go to the centres of an 8x8 grid of 8x8-cell blocks
SIMPLE_ACTION_IDS = [1, 2, 3, 4, 5, 7]   # our action indices 0..5 -> ACTION1..ACTION5, ACTION7
N_ACTIONS = len(SIMPLE_ACTION_IDS) + COARSE_GRID * COARSE_GRID   # = 70
DATASET_NAME = "arc-rl-agent-files"


def find_dataset_directory() -> str | None:
    """Locate our dataset under /kaggle/input. Kaggle's mount layout has changed over time
    (/kaggle/input/<name> vs /kaggle/input/datasets/<owner>/<name>), so search a few levels deep
    instead of hard-coding one path. ARC_RL_DATASET overrides this for local testing."""
    if os.environ.get("ARC_RL_DATASET"):
        return os.environ["ARC_RL_DATASET"]
    for root, folders, _files in os.walk("/kaggle/input"):
        if os.path.basename(root) == DATASET_NAME and "pylibs" in folders:
            return root
        if root.count(os.sep) > 5:          # don't crawl the whole competition data tree
            folders[:] = []
    return None


DATASET_DIRECTORY = find_dataset_directory()
if DATASET_DIRECTORY and os.path.isdir(os.path.join(DATASET_DIRECTORY, "pylibs")):
    sys.path.insert(0, os.path.join(DATASET_DIRECTORY, "pylibs"))   # makes sb3 / sb3_contrib importable offline

_LOADED_MODEL = None   # loaded once and shared by every game's agent instance


def action_number(action) -> int:
    """The action's number (ACTION3 -> 3), whether the game lists it as an int or as a GameAction.
    GameAction's enum value is not a plain int, so read the number from the member's name instead."""
    if isinstance(action, GameAction):
        return int(action.name.replace("ACTION", "")) if action.name.startswith("ACTION") else -1
    return int(action)


def load_model():
    """Load the MaskablePPO model once. custom_objects replaces the training-only schedules, which
    can fail to unpickle on a different Python version and are not needed to choose actions."""
    global _LOADED_MODEL
    if _LOADED_MODEL is None:
        from sb3_contrib import MaskablePPO
        model_path = os.path.join(DATASET_DIRECTORY or ".", "model", "agent_model.zip")
        unpacked_model = os.path.join(DATASET_DIRECTORY or ".", "model", "agent_model")
        if not os.path.isfile(model_path) and os.path.isdir(unpacked_model):
            # Kaggle auto-extracts uploaded .zip files, so the model arrives as a folder of its parts
            # (data, policy.pth, ...). SB3 loads from a zip, so pack the folder back into one in /tmp.
            import shutil, tempfile
            model_path = shutil.make_archive(os.path.join(tempfile.gettempdir(), "agent_model"), "zip", unpacked_model)
        _LOADED_MODEL = MaskablePPO.load(
            model_path,
            device="cpu",
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
        )
    return _LOADED_MODEL


class MyAgent(Agent):
    """Plays with our trained MaskablePPO policy: one screen in, one legal action out."""

    MAX_ACTIONS = 80   # starter-kit default cap per game; the framework also enforces global limits

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = load_model()
        self.recent_screens: deque = deque(maxlen=STACK_FRAMES)   # memory, if the model uses frame stacking
        self.last_screen = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop only on a win. On GAME_OVER we RESET and try again (choose_action handles it).
        return latest_frame.state is GameState.WIN

    # ------------------------------------------------------------------ same preparation as training

    def screen_from_frame(self, frame: FrameData) -> np.ndarray:
        """Last grid of the frame -> padded 64x64 -> x16 -> uint8. Mirrors ArcGymEnv.frame_to_observation,
        including the fallback when a frame arrives with no grid at all."""
        if not frame.frame:
            return self.last_screen
        grid = np.array(frame.frame[-1], dtype=np.int32)
        height, width = grid.shape
        grid = np.pad(grid, ((0, max(0, GRID_SIZE - height)), (0, max(0, GRID_SIZE - width))))
        self.last_screen = (grid * 16).astype(np.uint8)
        return self.last_screen

    def action_mask(self, frame: FrameData) -> np.ndarray:
        """70 booleans, True where this game currently offers the action. Mirrors ArcGymEnv.action_masks."""
        offered = {action_number(action) for action in (frame.available_actions or [])}
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for index, action_id in enumerate(SIMPLE_ACTION_IDS):
            mask[index] = action_id in offered
        if 6 in offered:
            mask[len(SIMPLE_ACTION_IDS):] = True
        return mask

    # ------------------------------------------------------------------ the decision

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # First call, or after losing: restart the level, and forget the old screens.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.recent_screens.clear()
            return GameAction.RESET

        screen = self.screen_from_frame(latest_frame)
        if not self.recent_screens:                    # a fresh start fills the memory with the first screen,
            self.recent_screens.extend([screen] * STACK_FRAMES)   # exactly like FrameStackImage.reset
        else:
            self.recent_screens.append(screen)
        observation = np.stack(self.recent_screens, axis=0)       # shape (STACK_FRAMES, 64, 64)

        mask = self.action_mask(latest_frame)
        if not mask.any():                             # nothing playable is offered: restart rather than guess
            return GameAction.RESET

        action_index, _ = self.model.predict(observation, action_masks=mask, deterministic=DETERMINISTIC)
        action_index = int(action_index)

        if action_index < len(SIMPLE_ACTION_IDS):
            action = GameAction[f"ACTION{SIMPLE_ACTION_IDS[action_index]}"]   # look up by name: the enum's value isn't a plain int
            action.reasoning = f"policy picked simple action {action.name}"
        else:
            row, column = divmod(action_index - len(SIMPLE_ACTION_IDS), COARSE_GRID)
            action = GameAction.ACTION6
            action.set_data({"x": COARSE_GRID * column + 4, "y": COARSE_GRID * row + 4})   # centre of that block
            action.reasoning = {"why": f"policy picked click block row {row}, column {column}"}
        return action
