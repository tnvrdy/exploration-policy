"""
Goal-directed exploration agent.

Runs short episodes (default 4 steps) against a live browser, using Qwen to pick actions
from a constrained action vocabulary. Every step captures rich observations (AX tree,
HTML, viewport screenshot) and writes them to a trajectory directory.

Usage:
    from agent_goaldirected import run_exploration_episode
    traj_dir = run_exploration_episode("https://en.wikipedia.org", "Find the article on black holes")

    # or from cli:
    python agent_goaldirected.py https://www.stanford.edu "Find the page for the next poetry reading event" 4
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_core import run_episode_steps
from browser_env import BrowserEnv
from trajectory_store import TrajectoryWriter


def run_exploration_episode(
    url: str,
    goal: str,
    trajectories_dir: str | Path = "trajectories",
    model: str | None = None,
    max_steps: int = 4,
    headless: bool = True,
) -> Path:
    """
    Run one short exploration episode and persist the full trajectory.

    Returns the path to the trajectory directory
    (contains metadata.json, steps.jsonl, screenshots/).
    """
    with (
        BrowserEnv(headless=headless) as env,
        TrajectoryWriter(trajectories_dir, goal=goal, start_url=url) as tw,
    ):
        env.goto(url)
        print(f"[episode] goal={goal!r}  url={url}")

        reason = run_episode_steps(env, tw, goal, model=model, max_steps=max_steps)
        tw.set_termination_reason(reason)
        print(f"[episode] done ({reason})")

        return tw.traj_dir


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python agent.py <url> <goal> [max_steps]")
        sys.exit(1)

    _url = sys.argv[1]
    _goal = sys.argv[2]
    _max = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    traj = run_exploration_episode(_url, _goal, max_steps=_max, headless=False)
    print(f"\nTrajectory saved: {traj}")
