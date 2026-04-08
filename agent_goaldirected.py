"""
Goal-directed exploration agent.

Runs episodes against a live browser, using Qwen to pick actions from a
constrained action vocabulary. Supports both single-episode and batching.

Usage:
    python agent_goaldirected.py https://www.stanford.edu "Find the next poetry reading event" 6
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_core import run_steps
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
    """Run one episode with its own browser. Fine for testing, not for scale."""
    with (
        BrowserEnv(headless=headless) as env,
        TrajectoryWriter(trajectories_dir, goal=goal, start_url=url) as tw,
    ):
        env.goto(url)
        print(f"[episode] goal={goal!r}  url={url}")
        reason = run_steps(env, tw, goal=goal, model=model, max_steps=max_steps)
        tw.set_termination_reason(reason)
        print(f"[episode] done ({reason})")
        return tw.traj_dir


def run_task_batch(
    tasks: list[dict],
    trajectories_dir: str | Path = "trajectories",
    model: str | None = None,
    max_steps: int = 4,
    headless: bool = True,
) -> list[dict]:
    """
    Run a batch of (url, goal) tasks on a single persistent browser.

    Each task produces its own trajectory directory. The browser
    navigates to the task URL at the start of each episode.

    Returns a list of result dicts (one per task)
    """
    results: list[dict] = []

    with BrowserEnv(headless=headless) as env:
        for i, task in enumerate(tasks):
            url, goal = task["url"], task["goal"]
            try:
                env.goto(url)
                print(f"[batch {i + 1}/{len(tasks)}] goal={goal!r}  url={url}")

                with TrajectoryWriter(trajectories_dir, goal=goal, start_url=url) as tw:
                    reason = run_steps(env, tw, goal=goal, model=model, max_steps=max_steps)
                    tw.set_termination_reason(reason)
                    traj_dir = tw.traj_dir

                from trajectory_store import load_trajectory
                meta = load_trajectory(traj_dir)["metadata"]

                results.append({
                    "status": "ok",
                    "trajectory_id": meta.get("trajectory_id", traj_dir.name),
                    "num_steps": meta.get("num_steps", 0),
                    "termination_reason": meta.get("termination_reason", "unknown"),
                })
                print(f"[batch {i + 1}/{len(tasks)}] done ({reason})")

            except Exception as e:
                results.append({
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "task": task,
                })
                print(f"[batch {i + 1}/{len(tasks)}] ERROR: {e}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python agent_goaldirected.py <url> <goal> [max_steps]")
        sys.exit(1)

    _url = sys.argv[1]
    _goal = sys.argv[2]
    _max = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    traj = run_exploration_episode(_url, _goal, max_steps=_max, headless=False)
    print(f"\nTrajectory saved: {traj}")
