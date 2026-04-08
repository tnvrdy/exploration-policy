"""
Raw trajectory storage for exploration episodes.

Layout on disk:
    <base_dir>/
      traj_<id>/
        metadata.json
        steps.jsonl          # lightweight: action, url, title, ok, file paths
        screenshots/step_000.png
        ax_trees/step_000.txt
        html/step_000.html

Purely raw data capture, wrangling into chatml/training formats will happen downstream.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class TrajectoryWriter:
    """
    Accumulates steps for a single trajectory and writes them to disk.

    Heavy fields (ax_tree, html) are written to separate files per step
    so that steps.jsonl is easily human-readable.

    Usage:
        with TrajectoryWriter(base_dir, goal=goal, start_url=url) as tw:
            tw.write_step(step=0, state=state, action="click 3", action_ok=True)
            tw.write_step(step=1, ...)
        # metadata.json is finalized on __exit__
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        goal: str,
        start_url: str,
        trajectory_id: str | None = None,
    ) -> None:
        self.trajectory_id = trajectory_id or _make_id()
        self.goal = goal
        self.start_url = start_url

        self.traj_dir = Path(base_dir) / self.trajectory_id
        self.screenshots_dir = self.traj_dir / "screenshots"
        self.ax_trees_dir = self.traj_dir / "ax_trees"
        self.html_dir = self.traj_dir / "html"
        self.steps_path = self.traj_dir / "steps.jsonl"
        self.metadata_path = self.traj_dir / "metadata.json"

        self._step_count = 0
        self._start_time = _now_iso()
        self._termination_reason: str = "unknown"
        self._steps_file = None

    def __enter__(self) -> TrajectoryWriter:
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        self.ax_trees_dir.mkdir(exist_ok=True)
        self.html_dir.mkdir(exist_ok=True)
        self._steps_file = open(self.steps_path, "a")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._steps_file:
            self._steps_file.close()
        self._write_metadata()

    def screenshot_path_for(self, step: int) -> Path:
        """Return the screenshot path for a given step number."""
        return self.screenshots_dir / f"step_{step:03d}.png"

    def write_step(
        self,
        *,
        step: int,
        state: dict,
        action: str,
        action_ok: bool,
        extra: dict | None = None,
    ) -> None:
        ax_tree = state.get("ax_tree", "")
        html = state.get("html", "")

        ax_path = self.ax_trees_dir / f"step_{step:03d}.txt"
        html_path = self.html_dir / f"step_{step:03d}.html"
        ax_path.write_text(ax_tree, encoding="utf-8")
        html_path.write_text(html, encoding="utf-8")

        record: dict[str, Any] = {
            "step": step,
            "timestamp": _now_iso(),
            "url": state.get("url", ""),
            "title": state.get("title", ""),
            "action": action,
            "action_ok": action_ok,
            "screenshot_path": state.get("screenshot_path", ""),
            "ax_tree_path": str(ax_path),
            "html_path": str(html_path),
        }
        if extra:
            record.update(extra)

        self._steps_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._steps_file.flush()
        self._step_count += 1

    def set_termination_reason(self, reason: str) -> None:
        self._termination_reason = reason

    def set_goal(self, goal: str) -> None:
        """Update the goal after trajectory collection (for retroactive labeling)."""
        self.goal = goal

    def _write_metadata(self) -> None:
        meta = {
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "start_url": self.start_url,
            "num_steps": self._step_count,
            "start_time": self._start_time,
            "end_time": _now_iso(),
            "termination_reason": self._termination_reason,
        }
        self.metadata_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        )


# post-hoc metadata updates

def update_metadata(traj_dir: str | Path, updates: dict) -> dict:
    """
    Merge key-value pairs into an existing trajectory's metadata.json.
    Returns the updated metadata dict.
    """
    traj_dir = Path(traj_dir)
    meta_path = traj_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


# load/read trajectories

def load_trajectory(traj_dir: str | Path, include_heavy: bool = True) -> dict:
    """
    Load a single trajectory from disk.

    Args:
        traj_dir: path to the trajectory directory
        include_heavy: if True (default), read ax_tree and html content
            from their separate files and include them in each step dict.
            If False, steps only contain the lightweight fields from steps.jsonl.

    Returns:
        {"metadata": dict, "steps": list[dict]}
    """
    traj_dir = Path(traj_dir)
    meta_path = traj_dir / "metadata.json"
    steps_path = traj_dir / "steps.jsonl"

    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    steps: list[dict] = []
    if steps_path.exists():
        with open(steps_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    step = json.loads(line)

                    if include_heavy:
                        ax_path = step.get("ax_tree_path", "")
                        html_path = step.get("html_path", "")
                        step["ax_tree"] = Path(ax_path).read_text(encoding="utf-8") if ax_path else ""
                        step["html"] = Path(html_path).read_text(encoding="utf-8") if html_path else ""

                    steps.append(step)

    return {"metadata": metadata, "steps": steps}


def iter_trajectories(base_dir: str | Path):
    """
    Yield (trajectory_id, traj_dir_path) for every trajectory under base_dir,
    sorted by directory name (i.e. chronological)
    """
    base = Path(base_dir)
    if not base.exists():
        return
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "metadata.json").exists():
            yield child.name, child


def _make_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:6]
    return f"traj_{ts}_{short}"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
