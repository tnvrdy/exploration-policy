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

import gzip
import json
import queue
import threading
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
        flush_every: int = 1,
        async_writer: bool = False,
        queue_size: int = 256,
        compress_heavy: bool = False,
    ) -> None:
        self.trajectory_id = trajectory_id or _make_id()
        self.goal = goal
        self.start_url = start_url

        self.base_dir = Path(base_dir)
        self.traj_dir = self.base_dir / self.trajectory_id
        self.screenshots_dir = self.traj_dir / "screenshots"
        self.ax_trees_dir = self.traj_dir / "ax_trees"
        self.html_dir = self.traj_dir / "html"
        self.steps_path = self.traj_dir / "steps.jsonl"
        self.metadata_path = self.traj_dir / "metadata.json"
        self.manifest_path = self.base_dir / "manifest.jsonl"

        self._step_count = 0
        self._record_count = 0
        self._start_time = _now_iso()
        self._termination_reason: str = "unknown"
        self._extra_metadata: dict[str, Any] = {}
        self._steps_file = None
        self._flush_every = max(1, flush_every)
        self._steps_since_flush = 0
        self._compress_heavy = compress_heavy
        self._async_writer = async_writer
        self._queue_size = max(16, queue_size)
        self._write_queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._worker_error: Exception | None = None
        self._closed = False

    def __enter__(self) -> TrajectoryWriter:
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        self.ax_trees_dir.mkdir(exist_ok=True)
        self.html_dir.mkdir(exist_ok=True)
        self._steps_file = open(self.steps_path, "a", encoding="utf-8")
        if self._async_writer:
            self._write_queue = queue.Queue(maxsize=self._queue_size)
            self._worker = threading.Thread(target=self._writer_loop, daemon=True)
            self._worker.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._closed = True
        if self._async_writer and self._write_queue is not None:
            self._write_queue.put(None)
            if self._worker is not None:
                self._worker.join()
        if self._worker_error:
            raise self._worker_error
        if self._steps_file:
            self._steps_file.flush()
            self._steps_file.close()
        self._write_metadata()
        self._append_manifest()

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
        count_toward_steps: bool = True,
    ) -> None:
        if self._worker_error:
            raise self._worker_error
        ax_tree = state.get("ax_tree", "")
        html = state.get("html", "")

        ax_path: Path | None
        html_path: Path | None
        if ax_tree:
            if self._compress_heavy:
                ax_path = self.ax_trees_dir / f"step_{step:03d}.txt.gz"
            else:
                ax_path = self.ax_trees_dir / f"step_{step:03d}.txt"
        else:
            ax_path = None
        if html:
            if self._compress_heavy:
                html_path = self.html_dir / f"step_{step:03d}.html.gz"
            else:
                html_path = self.html_dir / f"step_{step:03d}.html"
        else:
            html_path = None

        payload = {
            "step": step,
            "state": state,
            "action": action,
            "action_ok": action_ok,
            "extra": extra,
            "ax_path": ax_path,
            "html_path": html_path,
            "ax_tree": ax_tree,
            "html": html,
        }
        if self._async_writer:
            assert self._write_queue is not None
            self._write_queue.put(payload)
        else:
            self._write_payload(payload)
        self._record_count += 1
        if count_toward_steps:
            self._step_count += 1

    def set_termination_reason(self, reason: str) -> None:
        self._termination_reason = reason

    def set_goal(self, goal: str) -> None:
        """Update the goal after trajectory collection (for retroactive labeling)."""
        self.goal = goal

    def add_metadata(self, data: dict[str, Any]) -> None:
        """Attach extra metadata fields to metadata.json."""
        self._extra_metadata.update(data)

    def _write_metadata(self) -> None:
        meta = {
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "start_url": self.start_url,
            "num_steps": self._step_count,
            "num_records": self._record_count,
            "start_time": self._start_time,
            "end_time": _now_iso(),
            "termination_reason": self._termination_reason,
        }
        meta.update(self._extra_metadata)
        self.metadata_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
        )

    def _append_manifest(self) -> None:
        entry = {
            "schema_version": 1,
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "start_url": self.start_url,
            "num_steps": self._step_count,
            "num_records": self._record_count,
            "termination_reason": self._termination_reason,
            "trajectory_dir": str(self.traj_dir),
            "metadata_path": str(self.metadata_path),
            "steps_path": str(self.steps_path),
            "created_at": self._start_time,
            "updated_at": _now_iso(),
        }
        lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_f:
            _flock(lock_f, exclusive=True)
            try:
                with open(self.manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            finally:
                _flock(lock_f, exclusive=False)

    def _writer_loop(self) -> None:
        assert self._write_queue is not None
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            try:
                self._write_payload(item)
            except Exception as e:  # pragma: no cover
                self._worker_error = e
                break

    def _write_payload(self, payload: dict[str, Any]) -> None:
        step = payload["step"]
        ax_path = payload["ax_path"]
        html_path = payload["html_path"]
        ax_tree = payload["ax_tree"]
        html = payload["html"]
        state = payload["state"]
        action = payload["action"]
        action_ok = payload["action_ok"]
        extra = payload.get("extra")

        if ax_path:
            _write_text(ax_path, ax_tree)
        if html_path:
            _write_text(html_path, html)

        record: dict[str, Any] = {
            "step": step,
            "timestamp": _now_iso(),
            "url": state.get("url", ""),
            "title": state.get("title", ""),
            "action": action,
            "action_ok": action_ok,
            "screenshot_path": state.get("screenshot_path", ""),
            "ax_tree_path": str(ax_path) if ax_path else "",
            "html_path": str(html_path) if html_path else "",
        }
        if extra:
            record.update(extra)

        self._steps_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._steps_since_flush += 1
        if self._steps_since_flush >= self._flush_every:
            self._steps_file.flush()
            self._steps_since_flush = 0


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
                        step["ax_tree"] = _read_text(Path(ax_path)) if ax_path else ""
                        step["html"] = _read_text(Path(html_path)) if html_path else ""

                    steps.append(step)

    return {"metadata": metadata, "steps": steps}


def load_trajectory_metadata(traj_dir: str | Path) -> dict:
    """Load only metadata.json for a trajectory."""
    meta_path = Path(traj_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


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


def iter_manifest(base_dir: str | Path):
    """
    Yield manifest entries from base_dir/manifest.jsonl if present.
    """
    manifest_path = Path(base_dir) / "manifest.jsonl"
    if not manifest_path.exists():
        return
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("trajectory_id")
            tdir = row.get("trajectory_dir")
            if tid and tdir:
                yield tid, Path(tdir), row


def _make_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:6]
    return f"traj_{ts}_{short}"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_text(path: Path, text: str) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


def _flock(file_obj, *, exclusive: bool) -> None:
    try:
        import fcntl

        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_UN
        fcntl.flock(file_obj.fileno(), mode)
    except Exception:
        # Best-effort only; some platforms may not support fcntl.
        return
