"""
Shared step-loop logic used by both goal-directed and freeform exploration agents.
"""

from __future__ import annotations

from actions import ACTION_VOCABULARY, ActionParseError, parse_action
from browser_env import BrowserEnv, Observation
from llm import chat
from trajectory_store import TrajectoryWriter


SYSTEM_PROMPT = f"""You are an autonomous browser agent. You control a real web browser.

Each turn you will receive:
  - GOAL: the task you must complete
  - URL: the current page URL
  - PAGE: a numbered list of interactive elements on the current page
  - HISTORY: the actions you have already taken (empty on the first turn)

Your job is to decide the single next action that best makes progress toward the GOAL.

{ACTION_VOCABULARY}

Rules:
1. Output EXACTLY ONE action per reply, on a single line. Nothing else — no explanation, no preamble.
2. Use only the actions listed above. Any other output will be treated as a parse error.
3. When the GOAL is achieved, output: stop
4. If the page does not help and you cannot make progress, output: stop""".strip()


def build_user_message(
    goal: str,
    obs_text: str,
    action_history: list[str],
) -> str:
    history_block = (
        "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_history))
        if action_history
        else "  (none)"
    )
    return (
        f"GOAL: {goal}\n\n"
        f"HISTORY:\n{history_block}\n\n"
        f"PAGE:\n{obs_text}"
    )


def first_line(text: str) -> str:
    """Extract the first non-empty line from model output."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def annotate_action(parsed, raw: str, obs: Observation, env: BrowserEnv) -> str:
    """Build a history entry with semantic context."""
    if parsed.action_type == "click" and parsed.index is not None:
        label = (
            obs.element_descs[parsed.index]
            if parsed.index < len(obs.element_descs)
            else "?"
        )
        return f'click {parsed.index}  →  "{label}"  →  {env.page.url}'
    if parsed.action_type == "type" and parsed.submit:
        return f"{raw}  →  {env.page.url}"
    if parsed.action_type == "goto":
        return f"{raw}  →  {env.page.url}"
    return raw


def run_episode_steps(
    env: BrowserEnv,
    tw: TrajectoryWriter,
    goal: str,
    model: str | None = None,
    max_steps: int = 4,
) -> str:
    """
    Run the core observe -> act -> record loop for one micro-episode.

    Assumes env is already navigated to a starting page and tw is open.
    Returns the termination reason string
    """
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    action_history: list[str] = []
    consecutive_failures = 0

    for step_num in range(max_steps):
        obs = env.get_text_observation()

        user_msg = {
            "role": "user",
            "content": build_user_message(goal, obs.text, action_history),
        }
        raw_full = chat([system_msg, user_msg], model=model)
        raw = first_line(raw_full)
        print(f"  [step {step_num}] model: {raw}")

        parsed = None
        parse_error: str | None = None
        try:
            parsed = parse_action(raw)
        except ActionParseError as e:
            parse_error = str(e)
            print(f"  [step {step_num}] parse error: {parse_error!r}")

        exec_result: dict | None = None
        if parsed is not None:
            exec_result = env.execute_action(parsed)
            print(f"  [step {step_num}] exec: {exec_result!r}")

        screenshot_path = tw.screenshot_path_for(step_num)
        state = env.capture_full_state(screenshot_path)

        exec_ok = bool(exec_result and exec_result.get("ok"))
        tw.write_step(
            step=step_num,
            state=state,
            action=raw,
            action_ok=exec_ok,
            extra={
                "parse_error": parse_error,
                "exec_error": (exec_result or {}).get("error"),
                "raw_model_output": raw_full,
            },
        )

        if parsed is not None:
            if not exec_ok:
                err = (exec_result or {}).get("error", "unknown")
                history_entry = f"{raw}  [failed: {err.splitlines()[0]}]"
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                history_entry = annotate_action(parsed, raw, obs, env)
            action_history.append(history_entry)

        if parsed is not None and parsed.action_type == "stop":
            return "stop"
        if consecutive_failures >= 3:
            return "consecutive_failures"

    return "max_steps"
