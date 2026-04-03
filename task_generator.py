"""
Synthetic task generator for the exploration policy.

Calls Qwen to produce diverse (url, goal) pairs for a set of seed websites. Outputs jsonl that feeds 
directly into the exploration loop (agent.py)

Usage:
    python task_generator.py -o tasks.jsonl -n 20
    python task_generator.py -o tasks.jsonl -n 50 --seeds custom_seeds.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from llm import chat

# seed websites

DEFAULT_SEEDS: list[dict] = [
    {
        "url": "https://en.wikipedia.org",
        "description": "Free encyclopedia with articles on every topic, internal links, tables, infoboxes, and references sections",
    },
    {
        "url": "https://www.reddit.com",
        "description": "Social news aggregation with subreddits, posts, comments, upvotes, and user profiles",
    },
    {
        "url": "https://news.ycombinator.com",
        "description": "Tech news aggregator with ranked story links, comment threads, and user pages",
    },
    {
        "url": "https://stackoverflow.com",
        "description": "Programming Q&A with questions, answers, tags, voting, and user profiles",
    },
    {
        "url": "https://github.com/explore",
        "description": "Code hosting with repositories, issues, pull requests, trending projects, and topic pages",
    },
    {
        "url": "https://www.amazon.com",
        "description": "E-commerce with product search, categories, filters, product pages, reviews, and wishlists",
    },
    {
        "url": "https://www.bbc.com/news",
        "description": "International news with articles, sections (world, business, tech, sport), and live coverage pages",
    },
    {
        "url": "https://www.imdb.com",
        "description": "Movie and TV database with search, title pages, cast lists, ratings, and top-250 charts",
    },
    {
        "url": "https://www.espn.com",
        "description": "Sports news with scores, standings, team pages, player stats, and schedules",
    },
    {
        "url": "https://arxiv.org",
        "description": "Academic preprint archive with search, paper abstracts, author pages, and subject categories",
    },
    {
        "url": "https://www.allrecipes.com",
        "description": "Recipe site with search, categories, recipe pages with ingredients and instructions, and reviews",
    },
    {
        "url": "https://www.craigslist.org",
        "description": "Classified ads organized by city and category (housing, jobs, for sale, services)",
    },
    {
        "url": "https://duckduckgo.com",
        "description": "Privacy-focused search engine with web search, instant answers, and settings",
    },
    {
        "url": "https://www.goodreads.com",
        "description": "Book discovery with search, book pages, author pages, reviews, and reading lists",
    },
    {
        "url": "https://www.weather.gov",
        "description": "US weather forecasts with location search, radar maps, and detailed forecasts by region",
    },
    {
        "url": "https://www.openstreetmap.org",
        "description": "Open-source map with search, zoom, pan, layer selection, and location details",
    },
]


_TASK_GEN_PROMPT = """\
You are generating synthetic browsing tasks for a web agent that will explore {url}.

Site description: {description}

The agent can perform these actions:
  - click on links, buttons, and interactive elements
  - type text into input fields (optionally pressing Enter to submit)
  - scroll up or down on the page
  - navigate to a URL
  - stop when done

Generate exactly {n} diverse, specific browsing tasks for this site. Each task should:
  - Be achievable in 1-4 browser actions (clicks, typing, scrolling)
  - Be concrete and specific (not vague like "explore the site")
  - NOT require logging in, creating an account, or entering personal information
  - NOT require drag-and-drop, file uploads, or other actions beyond click/type/scroll
  - Cover a variety of interaction types: searching, navigating to specific pages, finding specific information, using filters, reading content

Return a JSON array of objects, each with a "goal" field. Example:
[
  {{"goal": "Search for 'machine learning' and click on the first result"}},
  {{"goal": "Navigate to the sports section and find today's top headline"}}
]

Return ONLY the JSON array, no other text."""


def _build_prompt(url: str, description: str, n: int) -> str:
    return _TASK_GEN_PROMPT.format(url=url, description=description, n=n)


# json response parsing

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that LLMs commonly add."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_tasks_response(raw: str, url: str) -> list[dict]:
    """Parse LLM response into a list of {"url": ..., "goal": ...} dicts."""
    cleaned = _strip_markdown_fences(raw)
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    tasks = []
    for item in items:
        goal = item.get("goal", "").strip() if isinstance(item, dict) else ""
        if goal:
            tasks.append({"url": url, "goal": goal})
    return tasks

# core task generation

def generate_tasks_for_site(
    url: str,
    description: str,
    n: int = 20,
    model: str | None = None,
) -> list[dict]:
    """
    Call Qwen to generate n tasks for a single site.
    Returns a list of {"url": ..., "goal": ...} dicts.
    """
    prompt = _build_prompt(url, description, n)
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.9,
        max_tokens=4096,
    )
    return _parse_tasks_response(raw, url)


def generate_all_tasks(
    seeds: list[dict] | None = None,
    tasks_per_site: int = 20,
    output_path: str | Path = "tasks.jsonl",
    model: str | None = None,
) -> int:
    """
    Generate tasks for all seed sites and write to a JSONL file.
    Returns the total number of tasks generated.
    """
    seeds = seeds or DEFAULT_SEEDS
    output_path = Path(output_path)
    total = 0

    with open(output_path, "w") as f:
        for i, seed in enumerate(seeds):
            url = seed["url"]
            desc = seed["description"]
            print(f"[{i + 1}/{len(seeds)}] generating {tasks_per_site} tasks for {url} ...")

            try:
                tasks = generate_tasks_for_site(url, desc, n=tasks_per_site, model=model)
            except Exception as e:
                print(f"  WARNING: failed for {url}: {e}")
                continue

            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

            total += len(tasks)
            print(f"  got {len(tasks)} tasks (total so far: {total})")

    print(f"\nDone. {total} tasks written to {output_path}")
    return total


def _load_seeds(path: str | Path) -> list[dict]:
    """Load seeds from a JSONL file (each line: {"url": ..., "description": ...})."""
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic browsing tasks")
    parser.add_argument("-o", "--output", default="tasks.jsonl", help="Output JSONL path")
    parser.add_argument("-n", "--tasks-per-site", type=int, default=20, help="Tasks to generate per site")
    parser.add_argument("--seeds", default=None, help="Custom seeds JSONL file (overrides defaults)")
    parser.add_argument("--model", default=None, help="LLM model override")
    args = parser.parse_args()

    seeds = _load_seeds(args.seeds) if args.seeds else None
    generate_all_tasks(
        seeds=seeds,
        tasks_per_site=args.tasks_per_site,
        output_path=args.output,
        model=args.model,
    )
