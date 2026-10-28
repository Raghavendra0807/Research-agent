
import json
from typing import Callable

from backend.mistral_client import ask_mistral
from backend.verifier import parse_json_response


def filter_relevant_sources(
    topic: str,
    sources: list[dict],
    max_sources: int = 6,
    ask_fn: Callable[[str], str] = ask_mistral,
) -> list[dict]:
    """Rank candidate sources by relevance to a research topic."""
    if not sources or not topic or not topic.strip():
        return []

    candidates = [
        {
            "id": source.get("id", f"C{i}"),
            "title": source.get("title", ""),
            "text": source.get("text", "")[:1200],
        }
        for i, source in enumerate(sources, start=1)
    ]

    prompt = f"""
You are a strict research retrieval evaluator.

Research topic:
{topic.strip()}

Candidate Wikipedia sources:
{json.dumps(candidates, ensure_ascii=False)}

Select up to {max_sources} sources that are most directly relevant
to the research topic.

Rules:
- Judge relevance using both the title and the provided text.
- Prefer sources that directly address the topic.
- Do not select a source merely because it shares a broad keyword.
- Exclude unrelated pages.
- Do not claim that a source supports a specific fact unless its text
  actually contains evidence for that fact.
- Return only the selected candidate IDs.
- Do not invent IDs.

Return only valid JSON:
{{"selected_ids": ["C1", "C3"]}}
"""

    response = ask_fn(prompt)
    data = parse_json_response(response)

    if not isinstance(data, dict):
        raise ValueError("Relevance filter response was not a JSON dictionary.")

    selected_ids = data.get("selected_ids")
    if not isinstance(selected_ids, list):
        raise ValueError("Relevance filter did not return a 'selected_ids' list.")

    candidate_ids = {item["id"] for item in candidates}
    selected_ids = [
        source_id for source_id in selected_ids
        if isinstance(source_id, str) and source_id in candidate_ids
    ]

    source_by_id = {
        item["id"]: source
        for item, source in zip(candidates, sources)
    }

    selected = []
    for source_id in selected_ids:
        source = source_by_id[source_id]
        if source not in selected:
            selected.append(source)
        if len(selected) >= max_sources:
            break

    return selected



if __name__ == "__main__":
    sample_sources = [
        {
            "id": "C1",
            "title": "Food bank",
            "text": "A food bank collects and distributes food to people in need.",
        },
        {
            "id": "C2",
            "title": "2026 Vancouver municipal election",
            "text": "Information about candidates and the municipal election.",
        },
        {
            "id": "C3",
            "title": "Food rescue",
            "text": "Food rescue redirects surplus edible food to people and organizations.",
        },
    ]

    try:
        filtered = filter_relevant_sources(
            "food redistribution programs",
            sample_sources,
            max_sources=2,
        )

        print("Selected sources:")
        for source in filtered:
            print(f"- {source['title']}")
    except Exception as error:
        print(f"Filtering failed: {error}")
