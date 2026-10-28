
import json
from typing import Callable

from backend.mistral_client import ask_mistral
from backend.verifier import parse_json_response


def generate_research_questions(
    topic: str,
    count: int = 4,
    ask_fn: Callable[[str], str] = ask_mistral,
) -> list[str]:
    """Use Mistral to plan focused searches for a research topic."""
    cleaned_topic = topic.strip() if topic else ""
    if not cleaned_topic:
        raise ValueError("Research topic cannot be empty.")

    prompt = f"""
You are a research planning assistant.

The user wants to research this topic:
{cleaned_topic}

Generate exactly {count} short, distinct research questions that
cover different important aspects of the topic.

Requirements:
- Keep every question focused and searchable on Wikipedia.
- Avoid repeating the same idea.
- Include foundational concepts and relevant applications or limitations.
- Return ONLY a valid JSON object in this exact format:
{{"questions": ["question 1", "question 2"]}}
- Make each question cover a distinct aspect of the topic.
- For questions about solutions, ask for evidence of effectiveness.
- Prefer specific, searchable concepts over broad related topics.
- Do not assume a particular solution works without evidence.
- Include at least one question about limitations or unintended effects.

The questions array must contain exactly {count} strings.
Do not include Markdown fences or explanations.
"""

    response = ask_fn(prompt)

    try:
        data = parse_json_response(response)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got: {type(data).__name__}")

        questions = data.get("questions")
        if not isinstance(questions, list):
            raise ValueError("'questions' must be a list")

        # Keep only non-empty strings and cap the number of searches.
        cleaned = [
            question.strip()
            for question in questions
            if isinstance(question, str) and question.strip()
        ]

        # Remove duplicates while preserving order.
        unique_questions = list(dict.fromkeys(cleaned))
        unique_questions = unique_questions[:count]

        if not unique_questions:
            raise ValueError("The model returned no usable questions")

        return unique_questions

    except Exception as error:
        raise ValueError(
            f"Could not parse the research plan. Model response: {response[:200]}"
        ) from error


if __name__ == "__main__":
    topic = input("Enter a research topic: ").strip()

    if topic:
        try:
            questions = generate_research_questions(topic)

            print("\nResearch plan:")
            for number, question in enumerate(questions, start=1):
                print(f"{number}. {question}")

        except Exception as error:
            print(f"Planner failed: {error}")
    else:
        print("Please enter a topic.")

