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

Generate exactly {count} short, distinct research questions that cover different core aspects of the topic:
1. Definition & Core Purpose: What the topic is and its primary objective or nature.
2. Architecture & Mechanisms: Its underlying structure, workflow model, or components (e.g., for software/frameworks: graph-based workflow model, nodes, edges, and state management; for science: biological, physical, or chemical mechanisms).
3. Coordination & Practical Application: How it coordinates agents, tools, or processes in practical execution.
4. Documented Capabilities & Limitations: Established capabilities, constraints, and limitations documented in technical or scientific sources.

CRITICAL RULES:
- Ground questions strictly in the topic's actual domain. Do not confuse software framework names with unrelated domains (e.g., do not assume an LLM orchestration tool like LangGraph is a knowledge graph database, graph neural network, or graph theory topic).
- Do NOT generate speculative questions about bias, privacy, societal ethics, or real-time graph updates unless the topic fundamentally demands them.
- Keep each question concise, direct, and focused on searchable factual concepts on Wikipedia.
- Return ONLY a valid JSON object in this exact format:
{{"questions": ["question 1", "question 2", "question 3", "question 4"]}}
The 'questions' array must contain exactly {count} plain strings. Do NOT wrap list items in objects or curly braces.
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
        cleaned = []
        for question in questions:
            if isinstance(question, str) and question.strip():
                cleaned.append(question.strip())
            elif isinstance(question, dict):
                # Handle cases where model wrapped string in a dictionary
                for val in question.values():
                    if isinstance(val, str) and val.strip():
                        cleaned.append(val.strip())
                        break

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
