
from typing import Callable, Optional

from backend.mistral_client import ask_mistral
from backend.planner import generate_research_questions
from backend.relevance_filter import filter_relevant_sources
from backend.verifier import (
    generate_verification_summary,
    validate_citations,
    verify_briefing,
)
from backend.wikimedia_client import search_wikipedia


class ResearchResult(dict):
    """
    Structured research pipeline result. Supports dictionary access and
    string conversion for CLI backward-compatibility.
    """

    def __str__(self) -> str:
        return self.get("formatted_report", "")


def format_report_text(
    topic: str,
    briefing: str,
    sources: list[dict],
    claims: list[dict],
    warnings: list[str],
) -> str:
    """Format pipeline results into a clean human-readable CLI report."""
    sections = []

    if briefing:
        sections.append(briefing)

    if sources:
        sections.append("\n--- Retrieved sources ---")
        sections.append(
            "\n".join(
                f"[{source.get('id', 'S?')}] {source.get('title', 'Untitled')} — {source.get('url', '')}"
                for source in sources
            )
        )

    if claims:
        sections.append("\n--- Claim verification ---")
        for number, item in enumerate(claims, start=1):
            claim_text = item.get("claim", "")
            verdict = item.get("verdict", "unknown").upper()
            explanation = item.get("explanation", "")
            matched_sources = ", ".join(item.get("source_ids", [])) or "None"
            excerpt = item.get("supporting_excerpt", "")

            claim_block = [
                f"\n{number}. {claim_text}",
                f"Verdict: {verdict}",
                f"Explanation: {explanation}",
                f"Evidence sources: {matched_sources}",
            ]

            if excerpt:
                claim_block.append(f"Traceable excerpt: \"{excerpt}\"")

            sections.append("\n".join(claim_block))
    else:
        sections.append(
            "\nClaim verification was not completed. Treat the briefing as unverified."
        )

    if warnings:
        sections.append("\n--- Warnings & Integrity Checks ---")
        sections.append("\n".join(f"- {w}" for w in warnings))

    return "\n".join(sections)


def research_topic(
    topic: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    ask_fn: Callable[[str], str] = ask_mistral,
    search_fn: Callable[[str, int], list[dict]] = search_wikipedia,
) -> ResearchResult:
    """
    Execute the research pipeline: plan questions, retrieve sources,
    filter relevance, synthesize cited briefing, and verify factual claims.
    """
    def notify(stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage, message)
        else:
            print(f"[{stage}] {message}")

    cleaned_topic = topic.strip()
    if not cleaned_topic:
        empty_msg = "Please provide a valid research topic."
        return ResearchResult({
            "topic": "",
            "questions": [],
            "sources": [],
            "briefing": "",
            "claims": [],
            "verification_summary": {},
            "warnings": [empty_msg],
            "errors": [empty_msg],
            "formatted_report": empty_msg,
            "diagnostics": {
                "stage_failed": "input_validation",
                "details": empty_msg,
            },
        })

    notify("planning", f"Planning research questions for: {cleaned_topic}")

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Ask Mistral to plan the research.
    try:
        questions = generate_research_questions(cleaned_topic, count=4, ask_fn=ask_fn)
    except Exception as error:
        err_msg = f"Stage failed at Planning: Failed to generate research plan: {error}"
        errors.append(err_msg)
        return ResearchResult({
            "topic": cleaned_topic,
            "questions": [],
            "sources": [],
            "briefing": "",
            "claims": [],
            "verification_summary": {},
            "warnings": warnings,
            "errors": errors,
            "formatted_report": f"Research failed during planning: {error}",
            "diagnostics": {
                "stage_failed": "planning",
                "details": str(error),
            },
        })

    notify("planning_complete", f"Formulated {len(questions)} research questions.")

    # 2. Search Wikimedia for every research question.
    unique_sources: dict[str, dict] = {}
    for number, question in enumerate(questions, start=1):
        notify("retrieval", f"Searching ({number}/{len(questions)}): {question}")

        try:
            results = search_fn(question, 2)
        except Exception as error:
            warnings.append(f"Search failed for query '{question}': {error}")
            continue

        if not isinstance(results, list):
            continue

        for source in results:
            if not isinstance(source, dict):
                continue

            title = source.get("title", "")
            url = source.get("url", "")
            text = source.get("text", "")

            if not isinstance(title, str) or not isinstance(text, str):
                continue

            title = title.strip()
            text = text.strip()
            url = url.strip() if isinstance(url, str) else ""

            if not title or not text:
                continue

            key = title.casefold()
            if key not in unique_sources:
                unique_sources[key] = {
                    "title": title,
                    "url": url,
                    "text": text,
                }

    candidate_sources = list(unique_sources.values())

    if not candidate_sources:
        no_sources_msg = "Stage failed at Retrieval: Could not retrieve usable source material from Wikipedia. Please try another topic."
        errors.append(no_sources_msg)
        return ResearchResult({
            "topic": cleaned_topic,
            "questions": questions,
            "sources": [],
            "briefing": "",
            "claims": [],
            "verification_summary": {},
            "warnings": warnings,
            "errors": errors,
            "formatted_report": no_sources_msg,
            "diagnostics": {
                "stage_failed": "retrieval",
                "questions_count": len(questions),
                "candidates_count": 0,
                "details": "Wikipedia search returned no usable article extracts across all formulated research questions.",
            },
        })

    notify("filtering", f"Filtering {len(candidate_sources)} candidates for topic relevance...")

    # 3. Filter relevant sources.
    try:
        sources = filter_relevant_sources(
            topic=cleaned_topic,
            sources=candidate_sources,
            max_sources=6,
            ask_fn=ask_fn,
        )
    except Exception as error:
        err_msg = f"Stage failed at Relevance Filtering: Relevance filtering failed: {error}"
        errors.append(err_msg)
        return ResearchResult({
            "topic": cleaned_topic,
            "questions": questions,
            "sources": [],
            "briefing": "",
            "claims": [],
            "verification_summary": {},
            "warnings": warnings,
            "errors": errors,
            "formatted_report": "Research stopped because source relevance could not be verified.",
            "diagnostics": {
                "stage_failed": "relevance_filter",
                "questions_count": len(questions),
                "candidates_count": len(candidate_sources),
                "details": f"Relevance filtering error: {error}",
            },
        })

    if not sources:
        notify("filtering_failed", f"No sufficiently relevant sources found for '{cleaned_topic}'.")
        no_rel_msg = (
            f"Stage failed at Relevance Filtering: No sufficiently relevant sources were found. "
            f"Evaluated {len(candidate_sources)} candidate article(s) from Wikipedia across {len(questions)} research questions, "
            f"but none contained sufficient evidence directly addressing '{cleaned_topic}'. "
            f"Wikipedia may lack an established article or coverage for this topic; try a broader topic or one with dedicated Wikipedia articles."
        )
        errors.append(no_rel_msg)
        return ResearchResult({
            "topic": cleaned_topic,
            "questions": questions,
            "sources": [],
            "briefing": "",
            "claims": [],
            "verification_summary": {},
            "warnings": warnings,
            "errors": errors,
            "formatted_report": no_rel_msg,
            "diagnostics": {
                "stage_failed": "relevance_filter",
                "questions_count": len(questions),
                "candidates_count": len(candidate_sources),
                "relevant_count": 0,
                "candidate_titles": [s.get("title", "") for s in candidate_sources[:10]],
                "details": (
                    f"Evaluated {len(candidate_sources)} candidate article(s), but none met strict relevance "
                    f"criteria for '{cleaned_topic}'."
                ),
            },
        })

    # Assign stable source IDs and package evidence context
    source_context = []
    for index, source in enumerate(sources, start=1):
        source["id"] = f"S{index}"
        source_context.append(
            f"[{source['id']}]\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Evidence: {source['text']}"
        )

    context = "\n\n".join(source_context)
    valid_source_ids = [s["id"] for s in sources]

    notify("synthesis", f"Synthesizing briefing from {len(sources)} verified sources...")

    # 4. Strengthened briefing prompt to prevent unsupported facts and statistics.
    briefing_prompt = f"""
You are a rigorous, objective research assistant.

Research topic: {cleaned_topic}

Research questions to address:
{chr(10).join("- " + q for q in questions)}

TASK:
Write a concise, high-quality research briefing using ONLY the provided SOURCE MATERIAL.

CRITICAL FACTUAL INTEGRITY RULES:
1. STRICTLY GROUNDED: Every substantive statement, numerical figure, percentage, date, metric, causal assertion, or outcome MUST be directly supported by the text in SOURCE MATERIAL.
2. NO UNSUPPORTED FACTS OR STATISTICS: Do NOT introduce external knowledge, outside statistics, unevidenced examples, or policy recommendations as established facts. Never estimate, extrapolate, or invent numbers.
3. MANDATORY INLINE CITATIONS: Immediately follow every factual statement with its corresponding source ID in square brackets, e.g., [S1] or [S2].
4. VALID CITATIONS ONLY: Cite ONLY source IDs that exist in the SOURCE MATERIAL ({', '.join(valid_source_ids)}). Never invent or hallucinate source IDs.
5. NO CITATION AS EMPTY SHIELD: Do not attach a citation tag unless that source explicitly provides the fact.
6. EXPLICIT LIMITATIONS: If the source material does not provide data or answers for any aspect or research question, explicitly state: "Information not specified in the retrieved sources" rather than speculating.

RECOMMENDED STRUCTURE:
# Executive Summary
(Overview of findings directly grounded in the sources, with citations)

# Key Findings
(Detailed answers to the research questions, strictly cited with [S1], [S2], etc.)

# Limitations & Unanswered Questions
(Explicitly noting gaps where evidence was missing or inconclusive)

SOURCE MATERIAL:
{context}
"""

    briefing = ask_fn(briefing_prompt)

    # Validate overall briefing citations
    citation_audit = validate_citations(briefing, valid_source_ids)
    if citation_audit["invalid_ids"]:
        warnings.append(
            f"Briefing hallucinated nonexistent source ID(s): {', '.join(citation_audit['invalid_ids'])}"
        )

    notify("verification", "Verifying factual claims and citation integrity...")

    # 5. Extract and verify claims
    try:
        claims = verify_briefing(briefing, sources, ask_fn=ask_fn)
    except Exception as error:
        warnings.append(f"Claim verification error: {error}")
        claims = []

    # 6. Generate verification summary and compile warnings
    summary = generate_verification_summary(claims)
    warnings.extend(summary.get("warnings", []))

    formatted_report = format_report_text(
        topic=cleaned_topic,
        briefing=briefing,
        sources=sources,
        claims=claims,
        warnings=warnings,
    )

    notify("complete", "Research pipeline completed successfully.")

    return ResearchResult({
        "topic": cleaned_topic,
        "questions": questions,
        "sources": sources,
        "briefing": briefing,
        "claims": claims,
        "verification_summary": summary,
        "warnings": warnings,
        "errors": errors,
        "formatted_report": formatted_report,
        "diagnostics": {
            "stage_failed": None,
            "questions_count": len(questions),
            "candidates_count": len(candidate_sources),
            "relevant_count": len(sources),
            "claims_count": len(claims),
        },
    })


if __name__ == "__main__":
    topic_input = input("Enter a research topic: ").strip()

    if not topic_input:
        print("Please enter a topic.")
    else:
        try:
            print("\n" + "=" * 60)
            result = research_topic(topic_input)
            print(result)
            print("=" * 60)
        except Exception as exc:
            print(f"\nResearch failed: {exc}")
