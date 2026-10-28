
import json
import re
from typing import Callable, Optional

from backend.mistral_client import ask_mistral


def extract_citations_from_text(text: str) -> list[str]:
    """
    Extract source reference IDs such as [S1], [S2], [S1, S2] from text.
    Returns a sorted list of unique normalized source IDs (e.g. ['S1', 'S2']).
    """
    if not text:
        return []

    # Matches bracketed expressions like [S1], [S1, S2], [s3]
    raw_brackets = re.findall(r"\[([sS]\d+(?:\s*,\s*[sS]\d+)*)\]", text)
    found_ids = set()
    for bracket in raw_brackets:
        parts = re.split(r"\s*,\s*", bracket)
        for part in parts:
            normalized = part.strip().upper()
            if re.match(r"^S\d+$", normalized):
                found_ids.add(normalized)

    # Sort numerically by source number
    return sorted(found_ids, key=lambda sid: int(sid[1:]) if sid[1:].isdigit() else 999)


def validate_citations(text: str, valid_source_ids: set[str] | list[str]) -> dict:
    """
    Validate all citations in a text against a set of valid source IDs.
    Returns details on referenced, valid, and nonexistent IDs.
    """
    valid_set = set(valid_source_ids)
    referenced = extract_citations_from_text(text)
    valid_ids = [sid for sid in referenced if sid in valid_set]
    invalid_ids = [sid for sid in referenced if sid not in valid_set]

    return {
        "referenced_ids": referenced,
        "valid_ids": valid_ids,
        "invalid_ids": invalid_ids,
        "has_citations": len(referenced) > 0,
        "is_valid": len(referenced) > 0 and len(invalid_ids) == 0,
    }


def is_excerpt_traceable(
    excerpt: str,
    source_ids: list[str],
    sources: list[dict],
) -> bool:
    """
    Check whether a quoted excerpt is genuinely traceable to the retrieved text
    of the specified sources. Prevents fabricated quotations.
    """
    if not excerpt or not excerpt.strip():
        return False

    def normalize(s: str) -> str:
        # Collapse whitespace and lower-case for robust matching
        return re.sub(r"\s+", " ", s).strip().lower()

    norm_excerpt = normalize(excerpt)
    if not norm_excerpt:
        return False

    sources_by_id = {s.get("id"): s for s in sources if s.get("id")}
    for sid in source_ids:
        source = sources_by_id.get(sid)
        if not source:
            continue
        norm_source_text = normalize(source.get("text", ""))
        if norm_excerpt in norm_source_text:
            return True

    return False


def parse_json_response(raw_text: str) -> dict | list:
    """
    Safely parse JSON responses from language models, stripping markdown
    code fences and handling leading/trailing text.
    """
    cleaned = raw_text.strip()

    # Remove Markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to isolate first { ... } or [ ... ]
        start_obj = cleaned.find("{")
        end_obj = cleaned.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            try:
                return json.loads(cleaned[start_obj : end_obj + 1])
            except json.JSONDecodeError:
                pass

        start_arr = cleaned.find("[")
        end_arr = cleaned.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            try:
                return json.loads(cleaned[start_arr : end_arr + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from model response: {raw_text[:200]}")


def verify_claim(
    claim: str,
    sources: list[dict],
    ask_fn: Callable[[str], str] = ask_mistral,
) -> dict:
    """
    Check whether the supplied sources support a factual claim.
    Validates citation references, checks source existence, detects fabricated
    quotations, and ensures evidence-based verdicts.
    """
    source_context = []
    sources_by_id = {}

    for source in sources:
        source_id = source.get("id", "Unknown")
        title = source.get("title", "Untitled")
        text = source.get("text", "")
        sources_by_id[source_id] = source

        source_context.append(f"[{source_id}] {title}\nEvidence: {text}")

    evidence = "\n\n".join(source_context)
    valid_ids = {s.get("id") for s in sources if s.get("id")}

    # Validate any inline citations present in the claim text itself
    citation_check = validate_citations(claim, valid_ids)
    cited_ids = citation_check["referenced_ids"]
    invalid_cited_ids = citation_check["invalid_ids"]

    prompt = f"""
You are a strict fact-checking assistant.

Your task is to check ONE factual claim against the supplied evidence.

CLAIM:
{claim}

AVAILABLE EVIDENCE:
{evidence}

Choose exactly one verdict:
- supported: the evidence directly supports the claim.
- partially_supported: some parts are supported, but others are missing, exaggerated, or unproven.
- unsupported: the evidence does not provide enough information to verify the claim.
- contradicted: the evidence directly conflicts with the claim.

Rules:
- Judge ONLY the supplied evidence, not your background knowledge.
- Do NOT treat a source title or citation tag alone as proof. An assertion is only supported if the source text explicitly substantiates it.
- If the claim cites specific source IDs, examine those sources first.
- If the claim is supported or partially supported, copy a brief verbatim quotation from the source text into "supporting_excerpt".
- The "supporting_excerpt" MUST be an exact quote directly from the provided source text. If no exact phrase substantiates the claim, leave it empty. NEVER invent or alter a quote.
- If evidence is insufficient, set verdict to "unsupported" and explain clearly what is missing.
- Identify only valid source IDs from the evidence list.

Return ONLY a valid JSON object in this exact format:
{{
  "verdict": "supported",
  "explanation": "Brief explanation based strictly on the supplied evidence.",
  "source_ids": ["S1"],
  "supporting_excerpt": "exact quote from source text"
}}
"""

    response = ask_fn(prompt)
    result = parse_json_response(response)

    if not isinstance(result, dict):
        raise ValueError("Verifier response did not produce a dictionary.")

    allowed_verdicts = {
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    }

    raw_verdict = result.get("verdict")
    if raw_verdict not in allowed_verdicts:
        raise ValueError(f"Verifier returned invalid verdict: '{raw_verdict}'")

    explanation = result.get("explanation")
    if not isinstance(explanation, str):
        raise ValueError("Verifier did not return a valid explanation string.")

    raw_source_ids = result.get("source_ids", [])
    if not isinstance(raw_source_ids, list):
        raw_source_ids = []

    # Filter to genuinely valid source IDs from the retrieved set
    matched_source_ids = [sid for sid in raw_source_ids if sid in valid_ids]

    # Handle supporting excerpt validation
    raw_excerpt = result.get("supporting_excerpt")
    excerpt = raw_excerpt.strip() if isinstance(raw_excerpt, str) else ""

    # Verify that the excerpt actually exists in the source text
    excerpt_traceable = is_excerpt_traceable(excerpt, matched_source_ids or list(valid_ids), sources)
    if excerpt and not excerpt_traceable:
        # Detected a fabricated quotation from the model
        excerpt_verified = False
        verified_excerpt = ""
    elif excerpt and excerpt_traceable:
        excerpt_verified = True
        verified_excerpt = excerpt
    else:
        excerpt_verified = False
        verified_excerpt = ""

    # Citation integrity checks
    verdict = raw_verdict
    final_explanation = explanation

    if invalid_cited_ids:
        # Nonexistent source IDs were cited in the claim
        final_explanation += f" Warning: Claim cited nonexistent source ID(s): {', '.join(invalid_cited_ids)}."
        if verdict == "supported" and not matched_source_ids:
            verdict = "unsupported"

    if verdict == "supported" and not matched_source_ids:
        # Cannot be supported if no valid sources provide evidence
        verdict = "unsupported"
        final_explanation += " Downgraded to unsupported because no valid source IDs provide evidence."

    return {
        "verdict": verdict,
        "explanation": final_explanation.strip(),
        "source_ids": matched_source_ids,
        "cited_source_ids": cited_ids,
        "invalid_citations": invalid_cited_ids,
        "is_citation_valid": citation_check["is_valid"] or (not citation_check["has_citations"]),
        "supporting_excerpt": verified_excerpt,
        "excerpt_verified": excerpt_verified,
    }


def verify_briefing(
    briefing: str,
    sources: list[dict],
    ask_fn: Callable[[str], str] = ask_mistral,
) -> list[dict]:
    """
    Extract factual claims from a briefing and verify each one against
    retrieved sources with citation validation.
    """
    prompt = f"""
You are a careful fact-checking assistant.

Extract up to 8 important, checkable factual claims from this briefing.
Choose claims that are specific, numerical, causal, or factual, rather than headings or opinions.
Preserve any inline citation tags like [S1] or [S2] associated with each claim.
Do not rewrite a claim to make it easier to verify.

BRIEFING:
{briefing}

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    "First factual claim [S1].",
    "Second factual claim [S2]."
  ]
}}
"""

    response = ask_fn(prompt)
    data = parse_json_response(response)

    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError("Claim extractor did not return a valid claims dictionary.")

    claims = data.get("claims", [])
    verified = []

    for item in claims[:8]:
        if not isinstance(item, str) or not item.strip():
            continue

        claim_text = item.strip()

        try:
            result = verify_claim(claim_text, sources, ask_fn=ask_fn)
            verified.append({
                "claim": claim_text,
                **result,
            })
        except Exception as error:
            # Detect any citations in the claim even if verification fails
            valid_ids = {s.get("id") for s in sources if s.get("id")}
            citation_check = validate_citations(claim_text, valid_ids)
            verified.append({
                "claim": claim_text,
                "verdict": "verification_error",
                "explanation": f"Verification error: {error}",
                "source_ids": [],
                "cited_source_ids": citation_check["referenced_ids"],
                "invalid_citations": citation_check["invalid_ids"],
                "is_citation_valid": False,
                "supporting_excerpt": "",
                "excerpt_verified": False,
            })

    return verified


def generate_verification_summary(verified_claims: list[dict]) -> dict:
    """
    Summarize verification outcomes, tallying verdicts, invalid citations,
    and generating actionable warning messages.
    """
    summary = {
        "total": len(verified_claims),
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0,
        "contradicted": 0,
        "verification_error": 0,
        "invalid_citations_count": 0,
        "warnings": [],
    }

    for item in verified_claims:
        verdict = item.get("verdict", "unknown")
        if verdict in summary:
            summary[verdict] += 1

        invalid_citations = item.get("invalid_citations", [])
        if invalid_citations:
            summary["invalid_citations_count"] += len(invalid_citations)
            summary["warnings"].append(
                f"Claim references nonexistent source ID(s) {', '.join(invalid_citations)}: \"{item.get('claim', '')}\""
            )

        if verdict in {"unsupported", "contradicted"}:
            summary["warnings"].append(
                f"{verdict.upper()} claim: \"{item.get('claim', '')}\" — {item.get('explanation', '')}"
            )
        elif verdict == "partially_supported":
            summary["warnings"].append(
                f"PARTIALLY SUPPORTED claim: \"{item.get('claim', '')}\" — {item.get('explanation', '')}"
            )
        elif verdict == "verification_error":
            summary["warnings"].append(
                f"VERIFICATION ERROR: \"{item.get('claim', '')}\" — {item.get('explanation', '')}"
            )

    return summary


if __name__ == "__main__":
    example_sources = [
        {
            "id": "S1",
            "title": "Example source",
            "text": (
                "CRISPR-Cas9 uses a guide RNA to direct the Cas9 "
                "protein to a matching DNA sequence, where Cas9 "
                "can cut the DNA."
            ),
        }
    ]

    example_claim = (
        "CRISPR-Cas9 uses guide RNA to direct Cas9 to a target DNA sequence [S1]."
    )

    try:
        res = verify_claim(example_claim, example_sources)
        print(json.dumps(res, indent=2))
    except Exception as exc:
        print(f"Verification failed: {exc}")
