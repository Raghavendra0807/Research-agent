import json
import unittest
from unittest.mock import MagicMock, patch

from backend.research_agent import ResearchResult, format_report_text, research_topic
from backend.verifier import (
    extract_citations_from_text,
    generate_verification_summary,
    is_excerpt_traceable,
    parse_json_response,
    validate_citations,
    verify_briefing,
    verify_claim,
)


class TestCitationValidation(unittest.TestCase):
    def test_extract_citations_single_and_multiple(self):
        text = "Food rescue programs [S1] help reduce edible waste [S2, S3] and support families [s1]."
        citations = extract_citations_from_text(text)
        self.assertEqual(citations, ["S1", "S2", "S3"])

    def test_extract_citations_empty(self):
        self.assertEqual(extract_citations_from_text(""), [])
        self.assertEqual(extract_citations_from_text("No citations here."), [])

    def test_valid_citations(self):
        text = "This fact is substantiated [S1] and also [S2]."
        valid_ids = {"S1", "S2", "S3"}
        result = validate_citations(text, valid_ids)
        self.assertTrue(result["has_citations"])
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["valid_ids"], ["S1", "S2"])
        self.assertEqual(result["invalid_ids"], [])

    def test_missing_citations(self):
        text = "This statement has no citation tag."
        valid_ids = {"S1", "S2"}
        result = validate_citations(text, valid_ids)
        self.assertFalse(result["has_citations"])
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["referenced_ids"], [])
        self.assertEqual(result["invalid_ids"], [])

    def test_nonexistent_source_ids(self):
        text = "This claims something citing nonexistent sources [S1, S9, S10]."
        valid_ids = {"S1", "S2"}
        result = validate_citations(text, valid_ids)
        self.assertTrue(result["has_citations"])
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["valid_ids"], ["S1"])
        self.assertEqual(result["invalid_ids"], ["S9", "S10"])


class TestExcerptTraceability(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "id": "S1",
                "title": "CRISPR",
                "text": "CRISPR-Cas9 uses a guide RNA to direct the Cas9 enzyme to a specific target DNA sequence.",
            },
            {
                "id": "S2",
                "title": "DNA Repair",
                "text": "Non-homologous end joining is a pathway that repairs double-strand breaks in DNA.",
            },
        ]

    def test_exact_excerpt_found(self):
        quote = "guide RNA to direct the Cas9 enzyme"
        self.assertTrue(is_excerpt_traceable(quote, ["S1"], self.sources))

    def test_normalized_whitespace_match(self):
        quote = "guide   RNA to direct \n the Cas9 enzyme"
        self.assertTrue(is_excerpt_traceable(quote, ["S1"], self.sources))

    def test_fabricated_excerpt_rejected(self):
        fake_quote = "Cas9 was discovered in 1982 by Oxford scientists."
        self.assertFalse(is_excerpt_traceable(fake_quote, ["S1"], self.sources))

    def test_empty_or_whitespace_excerpt(self):
        self.assertFalse(is_excerpt_traceable("", ["S1"], self.sources))
        self.assertFalse(is_excerpt_traceable("   ", ["S1"], self.sources))


class TestJsonParsing(unittest.TestCase):
    def test_clean_json(self):
        data = parse_json_response('{"verdict": "supported", "explanation": "ok"}')
        self.assertEqual(data["verdict"], "supported")

    def test_markdown_code_fences(self):
        raw = '```json\n{"verdict": "supported", "explanation": "ok"}\n```'
        data = parse_json_response(raw)
        self.assertEqual(data["verdict"], "supported")

    def test_embedded_json_with_preamble(self):
        raw = 'Here is the result:\n{"verdict": "unsupported", "explanation": "missing"}\nHope this helps!'
        data = parse_json_response(raw)
        self.assertEqual(data["verdict"], "unsupported")

    def test_malformed_json_raises(self):
        with self.assertRaises(ValueError):
            parse_json_response("This is not JSON at all: { broken ")


class TestClaimVerification(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "id": "S1",
                "title": "Solar Energy",
                "text": "Photovoltaic cells convert sunlight directly into electricity using the photoelectric effect.",
            }
        ]

    def test_supported_claim_with_valid_traceable_excerpt(self):
        claim = "Photovoltaic cells convert sunlight into electricity [S1]."

        def mock_ask(prompt: str) -> str:
            return json.dumps({
                "verdict": "supported",
                "explanation": "The text states PV cells convert sunlight directly into electricity.",
                "source_ids": ["S1"],
                "supporting_excerpt": "convert sunlight directly into electricity",
            })

        result = verify_claim(claim, self.sources, ask_fn=mock_ask)
        self.assertEqual(result["verdict"], "supported")
        self.assertTrue(result["excerpt_verified"])
        self.assertEqual(result["supporting_excerpt"], "convert sunlight directly into electricity")
        self.assertEqual(result["source_ids"], ["S1"])
        self.assertEqual(result["invalid_citations"], [])

    def test_supported_claim_with_fabricated_excerpt_rejected(self):
        claim = "Photovoltaic cells convert sunlight into electricity [S1]."

        def mock_ask(prompt: str) -> str:
            return json.dumps({
                "verdict": "supported",
                "explanation": "Supported by the source.",
                "source_ids": ["S1"],
                "supporting_excerpt": "This exact phrase never occurs anywhere in the source text at all.",
            })

        result = verify_claim(claim, self.sources, ask_fn=mock_ask)
        self.assertEqual(result["verdict"], "supported")
        self.assertFalse(result["excerpt_verified"])
        self.assertEqual(result["supporting_excerpt"], "")

    def test_unsupported_claim(self):
        claim = "Photovoltaic cells achieve 95% efficiency in standard commercial panels [S1]."

        def mock_ask(prompt: str) -> str:
            return json.dumps({
                "verdict": "unsupported",
                "explanation": "The source makes no claim regarding 95% commercial panel efficiency.",
                "source_ids": [],
                "supporting_excerpt": "",
            })

        result = verify_claim(claim, self.sources, ask_fn=mock_ask)
        self.assertEqual(result["verdict"], "unsupported")
        self.assertEqual(result["source_ids"], [])
        self.assertFalse(result["excerpt_verified"])

    def test_claim_citing_nonexistent_source_id_downgraded(self):
        # Claim cites S9, which does not exist in self.sources
        claim = "Solar power was invented in ancient Rome [S9]."

        def mock_ask(prompt: str) -> str:
            return json.dumps({
                "verdict": "supported",
                "explanation": "Claim accepted based on citation.",
                "source_ids": ["S9"],  # Invalid source ID
                "supporting_excerpt": "",
            })

        result = verify_claim(claim, self.sources, ask_fn=mock_ask)
        self.assertEqual(result["invalid_citations"], ["S9"])
        # Should be downgraded to unsupported because no valid source exists
        self.assertEqual(result["verdict"], "unsupported")
        self.assertIn("nonexistent source ID(s)", result["explanation"])

    def test_malformed_model_response_handling_in_verify_briefing(self):
        def mock_broken_ask(prompt: str) -> str:
            return "Internal Server Error: 500 Model overloaded"

        briefing = "Solar energy is renewable [S1]."
        # Should gracefully fail or return empty rather than crash
        with self.assertRaises(ValueError):
            verify_briefing(briefing, self.sources, ask_fn=mock_broken_ask)


class TestVerificationSummary(unittest.TestCase):
    def test_summary_tallies_and_warnings(self):
        claims = [
            {
                "claim": "Claim 1 [S1]",
                "verdict": "supported",
                "explanation": "Verified.",
                "invalid_citations": [],
            },
            {
                "claim": "Claim 2 [S9]",
                "verdict": "unsupported",
                "explanation": "No source found.",
                "invalid_citations": ["S9"],
            },
            {
                "claim": "Claim 3 [S1]",
                "verdict": "partially_supported",
                "explanation": "Overstated.",
                "invalid_citations": [],
            },
        ]
        summary = generate_verification_summary(claims)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["supported"], 1)
        self.assertEqual(summary["unsupported"], 1)
        self.assertEqual(summary["partially_supported"], 1)
        self.assertEqual(summary["invalid_citations_count"], 1)
        self.assertGreater(len(summary["warnings"]), 0)


class TestStructuredResearchResult(unittest.TestCase):
    def test_empty_topic_returns_structured_error(self):
        result = research_topic("   ")
        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(result["topic"], "")
        self.assertIn("valid research topic", result["errors"][0])
        self.assertIn("valid research topic", str(result))

    def test_pipeline_structured_result_with_mocks(self):
        mock_questions = ["What is solar?", "How efficient is it?"]
        mock_sources = [
            {"id": "S1", "title": "Solar Energy", "url": "https://example.com/solar", "text": "Solar energy is light and heat."}
        ]

        def mock_ask(prompt: str) -> str:
            if "research planning assistant" in prompt:
                return json.dumps({"questions": mock_questions})
            elif "strict fact-checking assistant" in prompt and "CLAIM:" in prompt:
                return json.dumps({
                    "verdict": "supported",
                    "explanation": "Supported by S1 text.",
                    "source_ids": ["S1"],
                    "supporting_excerpt": "Solar energy is light and heat.",
                })
            elif "Extract up to 8 important" in prompt:
                return json.dumps({
                    "claims": ["Solar energy is light and heat from the Sun [S1]."]
                })
            else:
                # Briefing synthesis
                return "# Executive Summary\nSolar energy is light and heat from the Sun [S1]."

        def mock_search(query: str, limit: int = 2) -> list[dict]:
            return [
                {
                    "title": "Solar Energy",
                    "url": "https://example.com/solar",
                    "text": "Solar energy is light and heat from the Sun.",
                }
            ]

        progress_events = []
        def track_progress(stage: str, msg: str):
            progress_events.append(stage)

        with patch("backend.research_agent.generate_research_questions", return_value=mock_questions):
            with patch("backend.research_agent.filter_relevant_sources", return_value=mock_sources):
                result = research_topic(
                    topic="Solar Energy",
                    progress_callback=track_progress,
                    ask_fn=mock_ask,
                    search_fn=mock_search,
                )

        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(result["topic"], "Solar Energy")
        self.assertEqual(result["questions"], mock_questions)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("Executive Summary", result["briefing"])
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertTrue(result["claims"][0]["excerpt_verified"])
        self.assertIn("planning", progress_events)
        self.assertIn("complete", progress_events)

        # Verify CLI compatibility
        report_str = str(result)
        self.assertIn("Solar Energy", report_str)
        self.assertIn("--- Retrieved sources ---", report_str)
        self.assertIn("--- Claim verification ---", report_str)


if __name__ == "__main__":
    unittest.main()
