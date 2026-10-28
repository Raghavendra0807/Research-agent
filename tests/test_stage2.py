import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

from backend.mistral_client import (
    ask_mistral,
    get_mistral_api_key,
    get_mistral_client,
)
from backend.planner import generate_research_questions
from backend.relevance_filter import filter_relevant_sources
from backend.research_agent import ResearchResult, research_topic
from backend.verifier import generate_verification_summary, verify_briefing
from backend.wikimedia_client import search_wikipedia


class TestMistralClientConfiguration(unittest.TestCase):
    def test_missing_api_key_raises_runtime_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.is_file", return_value=False):
                # When no key is configured anywhere
                with patch.dict(sys.modules, {"streamlit": None}):
                    key = get_mistral_api_key()
                    self.assertEqual(key, "")
                    with self.assertRaises(RuntimeError) as ctx:
                        get_mistral_client(force_refresh=True)
                    self.assertIn("MISTRAL_API_KEY not found", str(ctx.exception))

    def test_env_var_configuration(self):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-env-key-12345"}):
            key = get_mistral_api_key()
            self.assertEqual(key, "test-env-key-12345")

    def test_streamlit_secrets_configuration(self):
        mock_streamlit = MagicMock()
        mock_streamlit.secrets = {"MISTRAL_API_KEY": "streamlit-secret-key-999"}

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.is_file", return_value=False):
                with patch.dict(sys.modules, {"streamlit": mock_streamlit}):
                    key = get_mistral_api_key()
                    self.assertEqual(key, "streamlit-secret-key-999")

    def test_secrets_never_exposed_in_error(self):
        # Even if someone accidentally has whitespace or bad format, ensure standard errors don't echo sensitive tokens
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.is_file", return_value=False):
                try:
                    get_mistral_client(force_refresh=True)
                except RuntimeError as exc:
                    self.assertNotIn("Bearer", str(exc))
                    self.assertNotIn("sk-", str(exc))

    def test_backward_compatibility_getattr_client(self):
        import backend.mistral_client as mc
        with patch.object(mc, "get_mistral_client", return_value="mock_client"):
            self.assertEqual(mc.client, "mock_client")


class TestWikimediaResilience(unittest.TestCase):
    def test_empty_or_whitespace_topic(self):
        self.assertEqual(search_wikipedia(""), [])
        self.assertEqual(search_wikipedia("   "), [])
        self.assertEqual(search_wikipedia("topic", limit=0), [])

    def test_wikimedia_timeout_error(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with self.assertRaises(requests.exceptions.RequestException) as ctx:
            search_wikipedia("CRISPR", limit=2, session=mock_session)
        self.assertIn("Wikipedia search endpoint failed", str(ctx.exception))

    def test_wikimedia_connection_error(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Failed to establish connection")

        with self.assertRaises(requests.exceptions.RequestException) as ctx:
            search_wikipedia("CRISPR", limit=2, session=mock_session)
        self.assertIn("Wikipedia search endpoint failed", str(ctx.exception))

    def test_wikimedia_search_malformed_json(self):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON in response")
        mock_session.get.return_value = mock_response

        with self.assertRaises(requests.exceptions.RequestException):
            search_wikipedia("CRISPR", limit=2, session=mock_session)

    def test_single_article_404_does_not_abort_search(self):
        mock_session = MagicMock()

        # 1. Search response with 2 articles
        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {
            "pages": [
                {"title": "Broken Article"},
                {"title": "Working Article"},
            ]
        }

        # 2. Summary response for Broken Article (404)
        summary_resp_404 = MagicMock()
        summary_resp_404.status_code = 404

        # 3. Summary response for Working Article (200)
        summary_resp_200 = MagicMock()
        summary_resp_200.status_code = 200
        summary_resp_200.raise_for_status.return_value = None
        summary_resp_200.json.return_value = {
            "extract": "Working Article provides legitimate scientific evidence.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Working_Article"}},
        }

        mock_session.get.side_effect = [search_resp, summary_resp_404, summary_resp_200]

        results = search_wikipedia("test topic", limit=2, session=mock_session)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Working Article")
        self.assertIn("scientific evidence", results[0]["text"])

    def test_single_article_timeout_does_not_abort_search(self):
        mock_session = MagicMock()

        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {
            "pages": [
                {"title": "Timeout Article"},
                {"title": "Success Article"},
            ]
        }

        summary_resp_200 = MagicMock()
        summary_resp_200.status_code = 200
        summary_resp_200.raise_for_status.return_value = None
        summary_resp_200.json.return_value = {
            "extract": "Success Article extract text.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Success_Article"}},
        }

        mock_session.get.side_effect = [
            search_resp,
            requests.exceptions.Timeout("Timeout fetching summary"),
            summary_resp_200,
        ]

        results = search_wikipedia("test topic", limit=2, session=mock_session)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Success Article")

    def test_empty_extract_not_treated_as_evidence(self):
        mock_session = MagicMock()

        search_resp = MagicMock()
        search_resp.raise_for_status.return_value = None
        search_resp.json.return_value = {"pages": [{"title": "Empty Summary Page"}]}

        summary_resp = MagicMock()
        summary_resp.status_code = 200
        summary_resp.raise_for_status.return_value = None
        summary_resp.json.return_value = {
            "extract": "   ",  # Empty/whitespace summary
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Empty"}},
        }

        mock_session.get.side_effect = [search_resp, summary_resp]
        results = search_wikipedia("test topic", limit=1, session=mock_session)
        self.assertEqual(results, [])


class TestPipelineResilience(unittest.TestCase):
    def test_empty_and_whitespace_topic_handling(self):
        res1 = research_topic("")
        self.assertIn("Please provide a valid research topic.", res1["errors"])
        self.assertEqual(res1["claims"], [])

        res2 = research_topic("   \n\t  ")
        self.assertIn("Please provide a valid research topic.", res2["errors"])

    def test_planner_malformed_json_error_handling(self):
        def mock_broken_planner_ask(prompt: str) -> str:
            return "Sure! Here are some thoughts: 1. Question one 2. Question two."

        with self.assertRaises(ValueError) as ctx:
            generate_research_questions("Solar energy", count=4, ask_fn=mock_broken_planner_ask)
        self.assertIn("Could not parse the research plan", str(ctx.exception))

    def test_relevance_filter_malformed_json_error_handling(self):
        def mock_broken_filter_ask(prompt: str) -> str:
            return "I recommend looking at the first source."

        sample_sources = [{"id": "C1", "title": "Test", "text": "Sample text"}]
        with self.assertRaises(ValueError) as ctx:
            filter_relevant_sources("Solar", sample_sources, ask_fn=mock_broken_filter_ask)
        self.assertIn("Could not parse JSON", str(ctx.exception))

    def test_partial_retrieval_failure_continues(self):
        mock_questions = ["Q1: Success", "Q2: Failure", "Q3: Success"]

        def mock_ask(prompt: str) -> str:
            if "research planning assistant" in prompt:
                return json.dumps({"questions": mock_questions})
            elif "strict research retrieval evaluator" in prompt:
                return json.dumps({"selected_ids": ["C1"]})
            elif "strict fact-checking assistant" in prompt and "CLAIM:" in prompt:
                return json.dumps({
                    "verdict": "supported",
                    "explanation": "Verified from source.",
                    "source_ids": ["S1"],
                    "supporting_excerpt": "Valid evidence text.",
                })
            elif "Extract up to 8 important" in prompt:
                return json.dumps({"claims": ["Valid factual statement [S1]."]})
            else:
                return "# Executive Summary\nValid factual statement [S1]."

        def mock_search(query: str, limit: int = 2) -> list[dict]:
            if "Failure" in query:
                raise requests.exceptions.RequestException("Network glitch for Q2")
            return [{"title": f"Source for {query}", "url": "https://example.com", "text": "Valid evidence text."}]

        result = research_topic("Test topic", ask_fn=mock_ask, search_fn=mock_search)
        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(len(result["sources"]), 1)
        self.assertTrue(any("Network glitch for Q2" in w for w in result["warnings"]))
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["verdict"], "supported")

    def test_total_retrieval_failure_reported_clearly(self):
        def mock_ask(prompt: str) -> str:
            return json.dumps({"questions": ["Q1", "Q2"]})

        def mock_failing_search(query: str, limit: int = 2) -> list[dict]:
            raise requests.exceptions.RequestException("Wikipedia connection refused")

        result = research_topic("Failing Topic", ask_fn=mock_ask, search_fn=mock_failing_search)
        self.assertEqual(len(result["sources"]), 0)
        self.assertTrue(any("Could not retrieve usable source material" in err for err in result["errors"]))
        self.assertIn("Could not retrieve usable source material", result["formatted_report"])

    def test_verification_errors_remain_visible_in_result(self):
        # If verify_claim encounters an error on a specific claim
        briefing = "Claim A [S1]. Claim B [S1]."
        sources = [{"id": "S1", "title": "Source 1", "text": "Evidence text."}]

        call_count = 0
        def mock_ask(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if "Extract up to 8 important" in prompt:
                return json.dumps({"claims": ["Claim A [S1].", "Claim B [S1]."]})
            elif call_count == 2:
                # First claim verification succeeds
                return json.dumps({
                    "verdict": "supported",
                    "explanation": "Supported.",
                    "source_ids": ["S1"],
                    "supporting_excerpt": "Evidence text.",
                })
            else:
                # Second claim verification raises malformed JSON
                return "Internal server error"

        claims = verify_briefing(briefing, sources, ask_fn=mock_ask)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["verdict"], "supported")
        self.assertEqual(claims[1]["verdict"], "verification_error")
        self.assertIn("Verification error", claims[1]["explanation"])

        summary = generate_verification_summary(claims)
        self.assertEqual(summary["verification_error"], 1)
        self.assertTrue(any("VERIFICATION ERROR" in w for w in summary["warnings"]))

    def test_unsupported_claims_remain_visible_with_warnings(self):
        claims = [
            {
                "claim": "Unproven assertion [S1].",
                "verdict": "unsupported",
                "explanation": "No evidence found in text.",
                "source_ids": [],
                "invalid_citations": [],
            }
        ]
        summary = generate_verification_summary(claims)
        self.assertEqual(summary["unsupported"], 1)
        self.assertEqual(summary["supported"], 0)
        self.assertTrue(any("UNSUPPORTED claim" in w for w in summary["warnings"]))


if __name__ == "__main__":
    unittest.main()
