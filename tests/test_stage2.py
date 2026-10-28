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
from backend.wikimedia_client import (
    WikimediaRateLimitError,
    clear_summary_cache,
    derive_search_query,
    search_wikipedia,
)


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

    def test_http_429_followed_by_successful_retry(self):
        """Wikipedia returns HTTP 429 once, then succeeds on retry with backoff."""
        mock_session = MagicMock()
        mock_sleep = MagicMock()

        # 1. 429 response
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}

        # 2. 200 response for search
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status.return_value = None
        resp_200.json.return_value = {"pages": [{"title": "Retried Article"}]}

        # 3. 200 response for summary
        summary_resp = MagicMock()
        summary_resp.status_code = 200
        summary_resp.raise_for_status.return_value = None
        summary_resp.json.return_value = {
            "extract": "Evidence from article fetched after 429 backoff.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Retried"}},
        }

        mock_session.get.side_effect = [resp_429, resp_200, summary_resp]

        results = search_wikipedia(
            "LangGraph",
            limit=1,
            session=mock_session,
            max_retries=2,
            sleep_fn=mock_sleep,
            pacing_delay=0.0,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Retried Article")
        # Assert retry sleep occurred
        self.assertTrue(mock_sleep.called)

    def test_http_429_persisting_after_retry_limit(self):
        """Wikipedia continually returns HTTP 429, raising WikimediaRateLimitError after retries exhausted."""
        mock_session = MagicMock()
        mock_sleep = MagicMock()

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}
        mock_session.get.return_value = resp_429

        with self.assertRaises(WikimediaRateLimitError) as ctx:
            search_wikipedia(
                "LangGraph",
                limit=1,
                session=mock_session,
                max_retries=2,
                sleep_fn=mock_sleep,
                pacing_delay=0.0,
            )

        self.assertIn("rate limited (HTTP 429)", str(ctx.exception))
        # Verify bounded retry count: exactly 2 retry sleeps
        self.assertEqual(mock_sleep.call_count, 2)

    def test_respect_retry_after_header_without_real_delays(self):
        """HTTP 429 with 'Retry-After' header delays exactly by the specified seconds without stalling tests."""
        mock_session = MagicMock()
        mock_sleep = MagicMock()

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "3.5"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status.return_value = None
        resp_200.json.return_value = {"pages": [{"title": "Article"}]}

        summary_resp = MagicMock()
        summary_resp.status_code = 200
        summary_resp.raise_for_status.return_value = None
        summary_resp.json.return_value = {
            "extract": "Content.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Article"}},
        }

        mock_session.get.side_effect = [resp_429, resp_200, summary_resp]

        search_wikipedia(
            "LangGraph",
            limit=1,
            session=mock_session,
            max_retries=1,
            sleep_fn=mock_sleep,
            pacing_delay=0.0,
        )

        mock_sleep.assert_called_with(3.5)

    def test_derive_search_query_concise_langgraph(self):
        """Conversational research questions for LangGraph are condensed into concise, topic-focused queries."""
        q1 = "What is LangGraph and how does it define its primary purpose in agentic workflows?"
        q2 = "How does LangGraph's graph-based workflow model use nodes, edges, and state?"
        q3 = "In what ways does LangGraph coordinate LLM-powered agents and tools?"
        q4 = "What are the documented capabilities and limitations of LangGraph in production?"

        query1 = derive_search_query(q1, "LangGraph")
        query2 = derive_search_query(q2, "LangGraph")
        query3 = derive_search_query(q3, "LangGraph")
        query4 = derive_search_query(q4, "LangGraph")

        for q, query in [(q1, query1), (q2, query2), (q3, query3), (q4, query4)]:
            self.assertIn("LangGraph", query)
            self.assertLessEqual(len(query.split()), 6)
            self.assertFalse(query.lower().startswith("what is"))
            self.assertFalse(query.lower().startswith("how does"))


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


class TestPipelineDiagnosticsAndEvidence(unittest.TestCase):
    """
    Mocked test suite verifying diagnostic reporting and evidence integrity across:
    1. A topic with relevant Wikipedia articles.
    2. A topic with genuinely insufficient evidence on Wikipedia (e.g., LangGraph).
    3. Malformed and empty search results.
    """

    def test_topic_with_relevant_wikipedia_articles(self):
        """Pipeline successfully completes with rich diagnostics when evidence is present."""
        def mock_ask(prompt: str) -> str:
            if "Generate exactly" in prompt:
                return json.dumps({
                    "questions": [
                        "How does CRISPR-Cas9 work?",
                        "What are applications of CRISPR gene editing?",
                    ]
                })
            elif "Select up to" in prompt:
                return json.dumps({"selected_ids": ["C1"]})
            elif "Extract up to 8 important" in prompt:
                return json.dumps({"claims": ["CRISPR-Cas9 enables targeted genome editing [S1]."]})
            elif "fact-checking assistant" in prompt:
                return json.dumps({
                    "verdict": "supported",
                    "explanation": "Extract explicitly confirms genome editing mechanism.",
                    "source_ids": ["S1"],
                    "supporting_excerpt": "targeted genome editing technology",
                })
            else:
                return "# Executive Summary\nCRISPR-Cas9 enables targeted genome editing [S1]."

        def mock_search(query: str, limit: int = 2) -> list[dict]:
            return [
                {
                    "title": "CRISPR gene editing",
                    "url": "https://en.wikipedia.org/wiki/CRISPR_gene_editing",
                    "text": "CRISPR-Cas9 is a targeted genome editing technology derived from bacterial immune systems.",
                }
            ]

        result = research_topic("CRISPR gene editing", ask_fn=mock_ask, search_fn=mock_search)

        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(len(result["errors"]), 0)
        self.assertIsNone(result["diagnostics"]["stage_failed"])
        self.assertEqual(result["diagnostics"]["questions_count"], 2)
        self.assertEqual(result["diagnostics"]["relevant_count"], 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("CRISPR-Cas9", result["briefing"])
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["verdict"], "supported")

    def test_topic_with_genuinely_insufficient_evidence(self):
        """
        When Wikipedia lacks coverage/evidence (e.g. LangGraph), the pipeline must:
        - Preserve evidence-based filtering: reject irrelevant sources (do NOT accept them just to succeed).
        - Not fabricate evidence or briefings.
        - Accurately diagnose failure at the relevance_filter stage.
        """
        def mock_ask(prompt: str) -> str:
            if "Generate exactly" in prompt:
                return json.dumps({
                    "questions": [
                        "What is LangGraph and how does it manage state?",
                        "What are the limitations of LangGraph in production?",
                    ]
                })
            elif "Select up to" in prompt:
                # The strict relevance evaluator correctly rejects candidates lacking evidence
                return json.dumps({"selected_ids": []})
            else:
                raise AssertionError(f"Unexpected ask call after filter failure: {prompt[:100]}")

        def mock_search(query: str, limit: int = 2) -> list[dict]:
            # Wikipedia search returns only tangential or unrelated pages without LangGraph evidence
            return [
                {
                    "title": "DOT (graph description language)",
                    "url": "https://en.wikipedia.org/wiki/DOT_(graph_description_language)",
                    "text": "DOT is a plain text graph description language developed for Graphviz.",
                },
                {
                    "title": "Java (programming language)",
                    "url": "https://en.wikipedia.org/wiki/Java_(programming_language)",
                    "text": "Java is a class-based, object-oriented computer programming language.",
                },
            ]

        result = research_topic("LangGraph", ask_fn=mock_ask, search_fn=mock_search)

        self.assertIsInstance(result, ResearchResult)
        # Evidence-based filtering preserved: no irrelevant sources admitted
        self.assertEqual(len(result["sources"]), 0)
        self.assertEqual(result["briefing"], "")
        self.assertEqual(len(result["claims"]), 0)

        # Stage failure accurately diagnosed
        self.assertEqual(result["diagnostics"]["stage_failed"], "relevance_filter")
        self.assertEqual(result["diagnostics"]["questions_count"], 2)
        self.assertGreaterEqual(result["diagnostics"]["candidates_count"], 2)
        self.assertEqual(result["diagnostics"]["relevant_count"], 0)
        self.assertIn("DOT (graph description language)", result["diagnostics"]["candidate_titles"])

        # Error message clearly identifies failure stage and provides diagnostic explanation
        self.assertTrue(any("Relevance Filtering" in err for err in result["errors"]))
        self.assertTrue(any("No sufficiently relevant sources were found" in err for err in result["errors"]))
        self.assertIn("Relevance Filtering", result["formatted_report"])

    def test_malformed_and_empty_search_results(self):
        """Pipeline safely handles empty or malformed search payloads without uncaught exceptions."""
        def mock_ask(prompt: str) -> str:
            return json.dumps({"questions": ["Q1", "Q2"]})

        # Test case A: Search returns empty list
        def mock_empty_search(query: str, limit: int = 2) -> list[dict]:
            return []

        result_empty = research_topic("Empty Topic", ask_fn=mock_ask, search_fn=mock_empty_search)
        self.assertEqual(len(result_empty["sources"]), 0)
        self.assertEqual(result_empty["diagnostics"]["stage_failed"], "retrieval")
        self.assertTrue(any("Could not retrieve usable source material" in err for err in result_empty["errors"]))

        # Test case B: Search returns malformed elements (non-dict, empty text/title, bad types)
        def mock_malformed_search(query: str, limit: int = 2) -> list:
            return [
                None,
                "unexpected-string",
                12345,
                {"title": "", "text": "missing title"},
                {"title": "No text", "text": "   "},
                {"bad_keys": True},
            ]

        result_malformed = research_topic("Malformed Topic", ask_fn=mock_ask, search_fn=mock_malformed_search)
        self.assertEqual(len(result_malformed["sources"]), 0)
        self.assertEqual(result_malformed["diagnostics"]["stage_failed"], "retrieval")
        self.assertTrue(any("Could not retrieve usable source material" in err for err in result_malformed["errors"]))

    def test_partial_search_failure_preserves_successful_results(self):
        """When one search query encounters rate-limiting or errors, successful results from other queries are preserved."""
        def mock_ask(prompt: str) -> str:
            if "Generate exactly" in prompt:
                return json.dumps({"questions": ["Q1: Success", "Q2: RateLimit", "Q3: Success"]})
            elif "Select up to" in prompt:
                return json.dumps({"selected_ids": ["C1", "C2"]})
            elif "Extract up to 8 important" in prompt:
                return json.dumps({"claims": ["Fact supported by surviving sources [S1]."]})
            elif "fact-checking assistant" in prompt:
                return json.dumps({
                    "verdict": "supported",
                    "explanation": "Supported by text.",
                    "source_ids": ["S1"],
                    "supporting_excerpt": "Valid evidence text",
                })
            else:
                return "# Executive Summary\nFact supported by surviving sources [S1]."

        def mock_mixed_search(query: str, limit: int = 2) -> list[dict]:
            if "RateLimit" in query:
                raise WikimediaRateLimitError("Throttled query Q2")
            return [
                {
                    "title": f"Source for {query}",
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "text": f"Valid evidence text from {query}.",
                }
            ]

        result = research_topic("Resilient Topic", ask_fn=mock_ask, search_fn=mock_mixed_search)

        self.assertIsInstance(result, ResearchResult)
        # Verify that successful results from Q1 and Q3 were preserved and synthesized
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(len(result["errors"]), 0)
        self.assertIsNone(result["diagnostics"]["stage_failed"])
        self.assertEqual(result["diagnostics"]["reason"], "success")
        self.assertIn("Fact supported by surviving sources", result["briefing"])
        # Verify warning was recorded for the rate-limited query
        self.assertTrue(any("rate limit" in w.lower() for w in result["warnings"]))

    def test_persistent_rate_limiting_diagnosed_accurately(self):
        """When all searches fail due to rate-limiting, diagnose rate_limited instead of claiming lack of coverage."""
        def mock_ask(prompt: str) -> str:
            return json.dumps({"questions": ["Q1", "Q2"]})

        def mock_rate_limited_search(query: str, limit: int = 2) -> list[dict]:
            raise WikimediaRateLimitError("429 Client Error: Too Many Requests")

        result = research_topic("Rate Limited Topic", ask_fn=mock_ask, search_fn=mock_rate_limited_search)

        self.assertEqual(len(result["sources"]), 0)
        self.assertEqual(result["diagnostics"]["stage_failed"], "rate_limited")
        self.assertEqual(result["diagnostics"]["reason"], "rate_limited")
        # Informative error message distinguishing rate-limiting
        self.assertTrue(any("rate limit reached (HTTP 429)" in err for err in result["errors"]))
        # Never misreport rate limiting as lack of Wikipedia content
        self.assertFalse(any("Wikipedia may lack an established article" in err for err in result["errors"]))

    def test_no_results_distinguished_from_relevance_rejection(self):
        """Distinguish zero search results from candidate articles being rejected by relevance filtering."""
        def mock_ask(prompt: str) -> str:
            return json.dumps({"questions": ["Q1", "Q2"]})

        # Search succeeds (HTTP 200) but finds 0 articles
        def mock_empty_search(query: str, limit: int = 2) -> list[dict]:
            return []

        result = research_topic("Obscure Query", ask_fn=mock_ask, search_fn=mock_empty_search)

        self.assertEqual(len(result["sources"]), 0)
        self.assertEqual(result["diagnostics"]["stage_failed"], "retrieval")
        self.assertEqual(result["diagnostics"]["reason"], "no_results")
        self.assertTrue(any("No matching Wikipedia articles were found" in err for err in result["errors"]))


if __name__ == "__main__":
    unittest.main()


