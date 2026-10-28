import unittest
from unittest.mock import MagicMock, patch

from backend.research_agent import ResearchResult


class TestStage3AppIntegration(unittest.TestCase):
    def test_app_imports_cleanly_without_api_key(self):
        """Confirm app.py imports without triggering a research run or requiring an API key."""
        with patch.dict("os.environ", {}, clear=True):
            import app
            self.assertTrue(hasattr(app, "main"))
            self.assertTrue(hasattr(app, "render_results"))
            self.assertTrue(hasattr(app, "format_verdict_badge"))

    def test_format_verdict_badge(self):
        """Confirm format_verdict_badge produces human-readable labels and colors."""
        import app
        badge, color = app.format_verdict_badge("supported")
        self.assertIn("Supported", badge)
        self.assertIn("🟢", badge)

        badge_p, _ = app.format_verdict_badge("partially_supported")
        self.assertIn("Partially Supported", badge_p)

        badge_u, _ = app.format_verdict_badge("unsupported")
        self.assertIn("Unsupported", badge_u)

        badge_c, _ = app.format_verdict_badge("contradicted")
        self.assertIn("Contradicted", badge_c)

        badge_e, _ = app.format_verdict_badge("verification_error")
        self.assertIn("Verification Error", badge_e)

    def test_render_results_with_valid_structured_result(self):
        """Confirm render_results processes all sections of a ResearchResult."""
        import app

        mock_result = ResearchResult({
            "topic": "Solar Energy",
            "questions": ["How do solar panels work?", "What are their limitations?"],
            "sources": [
                {
                    "id": "S1",
                    "title": "Solar Power",
                    "url": "https://en.wikipedia.org/wiki/Solar_power",
                    "text": "Solar power converts energy from sunlight into electricity.",
                }
            ],
            "briefing": "# Executive Summary\nSolar power converts sunlight into electricity [S1].",
            "claims": [
                {
                    "claim": "Solar power converts sunlight into electricity [S1].",
                    "verdict": "supported",
                    "explanation": "Supported by S1 text extract.",
                    "source_ids": ["S1"],
                    "cited_source_ids": ["S1"],
                    "invalid_citations": [],
                    "supporting_excerpt": "converts energy from sunlight into electricity",
                    "excerpt_verified": True,
                },
                {
                    "claim": "Commercial solar panels are 99% efficient [S1].",
                    "verdict": "unsupported",
                    "explanation": "No evidence for 99% efficiency in retrieved source.",
                    "source_ids": [],
                    "cited_source_ids": ["S1"],
                    "invalid_citations": [],
                    "supporting_excerpt": "",
                    "excerpt_verified": False,
                },
            ],
            "verification_summary": {
                "total": 2,
                "supported": 1,
                "partially_supported": 0,
                "unsupported": 1,
                "contradicted": 0,
                "verification_error": 0,
                "invalid_citations_count": 0,
                "warnings": ["UNSUPPORTED claim: Commercial solar panels are 99% efficient"],
            },
            "warnings": ["UNSUPPORTED claim: Commercial solar panels are 99% efficient"],
            "errors": [],
            "formatted_report": "Formatted report text.",
        })

        mock_col = MagicMock()
        with patch("streamlit.expander") as mock_expander, \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.columns", side_effect=lambda spec: [mock_col for _ in range(spec if isinstance(spec, int) else len(spec))]), \
             patch("streamlit.warning") as mock_warning:

            mock_expander.return_value.__enter__.return_value = MagicMock()
            app.render_results(mock_result)

            # Verify metrics were rendered for claim verdicts on column objects
            self.assertTrue(mock_col.metric.called)
            # Verify warning was shown for the unsupported claim
            self.assertTrue(mock_warning.called)

    def test_missing_api_key_banner(self):
        """Confirm missing API key displays actionable setup instructions without crashing."""
        import app

        mock_col = MagicMock()
        with patch("app.get_mistral_api_key", return_value=""), \
             patch("streamlit.columns", side_effect=lambda spec: [mock_col for _ in range(spec if isinstance(spec, int) else len(spec))]), \
             patch("streamlit.error") as mock_error:
            app.main()
            self.assertTrue(mock_error.called)
            # Check message instructions
            error_call_text = str(mock_error.call_args)
            self.assertIn("Mistral API Key Not Found", error_call_text)
            self.assertIn("MISTRAL_API_KEY", error_call_text)

    def test_empty_input_validation(self):
        """Confirm empty input does not call the research pipeline."""
        import app

        mock_col = MagicMock()
        with patch("app.get_mistral_api_key", return_value="dummy-key"), \
             patch("app.research_topic") as mock_research, \
             patch("streamlit.warning") as mock_warning, \
             patch("streamlit.session_state", {"topic_input": "", "trigger_research": True, "research_result": None, "pipeline_error": None}), \
             patch("streamlit.text_input", return_value="   "), \
             patch("streamlit.columns", side_effect=lambda spec: [mock_col for _ in range(spec if isinstance(spec, int) else len(spec))]):

            app.main()
            mock_research.assert_not_called()
            self.assertTrue(mock_warning.called)

    def test_invalid_citations_flagged_in_ui(self):
        """Confirm claims referencing non-existent source IDs are prominently flagged."""
        import app

        claims = [
            {
                "claim": "Claim with fake source [S9].",
                "verdict": "unsupported",
                "explanation": "No source found.",
                "source_ids": [],
                "cited_source_ids": ["S9"],
                "invalid_citations": ["S9"],
                "supporting_excerpt": "",
                "excerpt_verified": False,
            }
        ]
        summary = {
            "total": 1,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 1,
            "contradicted": 0,
            "verification_error": 0,
        }

        with patch("streamlit.expander") as mock_expander, \
             patch("streamlit.error") as mock_error:
            mock_expander.return_value.__enter__.return_value = MagicMock()
            app.render_verification_dashboard(claims, summary)
            # Verify error call was made for invalid citation tag S9
            self.assertTrue(mock_error.called)
            self.assertIn("S9", str(mock_error.call_args))

    def test_render_diagnostics_displays_stage_failure_safely(self):
        """Confirm render_diagnostics expands and safely shows failed stage and candidate count."""
        import app

        diagnostics = {
            "stage_failed": "relevance_filter",
            "details": "Evaluated 8 candidate article(s), but none met strict relevance criteria.",
            "questions_count": 4,
            "candidates_count": 8,
            "candidate_titles": ["LangChain", "DOT", "Java"],
        }

        mock_expander = MagicMock()
        mock_expander.return_value.__enter__.return_value = MagicMock()
        with patch("streamlit.expander", mock_expander), \
             patch("streamlit.markdown") as mock_markdown:
            app.render_diagnostics(diagnostics)
            self.assertTrue(mock_expander.called)
            expander_title = str(mock_expander.call_args)
            self.assertIn("relevance_filter", expander_title)
            # Confirm markdown received stage and counts
            markdown_calls = [str(call) for call in mock_markdown.call_args_list]
            self.assertTrue(any("relevance_filter" in c for c in markdown_calls))
            self.assertTrue(any("8" in c for c in markdown_calls))

    def test_render_results_with_diagnostics_on_early_failure(self):
        """Confirm render_results renders plan, diagnostics, and warnings/errors when research fails early."""
        import app

        failing_result = ResearchResult({
            "topic": "LangGraph",
            "questions": ["What is LangGraph?"],
            "sources": [],
            "briefing": "",
            "claims": [],
            "verification_summary": {},
            "warnings": [],
            "errors": ["Stage failed at Relevance Filtering: No sufficiently relevant sources were found."],
            "formatted_report": "Error report",
            "diagnostics": {
                "stage_failed": "relevance_filter",
                "details": "None of the candidates had evidence.",
                "questions_count": 1,
                "candidates_count": 2,
            },
        })

        mock_expander = MagicMock()
        mock_expander.return_value.__enter__.return_value = MagicMock()
        with patch("streamlit.expander", mock_expander), \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.error") as mock_error:
            app.render_results(failing_result)
            # Verify error was rendered
            self.assertTrue(mock_error.called)
            self.assertIn("Relevance Filtering", str(mock_error.call_args))
            # Verify diagnostics expander was rendered
            self.assertTrue(mock_expander.called)


if __name__ == "__main__":
    unittest.main()

