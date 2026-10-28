import os
from typing import Optional

import streamlit as st

from backend.mistral_client import get_mistral_api_key
from backend.research_agent import ResearchResult, research_topic


def format_verdict_badge(verdict: str) -> tuple[str, str]:
    """
    Return human-readable badge text and styling color for a claim verdict.
    """
    v = verdict.lower() if verdict else "unknown"
    if v == "supported":
        return "🟢 Supported", "#d4edda"
    elif v == "partially_supported":
        return "🟡 Partially Supported", "#fff3cd"
    elif v == "unsupported":
        return "🔴 Unsupported", "#f8d7da"
    elif v == "contradicted":
        return "🚨 Contradicted", "#f5c6cb"
    elif v == "verification_error":
        return "⚠️ Verification Error", "#e2e3e5"
    else:
        return f"⚪ {verdict.title()}", "#f8f9fa"


def render_header() -> None:
    """Render title, overview, and workflow overview."""
    st.title("🔬 Research Briefing Agent + Verifier")
    st.markdown(
        """
        An evidence-grounded research assistant that plans searches, retrieves authoritative 
        Wikipedia evidence, synthesizes a cited research briefing, and verifies factual claims.
        """
    )

    # Visual Workflow Overview
    flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
    with flow_col1:
        st.info("**1. Plan**\n\nGenerate focused research questions")
    with flow_col2:
        st.info("**2. Retrieve**\n\nFetch & filter Wikipedia evidence")
    with flow_col3:
        st.info("**3. Synthesize**\n\nDraft briefing with inline citations")
    with flow_col4:
        st.info("**4. Verify**\n\nCross-check claims against evidence")

    st.caption(
        "ℹ️ **Note on Evidence & Verification**: Briefings are synthesized strictly from retrieved "
        "Wikipedia articles. Verifier verdicts are model-generated assessments against available evidence, "
        "not mathematical proofs or guarantees of absolute truth."
    )
    st.divider()


def render_research_plan(questions: list[str]) -> None:
    """Render the generated research questions."""
    if not questions:
        return

    with st.expander(f"📋 Research Plan ({len(questions)} Focused Questions)", expanded=False):
        for idx, q in enumerate(questions, start=1):
            st.markdown(f"**{idx}.** {q}")


def render_briefing(briefing: str) -> None:
    """Render the synthesized research briefing."""
    st.subheader("📝 Research Briefing")
    if briefing and briefing.strip():
        st.markdown(briefing)
    else:
        st.info("No research briefing was generated for this topic.")


def render_sources(sources: list[dict]) -> None:
    """Render retrieved and verified Wikipedia sources."""
    st.subheader(f"🌐 Retrieved Sources ({len(sources)})")
    st.markdown(
        "All factual claims and verifications are strictly anchored to the following retrieved Wikipedia sources:"
    )

    if not sources:
        st.info("No sources were retrieved.")
        return

    for source in sources:
        sid = source.get("id", "S?")
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        text = source.get("text", "")

        with st.expander(f"[{sid}] {title}", expanded=False):
            if url:
                st.markdown(f"🔗 **Desktop URL**: [{url}]({url})")
            if text:
                st.markdown("**Evidence Extract:**")
                st.write(text)
            else:
                st.caption("No text extract available for this source.")


def render_verification_dashboard(claims: list[dict], summary: dict) -> None:
    """Render claim verification metrics, alerts, and detailed breakdowns."""
    st.subheader("🛡️ Factual Claim Verification Dashboard")
    st.caption(
        "Each claim extracted from the briefing was checked against the retrieved evidence text. "
        "Citations alone do not guarantee factual accuracy."
    )

    total = summary.get("total", len(claims))
    supported = summary.get("supported", 0)
    partially = summary.get("partially_supported", 0)
    unsupported = summary.get("unsupported", 0)
    contradicted = summary.get("contradicted", 0)
    errors = summary.get("verification_error", 0)

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Claims Evaluated", total)
    col2.metric("Supported", supported)
    col3.metric("Partially Supported", partially)
    col4.metric("Unsupported", unsupported)
    col5.metric("Contradicted / Errors", contradicted + errors)

    # Prominent warning if problematic claims exist
    has_issues = (unsupported > 0) or (contradicted > 0) or (errors > 0)
    if has_issues:
        st.warning(
            f"⚠️ **Attention**: Found {unsupported} unsupported, {contradicted} contradicted, "
            f"or {errors} unverified claims. Inspect the highlighted claims below."
        )

    if not claims:
        st.info("Claim verification was not performed or returned no checkable claims.")
        return

    st.markdown("#### Detailed Claim Assessments")

    for idx, item in enumerate(claims, start=1):
        claim_text = item.get("claim", "")
        verdict = item.get("verdict", "unknown")
        explanation = item.get("explanation", "")
        matched_sources = item.get("source_ids", [])
        cited_sources = item.get("cited_source_ids", [])
        invalid_citations = item.get("invalid_citations", [])
        excerpt = item.get("supporting_excerpt", "")
        excerpt_verified = item.get("excerpt_verified", False)

        badge_text, _ = format_verdict_badge(verdict)

        # Expand by default if claim is unsupported or contradicted
        is_problematic = verdict in {"unsupported", "contradicted", "verification_error"}
        expander_title = f"{badge_text} — Claim {idx}: {claim_text[:80]}{'...' if len(claim_text) > 80 else ''}"

        with st.expander(expander_title, expanded=is_problematic):
            st.markdown(f"**Full Claim:** {claim_text}")
            st.markdown(f"**Verdict:** {badge_text}")
            st.markdown(f"**Verifier Explanation:** {explanation}")

            if matched_sources:
                st.markdown(f"**Evidence Sources:** {', '.join(f'`[{s}]`' for s in matched_sources)}")
            else:
                st.markdown("**Evidence Sources:** _None verified_")

            if invalid_citations:
                st.error(
                    f"⚠️ **Invalid Citation Tag(s)**: Referenced non-existent source ID(s): "
                    f"{', '.join(invalid_citations)}"
                )

            if excerpt and excerpt_verified:
                st.success(f"💬 **Traceable Excerpt from Source Text:**\n> \"{excerpt}\"")
            elif excerpt and not excerpt_verified:
                st.warning(
                    f"⚠️ **Unverified Quotation**: A quote was proposed by the model but could not "
                    f"be traced to the retrieved source text: \"{excerpt}\""
                )


def render_diagnostics(diagnostics: dict) -> None:
    """Render concise, safe pipeline stage diagnostic details without exposing secrets."""
    if not diagnostics or not diagnostics.get("stage_failed"):
        return

    stage_failed = diagnostics.get("stage_failed", "unknown")
    details = diagnostics.get("details", "")
    candidates_count = diagnostics.get("candidates_count")
    questions_count = diagnostics.get("questions_count")
    candidate_titles = diagnostics.get("candidate_titles", [])

    with st.expander(f"🔍 Pipeline Diagnostics (Stage Failed: {stage_failed})", expanded=True):
        st.markdown(f"**Failed Pipeline Stage:** `{stage_failed}`")
        if details:
            st.markdown(f"**Diagnostic Details:** {details}")
        if questions_count is not None:
            st.markdown(f"**Research Questions Formulated:** {questions_count}")
        if candidates_count is not None:
            st.markdown(f"**Wikipedia Articles Evaluated:** {candidates_count}")
        if candidate_titles:
            st.markdown(f"**Candidate Page Titles Checked:** {', '.join(candidate_titles[:6])}")


def render_warnings_and_errors(warnings: list[str], errors: list[str]) -> None:
    """Display any operational warnings or pipeline errors."""
    if errors:
        for err in errors:
            st.error(f"❌ {err}")

    if warnings:
        with st.expander(f"⚠️ Warnings & Integrity Notices ({len(warnings)})", expanded=False):
            for w in warnings:
                st.markdown(f"- {w}")


def render_results(result: ResearchResult) -> None:
    """Render complete structured research result."""
    errors = result.get("errors", [])
    sources = result.get("sources", [])
    briefing = result.get("briefing", "")
    questions = result.get("questions", [])
    claims = result.get("claims", [])
    summary = result.get("verification_summary", {})
    warnings = result.get("warnings", [])
    diagnostics = result.get("diagnostics", {})

    # If fatal error occurred during early stage with no content
    if errors and not sources and not briefing:
        if questions:
            render_research_plan(questions)
        render_diagnostics(diagnostics)
        render_warnings_and_errors(warnings, errors)
        return

    # Section A: Research Plan
    render_research_plan(questions)

    # Section B: Briefing
    render_briefing(briefing)
    st.divider()

    # Section C: Sources
    render_sources(sources)
    st.divider()

    # Section D: Verification
    render_verification_dashboard(claims, summary)

    # Section E: Warnings
    if warnings or errors:
        st.divider()
        render_warnings_and_errors(warnings, errors)


def run_pipeline(topic: str) -> None:
    """Execute research pipeline with live status progress."""
    st.session_state["research_result"] = None
    st.session_state["pipeline_error"] = None

    status_container = st.status("🔬 Conducting research...", expanded=True)

    def progress_callback(stage: str, message: str) -> None:
        stage_icons = {
            "planning": "📋",
            "planning_complete": "📋",
            "retrieval": "🌐",
            "filtering": "⚖️",
            "filtering_failed": "⚠️",
            "synthesis": "✍️",
            "verification": "🔍",
            "complete": "✅",
        }
        icon = stage_icons.get(stage, "🔹")
        status_container.write(f"{icon} {message}")

    try:
        result = research_topic(
            topic=topic,
            progress_callback=progress_callback,
        )
        st.session_state["research_result"] = result
        if result.get("errors"):
            failed_stage = result.get("diagnostics", {}).get("stage_failed", "pipeline")
            status_container.update(
                label=f"⚠️ Research stopped: Insufficient evidence at stage '{failed_stage}'",
                state="error",
                expanded=True,
            )
        else:
            status_container.update(label="✅ Research and claim verification complete!", state="complete", expanded=False)
    except Exception as exc:
        status_container.update(label="❌ Research encountered an error", state="error", expanded=True)
        st.session_state["pipeline_error"] = str(exc)


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Research Briefing Agent + Verifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    render_header()

    # Check API key configuration safely
    api_key = get_mistral_api_key()
    if not api_key:
        st.error(
            "⚠️ **Mistral API Key Not Found**\n\n"
            "To use the research assistant, please configure your Mistral API key:\n"
            "- **Local environment**: Add `MISTRAL_API_KEY=your_key_here` to the `.env` file in the project root.\n"
            "- **Environment variable**: Set `MISTRAL_API_KEY` in your system environment.\n"
            "- **Streamlit Cloud**: Add `MISTRAL_API_KEY = \"your_key_here\"` to Streamlit Secrets."
        )
        return

    # Initialize Session State
    if "topic_input" not in st.session_state:
        st.session_state["topic_input"] = ""
    if "research_result" not in st.session_state:
        st.session_state["research_result"] = None
    if "pipeline_error" not in st.session_state:
        st.session_state["pipeline_error"] = None

    # Suggested Topics
    st.markdown("**Suggested Research Topics:**")
    suggested_cols = st.columns(3)

    if suggested_cols[0].button("🧬 How does CRISPR gene editing work?", use_container_width=True):
        st.session_state["topic_input"] = "How does CRISPR gene editing work?"
        st.session_state["trigger_research"] = True

    if suggested_cols[1].button("🔭 How does the James Webb Space Telescope work?", use_container_width=True):
        st.session_state["topic_input"] = "How does the James Webb Space Telescope work?"
        st.session_state["trigger_research"] = True

    if suggested_cols[2].button("🔋 What are solid-state batteries?", use_container_width=True):
        st.session_state["topic_input"] = "What are solid-state batteries?"
        st.session_state["trigger_research"] = True

    # Topic Input Field
    topic = st.text_input(
        "Enter a topic to research:",
        value=st.session_state["topic_input"],
        placeholder="e.g. mRNA vaccines, Quantum computing error correction, Nuclear fusion reactors...",
        key="main_topic_field",
    )

    # Action Buttons
    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
    research_clicked = btn_col1.button("🚀 Research Topic", type="primary", use_container_width=True)
    reset_clicked = btn_col2.button("🔄 Reset / New Topic", use_container_width=True)

    if reset_clicked:
        st.session_state["topic_input"] = ""
        st.session_state["research_result"] = None
        st.session_state["pipeline_error"] = None
        st.session_state.pop("trigger_research", None)
        st.rerun()

    should_run = research_clicked or st.session_state.pop("trigger_research", False)

    if should_run:
        cleaned_topic = topic.strip()
        if not cleaned_topic:
            st.warning("⚠️ Please enter a research topic to begin.")
        else:
            st.session_state["topic_input"] = cleaned_topic
            run_pipeline(cleaned_topic)

    # Render pipeline errors if any occurred
    if st.session_state.get("pipeline_error"):
        st.error(f"❌ Research failed: {st.session_state['pipeline_error']}")

    # Render results if available
    if st.session_state.get("research_result"):
        st.divider()
        render_results(st.session_state["research_result"])


if __name__ == "__main__":
    main()
