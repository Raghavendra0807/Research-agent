
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mistralai.client import Mistral

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_client_instance: Optional[Mistral] = None
_cached_key: Optional[str] = None


def get_mistral_api_key() -> str:
    """
    Resolve the Mistral API key from:
    1. Environment variables (MISTRAL_API_KEY)
    2. Local repository .env file
    3. Streamlit secrets (st.secrets['MISTRAL_API_KEY']) when available
    Never logs or exposes the actual key value.
    """
    # 1. Check existing environment variable
    key = os.getenv("MISTRAL_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Check local .env file if it exists
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
        key = os.getenv("MISTRAL_API_KEY")
        if key and key.strip():
            return key.strip()

    # 3. Check Streamlit secrets if Streamlit is imported or accessible
    try:
        # Check sys.modules or import without making Streamlit mandatory
        st = sys.modules.get("streamlit")
        if st is None:
            import streamlit as st  # type: ignore

        if hasattr(st, "secrets") and "MISTRAL_API_KEY" in st.secrets:
            secret_key = st.secrets["MISTRAL_API_KEY"]
            if secret_key and str(secret_key).strip():
                return str(secret_key).strip()
    except Exception:
        # Streamlit not installed or secrets not configured; continue gracefully
        pass

    return ""


def get_mistral_client(force_refresh: bool = False) -> Mistral:
    """
    Get or create the Mistral client instance.
    Raises RuntimeError with actionable instructions if no API key is configured.
    """
    global _client_instance, _cached_key

    api_key = get_mistral_api_key()
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not found. Please set the MISTRAL_API_KEY environment "
            "variable, add it to your local .env file, or configure "
            "st.secrets['MISTRAL_API_KEY'] for Streamlit Cloud."
        )

    if _client_instance is None or force_refresh or api_key != _cached_key:
        _client_instance = Mistral(api_key=api_key)
        _cached_key = api_key

    return _client_instance


def ask_mistral(prompt: str, client: Optional[Mistral] = None) -> str:
    """Send a prompt to the Mistral model and return the text response."""
    active_client = client if client is not None else get_mistral_client()

    response = active_client.chat.complete(
        model="open-mistral-nemo",
        messages=[
            {
                "role": "system",
                "content": "You are a clear, concise research assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content or ""


def __getattr__(name: str):
    """Provide backward compatibility for code accessing mistral_client.client."""
    if name == "client":
        return get_mistral_client()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    try:
        answer = ask_mistral(
            "Explain how CRISPR gene editing works in three simple sentences."
        )
        print("\nMistral response:\n")
        print(answer)
    except Exception as exc:
        print(f"\nMistral call failed: {exc}")

