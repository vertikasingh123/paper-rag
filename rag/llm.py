"""A thin provider layer over two free-tier LLM APIs.

Groq and Google AI Studio both give you a usable free tier without a credit
card. Model IDs get deprecated regularly, so `list_models()` asks the provider
what it currently serves rather than trusting a hardcoded string.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class LLMError(RuntimeError):
    pass


def detect_provider() -> str:
    """Explicit LLM_PROVIDER wins; otherwise use whichever key is present."""
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if forced:
        return forced
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    raise LLMError(
        "No API key found. Copy .env.example to .env and set GROQ_API_KEY "
        "(https://console.groq.com/keys) or GEMINI_API_KEY "
        "(https://aistudio.google.com/apikey). Both are free."
    )


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or detect_provider()
        if self.provider == "groq":
            self.model = model or DEFAULT_GROQ_MODEL
        elif self.provider == "gemini":
            self.model = model or DEFAULT_GEMINI_MODEL
        else:
            raise LLMError(f"Unknown provider {self.provider!r}. Use 'groq' or 'gemini'.")

    # -- completion --------------------------------------------------------

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        if self.provider == "groq":
            return self._groq(system, user, temperature)
        return self._gemini(system, user, temperature)

    def _groq(self, system: str, user: str, temperature: float) -> str:
        from openai import OpenAI  # Groq speaks the OpenAI wire format

        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _gemini(self, system: str, user: str, temperature: float) -> str:
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature
            ),
        )
        return (response.text or "").strip()

    # -- introspection -----------------------------------------------------

    def list_models(self) -> list[str]:
        """Ask the provider what it serves today. Useful when a model 404s."""
        if self.provider == "groq":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
            return sorted(m.id for m in client.models.list().data)
        from google import genai

        key = os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        client = genai.Client(api_key=key)
        return sorted(m.name for m in client.models.list())


if __name__ == "__main__":  # python -m rag.llm  -> print available models
    client = LLMClient()
    print(f"provider: {client.provider}\ndefault model: {client.model}\n")
    for name in client.list_models():
        print(" ", name)
