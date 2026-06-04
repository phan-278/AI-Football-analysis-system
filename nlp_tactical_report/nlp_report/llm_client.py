"""
nlp_report/llm_client.py
────────────────────────
Gọi Claude API (mặc định) hoặc OpenAI
→ Nhận JSON báo cáo chiến thuật đã parse

Usage:
    client = LLMClient(api_key="sk-ant-...")
    report = client.generate(system_prompt, user_prompt)
    # hoặc shortcut:
    report = client.generate_from_stats(match_stats)
"""

import json
import os
import time
from typing import Any


class LLMClient:
    """
    Wrapper gọi LLM và parse kết quả JSON chiến thuật.

    Parameters
    ----------
    api_key  : str | None  — API key. None → đọc từ env ANTHROPIC_API_KEY / OPENAI_API_KEY
    provider : str         — "claude" (mặc định) hoặc "openai"
    model    : str | None  — override model mặc định
    max_retries : int      — số lần retry khi lỗi (default 3)
    """

    DEFAULT_MODELS = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
    }

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "claude",
        model: str | None = None,
        max_retries: int = 3,
    ):
        self.provider    = provider.lower()
        self.model       = model or self.DEFAULT_MODELS[self.provider]
        self.max_retries = max_retries

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "claude":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")

        if not self.api_key:
            env_var = "ANTHROPIC_API_KEY" if self.provider == "claude" else "OPENAI_API_KEY"
            raise ValueError(
                f"Thiếu API key. Truyền api_key= hoặc set biến môi trường {env_var}."
            )

    # ──────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2500,
    ) -> dict[str, Any]:
        """
        Gọi LLM và trả về report dict đã parse JSON.

        Returns dict với keys:
            match_overview, team1_analysis, team2_analysis,
            comparison, key_players, conclusion
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._call_api(system_prompt, user_prompt, max_tokens)
                return self._parse_response(raw)
            except (json.JSONDecodeError, KeyError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[LLMClient] Parse lỗi lần {attempt}: {e}. Retry sau {wait}s...")
                    time.sleep(wait)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[LLMClient] API lỗi lần {attempt}: {e}. Retry sau {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"LLMClient thất bại sau {self.max_retries} lần thử. "
            f"Lỗi cuối: {last_error}"
        ) from last_error

    def generate_from_stats(
        self,
        match_stats: dict[str, Any],
        max_tokens: int = 2500,
    ) -> dict[str, Any]:
        """Shortcut: nhận match_stats → tự build prompt → generate."""
        from .prompt_builder import PromptBuilder
        builder = PromptBuilder(match_stats)
        system_prompt, user_prompt = builder.build()
        return self.generate(system_prompt, user_prompt, max_tokens)

    # ──────────────────────────────────────────────
    # Private — API calls
    # ──────────────────────────────────────────────

    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if self.provider == "claude":
            return self._call_claude(system_prompt, user_prompt, max_tokens)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, max_tokens)
        else:
            raise ValueError(f"Provider không hỗ trợ: '{self.provider}'")

    def _call_claude(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Cần cài: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def _call_openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("Cần cài: pip install openai")

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    # ──────────────────────────────────────────────
    # Private — Parse
    # ──────────────────────────────────────────────

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Strip markdown fences nếu có → parse JSON → validate keys."""
        text = raw.strip()

        # Bỏ ```json ... ``` nếu LLM vẫn thêm
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        data = json.loads(text)

        required = {
            "match_overview", "team1_analysis", "team2_analysis",
            "comparison", "conclusion",
        }
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Response thiếu các key: {missing}")

        return data
