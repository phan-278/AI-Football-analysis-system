import json
import os
import time
from typing import Any


class LLMClient:
    DEFAULT_MODELS = {
        "claude": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "groq": "llama-3.3-70b-versatile",
    }

    # Model fallback khi model mặc định bị 503
    FALLBACK_MODELS = {
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "gemini",
        model: str | None = None,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()

        if self.provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Provider không hỗ trợ: '{self.provider}'. "
                f"Supported: {list(self.DEFAULT_MODELS.keys())}"
            )

        self.model = model or self.DEFAULT_MODELS[self.provider]
        self.max_retries = max_retries

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif self.provider == "claude":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        elif self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        elif self.provider == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
        elif self.provider == "groq":
            self.api_key = os.environ.get("GROQ_API_KEY", "")
        else:
            self.api_key = ""

        if not self.api_key:
            if self.provider == "claude":
                env_var = "ANTHROPIC_API_KEY"
            elif self.provider == "openai":
                env_var = "OPENAI_API_KEY"
            elif self.provider == "gemini":
                env_var = "GEMINI_API_KEY"
            elif self.provider == "groq":
                env_var = "GROQ_API_KEY"
            else:
                env_var = "API_KEY"

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
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        last_error = None
        # Danh sách model sẽ thử: model hiện tại + fallback (nếu có)
        fallbacks = self.FALLBACK_MODELS.get(self.provider, [])
        models_to_try = [self.model] + fallbacks

        for model_candidate in models_to_try:
            if model_candidate != self.model:
                print(f"[LLMClient] Thử fallback model: {model_candidate}...")
            original_model = self.model
            self.model = model_candidate

            for attempt in range(1, self.max_retries + 1):
                try:
                    raw = self._call_api(
                        system_prompt,
                        user_prompt,
                        max_tokens,
                    )
                    return self._parse_response(raw)

                except (json.JSONDecodeError, KeyError) as e:
                    last_error = e
                    if attempt < self.max_retries:
                        wait = 2 ** attempt
                        print(
                            f"[LLMClient] Parse lỗi lần {attempt}: "
                            f"{e}. Retry sau {wait}s..."
                        )
                        time.sleep(wait)

                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # 503 / overload → thử fallback model ngay
                    if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                        print(f"[LLMClient] Model {model_candidate} đang quá tải (503). Chuyển model khác...")
                        break  # thoát vòng retry, sang model tiếp theo
                    if attempt < self.max_retries:
                        wait = 2 ** attempt
                        print(
                            f"[LLMClient] API lỗi lần {attempt}: "
                            f"{e}. Retry sau {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        self.model = original_model
                        raise
            else:
                # Hết retry mà không có 503 break → lỗi parse không phục hồi
                self.model = original_model
                continue  # sang model fallback tiếp

            self.model = original_model

        raise RuntimeError(
            f"LLMClient thất bại sau khi thử tất cả models {models_to_try}. "
            f"Lỗi cuối: {last_error}"
        ) from last_error

    def generate_from_stats(
        self,
        match_stats: dict[str, Any],
        max_tokens: int = 2500,
    ) -> dict[str, Any]:
        """
        Shortcut:
        match_stats -> build prompt -> generate
        """
        from .prompt_builder import PromptBuilder

        builder = PromptBuilder(match_stats)
        system_prompt, user_prompt = builder.build()

        return self.generate(
            system_prompt,
            user_prompt,
            max_tokens,
        )

    # ──────────────────────────────────────────────
    # Private — API calls
    # ──────────────────────────────────────────────

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:

        if self.provider == "claude":
            return self._call_claude(
                system_prompt,
                user_prompt,
                max_tokens,
            )

        elif self.provider == "openai":
            return self._call_openai(
                system_prompt,
                user_prompt,
                max_tokens,
            )

        elif self.provider == "gemini":
            return self._call_gemini(
                system_prompt,
                user_prompt,
                max_tokens,
            )

        elif self.provider == "groq":
            return self._call_groq(
                system_prompt,
                user_prompt,
                max_tokens,
            )

        raise ValueError(
            f"Provider không hỗ trợ: '{self.provider}'"
        )

    # ──────────────────────────────────────────────
    # Gemini
    # ──────────────────────────────────────────────

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Cần cài: pip install google-genai"
            )

        client = genai.Client(api_key=self.api_key)

        model_name = self.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.7,
                response_mime_type="application/json",  # ép Gemini trả JSON thuần
            ),
        )

        return response.text

    # ──────────────────────────────────────────────
    # Claude
    # ──────────────────────────────────────────────

    def _call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Cần cài: pip install anthropic"
            )

        client = anthropic.Anthropic(
            api_key=self.api_key
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        return message.content[0].text

    # ──────────────────────────────────────────────
    # OpenAI
    # ──────────────────────────────────────────────

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Cần cài: pip install openai"
            )

        client = OpenAI(
            api_key=self.api_key
        )

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content

    # ──────────────────────────────────────────────
    # Groq
    # ──────────────────────────────────────────────

    def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Cần cài: pip install openai"
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content

    # ──────────────────────────────────────────────
    # Parse
    # ──────────────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
    ) -> dict[str, Any]:

        text = raw.strip()

        # Loại bỏ markdown code block nếu có (```json ... ``` hoặc ``` ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```")
            ).strip()

        # Gemini đôi khi wrap trong object thêm — thử tìm JSON object đầu tiên
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                text = text[start:]

        # Đảm bảo cắt đúng tại dấu } cuối cùng
        if not text.endswith("}"):
            end = text.rfind("}")
            if end != -1:
                text = text[:end + 1]

        data = json.loads(text)

        required = {
            "match_overview",
            "team1_analysis",
            "team2_analysis",
            "comparison",
            "conclusion",
        }

        missing = required - set(data.keys())

        if missing:
            raise KeyError(
                f"Response thiếu các key: {missing}"
            )

        return data