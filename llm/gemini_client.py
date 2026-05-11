"""
llm/gemini_client.py – Gemini API client using the new google-genai SDK.

Wraps the google-genai package for two use-cases:
  1. One-shot text completions (insight generation)
  2. Agentic multi-turn sessions with function calling (audio agent)

Migration note: uses `google-genai` (google.genai) which replaced the
deprecated `google-generativeai` (google.generativeai) package.
"""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """
    Facade over the google-genai SDK.

    Usage
    -----
    client = GeminiClient()

    # Simple one-shot completion:
    text = client.complete("Summarise this audio analysis: ...")

    # Agentic model with function calling:
    model = client.build_agent_model(tool_declarations, system_instruction)
    chat  = model.start_chat()
    resp  = chat.send_message("Analyse: /path/to/audio.wav")
    """

    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file. "
                "Get a free key at https://aistudio.google.com"
            )
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model_name = settings.GEMINI_MODEL
        logger.info("Gemini client initialised (model: %s)", self._model_name)

    # One-shot completion
    def complete(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Send a single prompt and return the model's text response.

        Parameters
        ----------
        prompt              : User message.
        system_instruction  : Optional system / persona instruction.
        temperature         : Sampling temperature (lower = more deterministic).
        """
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )

        text = response.text or ""
        logger.debug("Gemini completion (first 200 chars): %s", text[:200])
        return text

    # Agentic model (function calling)
    def build_agent_model(
        self,
        tool_declarations: list[types.FunctionDeclaration],
        system_instruction: str,
    ) -> "_AgentSession":
        """
        Return a helper that manages a multi-turn function-calling chat session.
        """
        return _AgentSession(
            client=self._client,
            model_name=self._model_name,
            tool_declarations=tool_declarations,
            system_instruction=system_instruction,
        )

    # Static helpers
    @staticmethod
    def make_function_response(name: str, result: Any) -> types.Part:
        """Wrap a tool-call result in a Gemini Part."""
        payload = result if isinstance(result, str) else json.dumps(result)
        return types.Part.from_function_response(
            name=name,
            response={"result": payload},
        )

    @staticmethod
    def extract_function_calls(response: Any) -> list[types.FunctionCall]:
        """Pull all function_call parts out of a GenerateContentResponse."""
        calls: list[types.FunctionCall] = []
        if not response or not response.candidates:
            return calls
        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if part.function_call and part.function_call.name:
                    calls.append(part.function_call)
        return calls

    @staticmethod
    def extract_text(response: Any) -> str:
        """Concatenate all text parts from a response."""
        if not response or not response.candidates:
            return ""
        texts: list[str] = []
        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if part.text:
                    texts.append(part.text)
        return "\n".join(texts)

# Internal: multi-turn chat session wrapper
class _AgentSession:
    """
    Wraps a google-genai Chat so AudioAnalysisAgent can call
    start_chat() and send_message() with the same interface.
    """

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        tool_declarations: list[types.FunctionDeclaration],
        system_instruction: str,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._config = types.GenerateContentConfig(
            temperature=0.1,
            tools=[types.Tool(function_declarations=tool_declarations)],
            system_instruction=system_instruction,
        )

    def start_chat(self) -> "_Chat":
        """Create a fresh multi-turn chat session."""
        chat = self._client.chats.create(
            model=self._model_name,
            config=self._config,
        )
        return _Chat(chat)

class _Chat:
    """Thin wrapper around a google-genai Chat for the agentic loop."""

    def __init__(self, chat: Any) -> None:
        self._chat = chat

    def send_message(self, content: Any) -> Any:
        """Send content (string or list of Parts) and return the response."""
        return self._chat.send_message(content)
