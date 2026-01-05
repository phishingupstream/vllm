"""
Nemotron 3 Nano Reasoning Parser for vLLM

This custom parser extends DeepSeekR1ReasoningParser to support the
enable_thinking parameter for Nemotron 3 Nano models.

Background:
-----------
Nemotron 3 Nano models support toggling reasoning via chat_template_kwargs:
    {"enable_thinking": True}   - Model generates <think>...</think> reasoning
    {"enable_thinking": False}  - Model skips reasoning, outputs directly

The model's chat template checks this parameter and either:
    - Injects "<think>\n" (thinking enabled) - model generates reasoning
    - Injects "<think></think>" (thinking disabled) - model skips to content

Bug Fixed:
----------
vLLM's base reasoning parser (DeepSeekR1ReasoningParser) didn't handle the
enable_thinking=False case for streaming. When thinking was disabled:
    - Non-streaming: Worked (we swap fields in extract_reasoning)
    - Streaming: BROKEN - content went to reasoning_content field

Root cause: The streaming parser couldn't detect enable_thinking=False from
token IDs alone because the prompt tokens aren't passed to the parser.

Fix: vLLM's serving_chat.py passes chat_template_kwargs to the parser
constructor. We capture this and use it to route streaming content correctly.

Responses API Limitation:
-------------------------
The Responses API (serving_responses.py) does NOT pass chat_template_kwargs
to the parser constructor. It calls:
    reasoning_parser = self.reasoning_parser(tokenizer)  # Missing kwargs!

Until vLLM fixes this, enable_thinking=False won't work with Responses API.
Use Chat Completions API (/v1/chat/completions) instead.

Usage:
------
In vLLM config:
    reasoning-parser-plugin: "/path/to/nano_v3_reasoning_parser.py"
    reasoning-parser: "nano_v3"

Client request:
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
"""
from typing import Sequence
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    """Nemotron 3 Nano reasoning parser with enable_thinking support.

    Extends DeepSeekR1ReasoningParser to handle enable_thinking=False correctly
    for both streaming and non-streaming Chat Completions API requests.

    Attributes:
        _thinking_disabled: Set at construction from chat_template_kwargs.
            Used by streaming methods which don't have access to request object.
    """

    def __init__(self, tokenizer, *args, chat_template_kwargs=None, **kwargs):
        """Initialize parser, capturing enable_thinking from chat_template_kwargs.

        Args:
            tokenizer: The tokenizer for the model.
            chat_template_kwargs: Dict passed from request, may contain
                {"enable_thinking": False} to disable reasoning output.
        """
        super().__init__(tokenizer, *args, **kwargs)
        # Capture enable_thinking at construction for streaming use
        # (streaming methods don't receive the request object)
        self._thinking_disabled = (
            chat_template_kwargs is not None
            and chat_template_kwargs.get("enable_thinking") is False
        )

    def _is_thinking_disabled(self, request) -> bool:
        """Check if thinking is disabled via request's chat_template_kwargs.

        Used by non-streaming extract_reasoning which receives the request.
        """
        return (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and request.chat_template_kwargs.get("enable_thinking") is False
        )

    def extract_reasoning(self, model_output, request):
        """Extract reasoning for non-streaming responses.

        When enable_thinking=False, the parent parser puts all content in
        reasoning_content (because it doesn't find </think> in output).
        We detect this and swap the fields.
        """
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        # If thinking disabled and content ended up in reasoning, swap them
        if self._is_thinking_disabled(request) and final_content is None:
            reasoning_content, final_content = final_content, reasoning_content
        return reasoning_content, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract reasoning for streaming responses.

        When thinking is disabled (detected at construction), all generated
        content goes directly to the content field, not reasoning.
        """
        if self._thinking_disabled:
            # All output is final content, no reasoning
            return DeltaMessage(content=delta_text)

        # Normal case: use parent's <think>...</think> parsing
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )