"""QA agent (MLflow ChatModel) backed by Anthropic Claude.

This file is the agent source kept in Git. Any change merged to the main branch is
registered as a new version of the Unity Catalog model by register_agent.py, which
automatically triggers the deployment (evaluation) job.

Tip for the demo: change SYSTEM_PROMPT (e.g. ask for one-word answers) and push to see
the evaluation result change between versions.
"""
import mlflow
from anthropic import Anthropic
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
)

AGENT_CODE_VERSION = "1.0.0"  # bump when you change behaviour; logged as a model version tag
MODEL_ID = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "You are a helpful assistant. Answer questions concisely."
DEFAULT_MAX_TOKENS = 1024


def _to_anthropic(messages):
    """Split MLflow chat messages into (system prompt, Anthropic message list)."""
    system_msg, out = SYSTEM_PROMPT, []
    for m in messages:
        if m.role == "system":
            system_msg = m.content
        else:
            out.append({"role": m.role, "content": m.content})
    return system_msg, out


class QAAgentModel(ChatModel):
    """Agent backed by Anthropic Claude, using ChatModel for native streaming support."""

    def predict(self, messages: list[ChatMessage], params: ChatParams = None) -> ChatCompletionResponse:
        client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        system_msg, anthropic_messages = _to_anthropic(messages)
        max_tokens = params.max_tokens if params and params.max_tokens else DEFAULT_MAX_TOKENS
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=max_tokens,
            system=system_msg,
            messages=anthropic_messages,
            timeout=60.0,
        )
        return ChatCompletionResponse(
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=response.content[0].text),
                                finish_reason="stop")],
            model=MODEL_ID,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    def predict_stream(self, messages: list[ChatMessage], params: ChatParams = None):
        """Stream the response token-by-token using Anthropic's streaming API."""
        client = Anthropic()
        system_msg, anthropic_messages = _to_anthropic(messages)
        max_tokens = params.max_tokens if params and params.max_tokens else DEFAULT_MAX_TOKENS
        with client.messages.stream(
            model=MODEL_ID,
            max_tokens=max_tokens,
            system=system_msg,
            messages=anthropic_messages,
            timeout=60.0,
        ) as stream:
            for text in stream.text_stream:
                yield ChatCompletionChunk(choices=[ChatChunkChoice(index=0, delta=ChatChoiceDelta(content=text))],
                                          model=MODEL_ID)


# Required for MLflow code-based logging
mlflow.models.set_model(QAAgentModel())
