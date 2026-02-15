"""RAG agent for answering informational questions using retrieved context."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage

from backend.agents.state import AgentState
from backend.config import Settings, get_settings
from backend.data.indexer import ArgoKnowledgeIndexer

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an expert oceanographer assistant. Answer the user's question
using the retrieved context below. Be accurate, concise, and helpful.

If the context doesn't contain enough information to fully answer the question,
say so and provide what you can based on your general knowledge of oceanography.

Always cite which source documents informed your answer.

CRITICAL FORMATTING RULES:
- Your response goes directly to the end user in a chat interface.
- NEVER include XML tags, tool calls, function calls, or any internal markup.
- NEVER show <tool_call>, <function_calls>, <invoke>, or similar tags.
- Write only clean, natural language.

Retrieved context:
{context}
"""


def create_rag_agent(
    settings: Settings | None = None,
    indexer: ArgoKnowledgeIndexer | None = None,
) -> "RagAgent":
    """Create a RAG agent instance."""
    return RagAgent(settings=settings, indexer=indexer)


class RagAgent:
    """Retrieval-Augmented Generation agent for Argo knowledge queries."""

    def __init__(
        self,
        settings: Settings | None = None,
        indexer: ArgoKnowledgeIndexer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._indexer = indexer or ArgoKnowledgeIndexer(settings=self._settings)
        self._llm = ChatAnthropic(
            model=self._settings.anthropic_model,
            api_key=self._settings.anthropic_api_key,
            max_tokens=1024,
        )

    def _retrieve_context(self, query: str) -> tuple[str, list[str]]:
        """Retrieve relevant documents and format as context string."""
        results = self._indexer.search(query)
        if not results:
            return "No relevant documents found.", []

        context_parts = []
        sources = []
        for doc in results:
            context_parts.append(f"[{doc['id']}]: {doc['content']}")
            sources.append(doc["id"])

        return "\n\n".join(context_parts), sources

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the RAG agent on the current state."""
        messages = state["messages"]
        user_query = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and (
                getattr(msg, "type", None) == "human"
                or getattr(msg, "role", None) == "user"
            ):
                user_query = msg.content
                break

        if not user_query:
            return {
                "messages": [AIMessage(content="I didn't receive a question. Could you please ask again?")],
                "data": {},
            }

        context, sources = self._retrieve_context(user_query)

        system_msg = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
        response = await self._llm.ainvoke([system_msg, *messages])

        return {
            "messages": [response],
            "data": {"sources": sources},
        }
