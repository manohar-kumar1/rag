import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.answering.generator import AnswerGenerator
from app.answering.protocol import (
    Answer,
    DoneEvent,
    Question,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from app.answering.service import answer_events, final_answer
from app.retrieval.protocol import RetrievedChunk, Retriever
from app.routing.protocol import Route, RouteDecider
from app.routing.rewrite import LlmQueryRewriter

logger = logging.getLogger(__name__)

RUN_CONFIG: RunnableConfig = {"run_name": "answer_question"}


class AnswerState(TypedDict, total=False):
    question: Question
    route: Route
    search_query: str
    sources: list[RetrievedChunk]
    answer: Answer


def build_answer_graph(
    decider: RouteDecider,
    rewriter: LlmQueryRewriter,
    retriever: Retriever,
    generator: AnswerGenerator,
) -> CompiledStateGraph:
    async def classify(state: AnswerState) -> AnswerState:
        try:
            route = await asyncio.to_thread(decider.decide, state["question"])
        except Exception as error:  # noqa: BLE001
            logger.warning("route decision failed, using complex path: %s", type(error).__name__)
            route = Route.COMPLEX
        return {"route": route}

    async def rewrite(state: AnswerState) -> AnswerState:
        question = state["question"]
        try:
            query = await asyncio.to_thread(rewriter.rewrite, question)
        except Exception as error:  # noqa: BLE001
            logger.warning("query rewrite failed, using the question: %s", type(error).__name__)
            query = question.text
        return {"search_query": query}

    def dispatch(state: AnswerState) -> AnswerState:
        return {}

    def complex_placeholder(state: AnswerState) -> AnswerState:
        return {}

    async def retrieve(state: AnswerState) -> AnswerState:
        sources = await asyncio.to_thread(retriever.retrieve, state["search_query"])
        get_stream_writer()(SourcesEvent(sources))
        return {"sources": sources}

    async def generate(state: AnswerState) -> AnswerState:
        write = get_stream_writer()
        question = question_to_answer(state["question"].text, state["search_query"])
        answer = await final_answer(
            forward_tokens(answer_events(generator, question, state["sources"]), write)
        )
        return {
            "answer": replace(answer, route=state["route"], search_query=state["search_query"])
        }

    builder = StateGraph(AnswerState)
    builder.add_node("classify", classify)
    builder.add_node("rewrite", rewrite)
    builder.add_node("dispatch", dispatch)
    builder.add_node("complex_placeholder", complex_placeholder)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "classify")
    builder.add_edge(START, "rewrite")
    builder.add_edge(["classify", "rewrite"], "dispatch")
    builder.add_conditional_edges(
        "dispatch",
        lambda state: state["route"],
        {Route.SIMPLE: "retrieve", Route.COMPLEX: "complex_placeholder"},
    )
    builder.add_edge("complex_placeholder", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def question_to_answer(asked: str, search_query: str) -> str:
    if search_query.strip() == asked.strip():
        return asked
    return f'{search_query}\n(The user asked it as: "{asked}")'


async def forward_tokens(
    events: AsyncIterator[TokenEvent | DoneEvent], write
) -> AsyncIterator[TokenEvent | DoneEvent]:
    async for event in events:
        if isinstance(event, TokenEvent):
            write(event)
        yield event


class GraphAnswerer:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    async def ask(self, question: Question) -> Answer:
        state = await self._graph.ainvoke({"question": question}, RUN_CONFIG)
        return state["answer"]

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        final: AnswerState = {}
        updates = self._graph.astream(
            {"question": question}, RUN_CONFIG, stream_mode=["custom", "values"]
        )
        async with aclosing(updates):
            async for mode, payload in updates:
                if mode == "custom":
                    yield payload
                else:
                    final = payload
        yield DoneEvent(final["answer"])
