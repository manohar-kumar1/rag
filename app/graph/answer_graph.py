import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.answering.generator import AnswerGenerator
from app.answering.protocol import (
    UNVERIFIED_MESSAGE,
    Answer,
    DoneEvent,
    ProgressEvent,
    Question,
    SourcesEvent,
    StreamEvent,
    TokenEvent,
)
from app.answering.service import answer_events, final_answer
from app.exceptions.exception import GenerationError
from app.retrieval.protocol import RetrievedChunk, Retriever
from app.routing.planning import LlmPlanner
from app.routing.protocol import Grader, Route, RouteDecider
from app.routing.rewrite import LlmQueryRewriter

logger = logging.getLogger(__name__)

RUN_NAME = "answer_question"
FIXED_SUPER_STEPS = 20


@dataclass(frozen=True)
class AgentPolicy:
    max_subquestions: int
    max_iterations: int
    max_rewrites: int
    grade_threshold: float
    groundedness_enabled: bool
    groundedness_threshold: float

    @property
    def recursion_limit(self) -> int:
        return 2 * (self.max_iterations + self.max_subquestions) + FIXED_SUPER_STEPS


class AnswerState(TypedDict, total=False):
    question: Question
    route: Route
    search_query: str
    sources: list[RetrievedChunk]
    answer: Answer
    sub_questions: list[str]
    current: int
    query: str
    tried: list[str]
    iterations: int
    evidence: list[RetrievedChunk]
    unanswered: list[str]
    sufficient: bool


@dataclass(frozen=True)
class AnswerComponents:
    decider: RouteDecider
    rewriter: LlmQueryRewriter
    retriever: Retriever
    generator: AnswerGenerator
    planner: LlmPlanner
    grader: Grader


def merge_evidence(
    evidence: Sequence[RetrievedChunk], found: Sequence[RetrievedChunk]
) -> list[RetrievedChunk]:
    seen = {chunk.chunk_id for chunk in evidence}
    return [*evidence, *(chunk for chunk in found if chunk.chunk_id not in seen)]


def question_to_answer(asked: str, search_query: str) -> str:
    if search_query.strip() == asked.strip():
        return asked
    return f'{search_query}\n(The user asked it as: "{asked}")'


def with_gaps(question: str, unanswered: Sequence[str]) -> str:
    if not unanswered:
        return question
    gaps = "; ".join(unanswered)
    return (
        f"{question}\n(The documents were searched but nothing was found for: {gaps}. "
        "Say that these parts could not be answered.)"
    )


def reported(step: str, node):
    async def run(state: AnswerState) -> AnswerState:
        write = get_stream_writer()
        write(ProgressEvent(step, "running"))
        update = await node(state)
        write(ProgressEvent(step, "done"))
        return update

    return run


def build_answer_graph(
    components: AnswerComponents, policy: AgentPolicy
) -> CompiledStateGraph:
    decider, rewriter, retriever = components.decider, components.rewriter, components.retriever
    planner, grader, generator = components.planner, components.grader, components.generator

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

    async def retrieve(state: AnswerState) -> AnswerState:
        sources = await asyncio.to_thread(
            retriever.retrieve, state["search_query"], state["question"].document_ids
        )
        get_stream_writer()(SourcesEvent(sources))
        return {"sources": sources}

    async def plan(state: AnswerState) -> AnswerState:
        try:
            sub_questions = await asyncio.to_thread(planner.decompose, state["search_query"])
        except Exception as error:  # noqa: BLE001
            logger.warning("planning failed, using the question: %s", type(error).__name__)
            sub_questions = [state["search_query"]]
        first = sub_questions[0]
        return {
            "sub_questions": sub_questions,
            "current": 0,
            "query": first,
            "tried": [first],
            "iterations": 0,
            "evidence": [],
            "unanswered": [],
        }

    async def research(state: AnswerState) -> AnswerState:
        found = await asyncio.to_thread(
            retriever.retrieve, state["query"], state["question"].document_ids
        )
        sub_question = state["sub_questions"][state["current"]]
        sufficient = await is_sufficient(sub_question, found)
        update: AnswerState = {"iterations": state["iterations"] + 1, "sufficient": sufficient}
        if sufficient:
            update["evidence"] = merge_evidence(state["evidence"], found)
        return update

    async def is_sufficient(sub_question: str, found: list[RetrievedChunk]) -> bool:
        if not found:
            return False
        try:
            probability = await asyncio.to_thread(
                grader.sufficiency, sub_question, [chunk.content for chunk in found]
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("grading failed, accepting the passages: %s", type(error).__name__)
            return True
        return probability >= policy.grade_threshold

    def after_research(state: AnswerState) -> str:
        rewrites_used = len(state["tried"]) - 1
        retry = (
            not state["sufficient"]
            and state["iterations"] < policy.max_iterations
            and rewrites_used < policy.max_rewrites
        )
        return "reformulate" if retry else "advance"

    async def reformulate(state: AnswerState) -> AnswerState:
        sub_question = state["sub_questions"][state["current"]]
        try:
            query = await asyncio.to_thread(planner.alternative, sub_question, state["tried"])
        except Exception as error:  # noqa: BLE001
            logger.warning("reformulation failed: %s", type(error).__name__)
            query = sub_question
        return {"query": query, "tried": [*state["tried"], query]}

    def advance(state: AnswerState) -> AnswerState:
        sub_questions = state["sub_questions"]
        unanswered = list(state["unanswered"])
        if not state["sufficient"]:
            unanswered.append(sub_questions[state["current"]])
        following = state["current"] + 1
        if following < len(sub_questions) and state["iterations"] >= policy.max_iterations:
            unanswered.extend(sub_questions[following:])
            following = len(sub_questions)
        update: AnswerState = {"current": following, "unanswered": unanswered}
        if following < len(sub_questions):
            update["query"] = sub_questions[following]
            update["tried"] = [sub_questions[following]]
        return update

    def after_advance(state: AnswerState) -> str:
        return "research" if state["current"] < len(state["sub_questions"]) else "synthesize"

    def synthesize(state: AnswerState) -> AnswerState:
        get_stream_writer()(SourcesEvent(state["evidence"]))
        return {"sources": state["evidence"]}

    async def generate(state: AnswerState) -> AnswerState:
        write = get_stream_writer()
        unanswered = state.get("unanswered", [])
        question = with_gaps(
            question_to_answer(state["question"].text, state["search_query"]), unanswered
        )
        events = answer_events(generator, question, state["sources"])
        shown = events if policy.groundedness_enabled else forward_tokens(events, write)
        answer = await final_answer(shown)
        return {
            "answer": replace(
                answer,
                route=state["route"],
                search_query=state["search_query"],
                unanswered_parts=tuple(unanswered),
            )
        }

    async def verified(answer: Answer) -> Answer:
        if not answer.answered:
            return answer
        cited = [citation.chunk.content for citation in answer.citations]
        passages = cited or [chunk.content for chunk in answer.sources]
        try:
            score = await asyncio.to_thread(grader.groundedness, answer.text, passages)
        except Exception as error:  # noqa: BLE001
            logger.warning("groundedness check failed: %s", type(error).__name__)
            return answer
        if score >= policy.groundedness_threshold:
            return replace(answer, grounded=True)
        logger.info("answer replaced after failing the groundedness check (%.2f)", score)
        return replace(
            answer, text=UNVERIFIED_MESSAGE, answered=False, citations=[], grounded=False
        )

    async def check(state: AnswerState) -> AnswerState:
        if not policy.groundedness_enabled:
            return {}
        answer = await verified(state["answer"])
        get_stream_writer()(TokenEvent(answer.text))
        return {"answer": answer}

    builder = StateGraph(AnswerState)
    for name, node in {
        "classify": reported("classify", classify),
        "rewrite": rewrite,
        "dispatch": dispatch,
        "retrieve": reported("retrieve", retrieve),
        "plan": reported("plan", plan),
        "research": reported("research", research),
        "reformulate": reformulate,
        "advance": advance,
        "synthesize": synthesize,
        "generate": reported("generate", generate),
        "check": reported("check", check) if policy.groundedness_enabled else check,
    }.items():
        builder.add_node(name, node)
    builder.add_edge(START, "classify")
    builder.add_edge(START, "rewrite")
    builder.add_edge(["classify", "rewrite"], "dispatch")
    builder.add_conditional_edges(
        "dispatch",
        lambda state: state["route"],
        {Route.SIMPLE: "retrieve", Route.COMPLEX: "plan"},
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("plan", "research")
    builder.add_conditional_edges("research", after_research, ["reformulate", "advance"])
    builder.add_edge("reformulate", "research")
    builder.add_conditional_edges("advance", after_advance, ["research", "synthesize"])
    builder.add_edge("synthesize", "generate")
    builder.add_edge("generate", "check")
    builder.add_edge("check", END)
    return builder.compile()


async def forward_tokens(
    events: AsyncIterator[TokenEvent | DoneEvent], write
) -> AsyncIterator[TokenEvent | DoneEvent]:
    async for event in events:
        if isinstance(event, TokenEvent):
            write(event)
        yield event


class GraphAnswerer:
    def __init__(self, graph: CompiledStateGraph, recursion_limit: int) -> None:
        self._graph = graph
        self._config: RunnableConfig = {"run_name": RUN_NAME, "recursion_limit": recursion_limit}

    async def ask(self, question: Question) -> Answer:
        state = await self._graph.ainvoke({"question": question}, self._config)
        return state["answer"]

    async def stream(self, question: Question) -> AsyncIterator[StreamEvent]:
        answer: Answer | None = None
        chunks = self._graph.astream(
            {"question": question},
            self._config,
            stream_mode=["custom", "updates"],
            version="v2",
        )
        async with aclosing(chunks):
            async for chunk in chunks:
                if chunk["type"] == "custom":
                    yield chunk["data"]
                    continue
                for update in chunk["data"].values():
                    if update and "answer" in update:
                        answer = update["answer"]
        if answer is None:
            raise GenerationError("answer graph ended without an answer")
        yield DoneEvent(answer)
