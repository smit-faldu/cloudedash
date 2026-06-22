"""
graph package — LangGraph graph compilation and state (Stage 5).

Modules
-------
graph/state.py      — GraphState TypedDict with add_messages and merge_entities reducers
graph/router.py     — Conditional edge functions (pure routing logic, no LLMs)
graph/graph.py      — Compiled StateGraph connecting all agent nodes

Public API
----------
    from graph.graph import build_graph, get_graph, create_initial_state
    from graph.state import GraphState, merge_entities
    from graph.router import route_after_triage, route_after_specialist
"""
from graph.graph import build_graph, create_initial_state, get_graph
from graph.router import route_after_specialist, route_after_triage
from graph.state import GraphState, merge_entities

__all__ = [
    "GraphState",
    "merge_entities",
    "route_after_triage",
    "route_after_specialist",
    "build_graph",
    "create_initial_state",
    "get_graph",
]
