"""A small LangGraph tool-call workflow backed by any Merit model executor."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from .agent_contract import AgentResponse, AgentSpec, AgentTask, AgentExecutor, Usage


class _ToolState(TypedDict):
    agent: AgentSpec
    task: AgentTask
    model_output: str
    usage: Usage
    output: str


def _json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LangGraphToolExecutor:
    """Ask a model for a tool call, then validate and record it in a graph node."""

    def __init__(self, model: AgentExecutor) -> None:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise RuntimeError("Install the LangGraph extra: pip install 'merit-agent-eval[langgraph]'") from error
        self.model = model
        self.name = f"langgraph_tools:{model.name}"
        self.tool_traces: list[dict[str, Any]] = []
        graph = StateGraph(_ToolState)
        graph.add_node("model", self._call_model)
        graph.add_node("record_tool_call", self._record_tool_call)
        graph.add_edge(START, "model")
        graph.add_edge("model", "record_tool_call")
        graph.add_edge("record_tool_call", END)
        self._graph = graph.compile()

    def _call_model(self, state: _ToolState) -> dict[str, Any]:
        response = self.model.invoke(state["agent"], state["task"])
        return {"model_output": response.output, "usage": response.usage}

    @staticmethod
    def _record(state: _ToolState) -> tuple[str, dict[str, Any]]:
        raw = state["model_output"]
        call = _json_object(raw)
        trace: dict[str, Any] = {
            "task_id": state["task"].task_id,
            "raw_model_output": raw,
            "valid_tool_call": False,
        }
        if not call or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
            trace["error"] = "Expected a JSON object with name and arguments"
            return raw, trace
        try:
            document = json.loads(state["task"].prompt)
            tools = {item["name"]: item for item in document["tools"]}
            tool = tools[call["name"]]
            parameters = tool["parameters"]
            allowed = set(parameters.get("properties", {}))
            required = set(parameters.get("required", []))
            arguments = call["arguments"]
            if not required.issubset(arguments) or not set(arguments).issubset(allowed):
                raise ValueError("Tool arguments do not match the supplied schema")
            canonical = json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            trace.update({"valid_tool_call": True, "tool_name": call["name"], "arguments": arguments})
            return canonical, trace
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            trace["error"] = str(error)
            return raw, trace

    def _record_tool_call(self, state: _ToolState) -> dict[str, Any]:
        output, trace = self._record(state)
        trace["final_output"] = output
        self.tool_traces.append(trace)
        return {"output": output}

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        state = self._graph.invoke({"agent": agent, "task": task})
        return AgentResponse(output=state["output"], usage=state["usage"])
