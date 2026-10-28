"""A bounded multi-step model loop with host-validated lookup and calculator tools."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict
from typing import Any, Mapping

from .agent_contract import AgentExecutor, AgentResponse, AgentSpec, AgentTask, Usage

_ALLOWED_BINARY = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
}
_ALLOWED_UNARY = {ast.UAdd: lambda value: value, ast.USub: lambda value: -value}


def safe_calculator(expression: str) -> int | float:
    """Evaluate numeric arithmetic without names, calls, or eval."""
    if len(expression) > 80:
        raise ValueError("Expression is too long")

    def visit(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 12:
            raise ValueError("Expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1_000_000_000:
                raise ValueError("Numeric literal is outside the allowed range")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")
            result = _ALLOWED_BINARY[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            result = _ALLOWED_UNARY[type(node.op)](visit(node.operand, depth + 1))
        else:
            raise ValueError("Only numeric +, -, *, / and parentheses are allowed")
        if not math.isfinite(result) or abs(result) > 1_000_000_000:
            raise ValueError("Result is outside the allowed range")
        return result

    return visit(ast.parse(expression, mode="eval"))


class MultiStepToolExecutor:
    """Let the model request host tools, then feed their results back to the model."""

    def __init__(
        self,
        model: AgentExecutor,
        lookup_values: Mapping[str, str | int | float],
        max_steps: int = 5,
        recover_tool_errors: bool = False,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model = model
        suffix = ":recover_tool_errors" if recover_tool_errors else ""
        self.name = f"multi_step_tools:{model.name}{suffix}"
        self.lookup_values = dict(lookup_values)
        self.max_steps = max_steps
        self.recover_tool_errors = recover_tool_errors
        self.traces: list[dict[str, Any]] = []

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        history: list[str] = []
        total_usage = Usage()
        output = ""
        for step in range(self.max_steps):
            prompt = task.prompt
            if history:
                prompt += "\nTool results so far:\n" + "\n".join(history)
            response = self.model.invoke(agent, AgentTask(task.task_id, prompt, task.expected))
            total_usage = total_usage + response.usage
            output = response.output
            trace: dict[str, Any] = {
                "artifact_id": agent.artifact_id,
                "task_id": task.task_id,
                "step": step,
                "input": prompt,
                "raw_model_output": response.output,
                "action": None,
                "argument": None,
                "tool_result": None,
                "tool_error": None,
                "usage": asdict(response.usage),
            }
            try:
                action = json.loads(response.output.strip())
                if not isinstance(action, dict) or set(action) != {"action", "argument"}:
                    raise ValueError("Expected one JSON object with action and argument")
                name = action["action"]
                argument = str(action["argument"]).strip()
                trace["action"] = name
                trace["argument"] = argument
                if name == "final":
                    output = argument
                elif name == "lookup":
                    if argument not in self.lookup_values:
                        raise ValueError("Unknown lookup key")
                    result = str(self.lookup_values[argument])
                    trace["tool_result"] = result
                    history.append(f"lookup {argument} => {result}")
                elif name == "calculator":
                    result = str(safe_calculator(argument))
                    trace["tool_result"] = result
                    history.append(f"calculator {argument} => {result}")
                else:
                    raise ValueError("Unsupported action")
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as error:
                trace["tool_error"] = str(error)
                trace["final_output"] = output
                self.traces.append(trace)
                if self.recover_tool_errors and trace["action"] in {"lookup", "calculator"}:
                    history.append(f"{trace['action']} {trace['argument']} => error: {error}")
                    continue
                break
            trace["final_output"] = output
            self.traces.append(trace)
            if trace["action"] == "final":
                break
        return AgentResponse(output=output, usage=total_usage)
