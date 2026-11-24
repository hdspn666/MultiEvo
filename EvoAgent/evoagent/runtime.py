"""EvoAgent's dependency-free durable workflow runtime and tool registry.

The runtime deliberately separates orchestration from agent behaviour:

* ``AgentRuntime`` executes named nodes with budgets, retry policy, cancellation
  checks and application-owned checkpoints.
Tool-using model loops live in ``agentic_core.BoundedRole``. Persistence remains
in the application store so a worker restart does not depend on a
framework-owned checkpoint format.
"""
# 是 EvoAgent 的自研工作流引擎，约 220 行，无 LangGraph 等外部编排依赖。文件里有两块能力：

# AgentRuntime — 节点图执行（checkpoint、重试、预算、取消）
# ToolRegistry — Agent 工具注册与参数校验

# harness.py          → 定义业务节点（planning / executing / reviewing）
# runtime.py          → 通用引擎：怎么跑节点、怎么存 checkpoint  ← 你在这里
# agentic_core.py     → Agent 行为：LLM 循环、多 Agent 协作
# repository_tools.py → 构建 ToolRegistry（读文件、跑测试等工具）
# store.py            → checkpoint 持久化（DB）
from contextlib import nullcontext
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# runtime.py
# ├── 异常类
# │   ├── RuntimeBudgetExceeded   # 步数/超时耗尽
# │   ├── RuntimeCancelled        # 任务被取消
# │   └── ToolProtocolError       # 工具参数不合法
# │
# ├── 工具系统（给 Agent 用）
# │   ├── AgentTool               # 单个工具定义
# │   └── ToolRegistry            # 注册 + 校验 + 调用
# │
# └── 工作流引擎（给 harness 用）
#     ├── RuntimeNode             # 节点定义
#     ├── RuntimeEvent            # 运行时事件（可选监听）
#     └── AgentRuntime            # 核心 execute() 方法


class RuntimeBudgetExceeded(RuntimeError):#步数/超时耗尽
    """The configured step or wall-clock budget was exhausted."""


class RuntimeCancelled(RuntimeError):#任务被取消
    """The owning task requested cancellation."""


class ToolProtocolError(RuntimeError):#工具参数不合法
    """A tool request does not match the registered tool contract."""


@dataclass(frozen=True)
class AgentTool:#单个工具定义
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:#注册 + 校验 + 调用
    """Explicit tool catalog with JSON-schema-like argument validation."""

    def __init__(self, tools: Iterable[AgentTool] = ()):
        self._tools: Dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError("tool names must be non-empty and unique")
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def catalog(self) -> List[Dict[str, Any]]:# 生成工具目录（给 LLM 看）
        return [self._tools[name].catalog_entry() for name in self.names()]

    def invoke(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolProtocolError("unknown agent tool: %s" % name)
        self._validate(tool.parameters, arguments)
        return tool.handler(**arguments)

# arguments 传入
#     │
#     ├─ ① 是 dict 吗？
#     │
#     ├─ ② required 字段都传了吗？
#     │
#     ├─ ③ additionalProperties=False 时，有多余字段吗？
#     │
#     ├─ ④ 每个字段类型对吗？（含 bool/integer 陷阱）
#     │
#     └─ ⑤ 数字在 minimum/maximum 范围内吗？
#             │
#             ▼
#         全部通过 → return（invoke 继续调 handler）
#         任一失败 → ToolProtocolError
    @staticmethod
    # schema：工具的 parameters 定义（来自 AgentTool.parameters）
#     以 read_file 为例：
#       schema = {
#           "type": "object",
#           "properties": {
#           "path": {"type": "string"},
#           "start_line": {"type": "integer", "minimum": 1},
#           "end_line": {"type": "integer", "minimum": 1, "maximum": 500},
#           },
#     "required": ["path"],
#     "additionalProperties": False,
# }
    # arguments：实际传入的参数 dict（通常来自 LLM 的 action.arguments）
    def _validate(schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolProtocolError("tool arguments must be an object")
        properties = dict(schema.get("properties") or {})
        required = set(schema.get("required") or [])
        missing = required.difference(arguments)
        if missing:
            raise ToolProtocolError(
                "missing required tool arguments: %s" % ", ".join(sorted(missing))
            )
        if schema.get("additionalProperties", False) is False:
            unknown = set(arguments).difference(properties)
            if unknown:
                raise ToolProtocolError(
                    "unknown tool arguments: %s" % ", ".join(sorted(unknown))
                )
        expected_types = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "object": dict, "array": list,
        }
        for key, value in arguments.items():
            spec = properties.get(key) or {}
            expected = expected_types.get(spec.get("type"))
            if expected and (not isinstance(value, expected) or (
                spec.get("type") in {"integer", "number"} and isinstance(value, bool)
            )):
                raise ToolProtocolError(
                    "tool argument %s must be %s" % (key, spec.get("type"))
                )
            if isinstance(value, (int, float)):
                if "minimum" in spec and value < spec["minimum"]:
                    raise ToolProtocolError("tool argument %s is below minimum" % key)
                if "maximum" in spec and value > spec["maximum"]:
                    raise ToolProtocolError("tool argument %s exceeds maximum" % key)


@dataclass(frozen=True)
class RuntimeNode:#节点定义
    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    retries: Optional[int] = None
    checkpoint: bool = True


@dataclass(frozen=True)
class RuntimeEvent:#运行时事件（可选监听）
    kind: str
    node: str
    step: int
    attempt: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


class AgentRuntime:#核心 execute() 方法
    """Execute a bounded node graph without a third-party orchestration engine."""

# max_steps 整次 execute() 最多执行多少次节点 handler（含重试）
# timeout_seconds 墙钟超时（秒），从 execute() 开始到结束
# node_retries 每个节点失败后的默认重试次数
    def __init__(
        self, max_steps: int = 8, timeout_seconds: int = 120,
        node_retries: int = 0,
    ):
        if max_steps < 1:
            raise ValueError("runtime max_steps must be at least 1")
        if timeout_seconds < 1:
            raise ValueError("runtime timeout_seconds must be at least 1")
        if node_retries < 0:
            raise ValueError("runtime node_retries cannot be negative")
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.node_retries = node_retries

# initial_state 初始状态
# nodes 节点列表
# task_id 任务 ID
# checkpoint_store 断点存储
# cancel_check 取消检查
# event_sink 事件监听
# span_factory 跨度工厂 ？？？
# non_retryable 不可重试的异常类型
    def execute(
        self, initial_state: Dict[str, Any], nodes: Iterable[RuntimeNode],
        task_id: str = "", checkpoint_store=None,
        cancel_check: Optional[Callable[[], bool]] = None,
        event_sink: Optional[Callable[[RuntimeEvent], None]] = None,
        span_factory: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        non_retryable: Tuple[type, ...] = (ValueError, RuntimeCancelled, RuntimeBudgetExceeded),
    ) -> Dict[str, Any]:
        state = dict(initial_state)
        started = time.monotonic()
        steps = 0
        checkpoints = (
            checkpoint_store.load_checkpoints(task_id)
            if checkpoint_store is not None and task_id else {}
        )

        # 可选事件通知 测试用
        def emit(kind: str, node: str, attempt: int = 0, **detail) -> None:
            if event_sink:
                event_sink(RuntimeEvent(kind, node, steps, attempt, detail))

        def guard(node: str) -> None:# 步数 ≥ max_steps 或超时 → RuntimeBudgetExceeded
            if cancel_check and cancel_check():
                emit("cancelled", node)
                raise RuntimeCancelled("Task was cancelled")
            if steps >= self.max_steps or time.monotonic() - started > self.timeout_seconds:
                emit("budget_exhausted", node)
                raise RuntimeBudgetExceeded("task execution budget exceeded")

        for node in nodes:
            cached = checkpoints.get(node.name) if node.checkpoint else None
            if cached and cached.get("status") == "completed":# checkpoint 已完成 → 跳过 handler
                output = dict(cached.get("state") or {})
                state.update(output)
                emit("checkpoint_restored", node.name, int(cached.get("attempt", 0)))
                continue

            retries = self.node_retries if node.retries is None else node.retries
            previous_attempt = int((cached or {}).get("attempt", 0))
            last_error: Optional[Exception] = None
            for offset in range(1, retries + 2):
                guard(node.name)
                steps += 1
                attempt = previous_attempt + offset
                emit("node_started", node.name, attempt)
                try:
                    attrs = {
                        "task_id": task_id, "node": node.name,
                        "attempt": attempt, "runtime_step": steps,
                    }
                    context = (
                        span_factory("runtime.%s" % node.name, attrs)
                        if span_factory else nullcontext()
                    )
                    with context:
                        output = node.handler(state) or {}
                    if not isinstance(output, dict):
                        raise TypeError("runtime node %s must return a dict" % node.name)
                    state.update(output)# 合并产出
                    if checkpoint_store is not None and task_id and node.checkpoint:
                        checkpoint_store.save_checkpoint(
                            task_id, node.name, output, "completed", attempt
                        )
                    emit("node_completed", node.name, attempt, output_keys=sorted(output))
                    last_error = None
                    break
                except non_retryable:
                    raise
                except Exception as exc:
                    last_error = exc
                    if checkpoint_store is not None and task_id and node.checkpoint:
                        checkpoint_store.save_checkpoint(
                            task_id, node.name, {}, "failed", attempt, str(exc)
                        )
                    emit(
                        "node_failed", node.name, attempt,
                        error=str(exc)[:1000], will_retry=offset <= retries,
                    )
            if last_error is not None:
                raise last_error
        return state
