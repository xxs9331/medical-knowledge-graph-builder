"""候选图构建与评测流程的顺序事件账本。

本模块借鉴 DeepSeek Harness 的事件流水设计：本地 JSONL 是可回放的权威记录，
未来的 OpenTelemetry 等后端只能消费该记录，不能成为业务流程依赖。Trace 只记录
阶段身份、状态、数量、耗时和工件引用，不保存原文、提示词或模型完整响应。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self


TRACE_SCHEMA_VERSION = "graph-builder-trace/v0.1"


class TraceRecorder(Protocol):
    """业务模块使用的最小 Trace 合同。"""

    @property
    def run_id(self) -> str: ...

    def record(self, event_type: str, **data: Any) -> None: ...

    def stage(self, stage: str, **data: Any) -> AbstractContextManager["TraceStage"]: ...


class TraceStage:
    """保存一个阶段结束时才知道的统计字段。"""

    def __init__(self) -> None:
        self._result: dict[str, Any] = {}

    def update(self, **data: Any) -> Self:
        """追加阶段结果；同名字段以后一次写入为准。"""
        self._result.update(data)
        return self

    @property
    def result(self) -> Mapping[str, Any]:
        return self._result


class NullTrace:
    """关闭 Trace 时使用的空实现，使业务代码不需要反复判断 ``None``。"""

    run_id: str = "disabled"

    def record(self, event_type: str, **data: Any) -> None:
        del event_type, data

    @contextmanager
    def stage(self, stage: str, **data: Any) -> Generator[TraceStage, None, None]:
        del stage, data
        yield TraceStage()


NULL_TRACE = NullTrace()


class JsonlTrace:
    """将带顺序号的流程事件追加到一个 JSONL 文件。

    写入采用进程内锁和单次 ``os.write``。记录失败只保存在 ``write_errors`` 中，
    不允许旁路观测故障改变抽取、Judge 或评分结果。
    """

    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        self.path = path
        self._run_id = run_id or str(uuid.uuid4())
        self._sequence = 0
        self._lock = threading.Lock()
        self._write_errors: list[str] = []
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._write_errors.append(type(error).__name__)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def write_errors(self) -> tuple[str, ...]:
        """返回被隔离的写入错误类型，不暴露可能包含路径或正文的异常消息。"""
        return tuple(self._write_errors)

    def record(self, event_type: str, **data: Any) -> None:
        """尽力追加一个事件；任何序列化或文件错误都不会向业务层传播。"""
        try:
            normalized = _normalize_json(data)
            if not isinstance(normalized, dict):
                raise TypeError("trace event data must be an object")
            with self._lock:
                sequence = self._sequence
                record = {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "run_id": self._run_id,
                    "seq": sequence,
                    "at": datetime.now(UTC).isoformat(),
                    "type": event_type,
                    "data": normalized,
                }
                line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    written = os.write(descriptor, line)
                    if written != len(line):
                        raise OSError("incomplete trace write")
                finally:
                    os.close(descriptor)
                self._sequence += 1
        except (OSError, TypeError, ValueError) as error:
            self._write_errors.append(type(error).__name__)

    @contextmanager
    def stage(self, stage: str, **data: Any) -> Generator[TraceStage, None, None]:
        """成对记录阶段开始和结束；异常结束只记录异常类型并继续向上抛出。"""
        started = time.perf_counter()
        self.record(f"{stage}/start", **data)
        state = TraceStage()
        try:
            yield state
        except BaseException as error:
            self.record(
                f"{stage}/end",
                **data,
                **state.result,
                status="error",
                error_type=type(error).__name__,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
            raise
        else:
            result = dict(state.result)
            result.setdefault("status", "success")
            self.record(
                f"{stage}/end",
                **data,
                **result,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )


def _normalize_json(value: Any) -> Any:
    """复制为稳定的 JSON 值；拒绝无法安全表达的任意对象和非有限浮点数。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("trace numbers must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported trace value: {type(value).__name__}")
