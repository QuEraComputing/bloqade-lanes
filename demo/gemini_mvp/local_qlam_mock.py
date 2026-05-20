"""Small local QLAM-compatible mock for notebook integration tests.

This server intentionally implements only the endpoints used by the
``bloqade-internal`` demo path:

    POST /v2/{qpu_mode}/tasks
    GET  /v2/{qpu_mode}/tasks
    GET  /v2/{qpu_mode}/tasks/{task_id}
    PUT  /v2/{qpu_mode}/tasks/{task_id}/cancel
    GET  /v2/{qpu_mode}/tasks/{task_id}/results

It is not a compiler, scheduler, simulator, or production QLAM replacement.
Its job is to test the notebook -> SDK -> qlam-core -> HTTP -> Future.result()
plumbing without needing a deployed QLAM stack.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

USER_ID = "00000000-0000-0000-0000-000000000001"
MOCK_MODE_ZEROS = "zeros"
MOCK_MODE_ONES = "ones"
MOCK_MODE_RANDOM = "random"


def collect_squin_gate_classes(value: object) -> set[str]:
    """Return serialized Squin gate statement class names found in a program."""
    gate_classes: set[str] = set()
    if isinstance(value, dict):
        module_name = value.get("module_name")
        class_name = value.get("class_name")
        if (
            isinstance(module_name, str)
            and module_name.startswith("bloqade.squin.gate")
            and isinstance(class_name, str)
        ):
            gate_classes.add(class_name)

        for child in value.values():
            gate_classes.update(collect_squin_gate_classes(child))
    elif isinstance(value, list):
        for child in value:
            gate_classes.update(collect_squin_gate_classes(child))

    return gate_classes


def infer_mock_measurement_mode(program_content: object) -> str:
    """Infer an MVP measurement response from the submitted Squin program.

    This intentionally recognizes only the three kernels used by the Gemini MVP:
    no gate operations, a global X pulse, or a global sqrt(X) pulse.
    """
    if isinstance(program_content, str):
        try:
            program_content = json.loads(program_content)
        except json.JSONDecodeError:
            return MOCK_MODE_RANDOM

    gate_classes = collect_squin_gate_classes(program_content)
    if not gate_classes:
        return MOCK_MODE_ZEROS
    if gate_classes == {"X"}:
        return MOCK_MODE_ONES
    if gate_classes == {"SqrtX"}:
        return MOCK_MODE_RANDOM

    return MOCK_MODE_RANDOM


def utc_now() -> str:
    """Return a JSON-friendly aware timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, 0)


@dataclass
class MockTask:
    qpu_mode: str
    definition: dict
    bitstring_width: int
    execution_latency_seconds: float
    task_id: str = field(default_factory=lambda: str(uuid4()))
    definition_id: str = field(default_factory=lambda: str(uuid4()))
    compilation_id: str = field(default_factory=lambda: str(uuid4()))
    task_status: str = "PayloadProcessing"
    created_date: str = field(default_factory=utc_now)
    modified_date: str = field(default_factory=utc_now)
    created_monotonic: float = field(default_factory=time.monotonic)
    program_modes: list[str] = field(init=False)

    def __post_init__(self) -> None:
        programs = self.definition.get("programs", [])
        self.program_modes = [
            infer_mock_measurement_mode(program.get("content"))
            for program in programs
            if isinstance(program, dict)
        ]

    def current_status(self) -> str:
        if self.task_status in {"Cancelled", "Failed", "PayloadProcessingError"}:
            return self.task_status

        if time.monotonic() - self.created_monotonic < self.execution_latency_seconds:
            return "PayloadProcessing"

        if self.task_status != "Completed":
            self.task_status = "Completed"
            self.modified_date = utc_now()
        return self.task_status

    def as_task_response(self) -> dict:
        status = self.current_status()
        return {
            "id": self.task_id,
            "task_status": status,
            "definition_id": self.definition_id,
            "compilation_id": self.compilation_id,
            "created_by": USER_ID,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "modified_by": USER_ID,
            "error_reasons": [],
        }

    def _bitstring(self, subtask_index: int, shot_index: int) -> list[bool]:
        subtasks = self.definition.get("subtasks", [])
        if subtask_index < len(subtasks) and isinstance(subtasks[subtask_index], dict):
            program_index = int(
                subtasks[subtask_index].get("program_index", subtask_index)
            )
        else:
            program_index = subtask_index

        if program_index < len(self.program_modes):
            mode = self.program_modes[program_index]
        else:
            mode = MOCK_MODE_RANDOM

        if mode == MOCK_MODE_ZEROS:
            return [False] * self.bitstring_width

        if mode == MOCK_MODE_ONES:
            return [True] * self.bitstring_width

        # Deterministic per task/shot so repeated result fetches are stable.
        seed = f"{self.task_id}:{subtask_index}:{shot_index}"
        rng = random.Random(seed)
        return [bool(rng.getrandbits(1)) for _ in range(self.bitstring_width)]

    def result_subtasks(self, shots_page: int, shots_size: int) -> list[dict]:
        subtasks = self.definition.get("subtasks", [])
        result_subtasks = []
        status = self.current_status()

        for subtask_index, subtask in enumerate(subtasks):
            num_shots = int(subtask.get("num_shots", 0))
            start = shots_page * shots_size
            end = min(start + shots_size, num_shots)

            shot_results = []
            if status == "Completed":
                for shot in range(start, end):
                    shot_results.append(
                        {
                            "shot_index": shot,
                            "subtask_shot_index": shot,
                            "subtask_index": subtask_index,
                            "frame_type": "DETECTED",
                            "measurement": {
                                "measurement_values": self._bitstring(
                                    subtask_index, shot
                                )
                            },
                        }
                    )

            result_subtasks.append(
                {
                    "subtask_index": subtask_index,
                    "program_index": subtask.get("program_index", subtask_index),
                    "num_shots": num_shots,
                    "status": status.upper(),
                    "shot_results": shot_results,
                }
            )

        return result_subtasks


class LocalQlamState:
    def __init__(
        self,
        *,
        bitstring_width: int,
        require_auth: bool,
        execution_latency_seconds: float,
    ):
        self.bitstring_width = bitstring_width
        self.require_auth = require_auth
        self.execution_latency_seconds = execution_latency_seconds
        self.tasks: dict[str, MockTask] = {}

    def create_task(self, qpu_mode: str, definition: dict) -> MockTask:
        task = MockTask(
            qpu_mode=qpu_mode,
            definition=definition,
            bitstring_width=self.bitstring_width,
            execution_latency_seconds=self.execution_latency_seconds,
        )
        self.tasks[task.task_id] = task
        return task


class LocalQlamHandler(BaseHTTPRequestHandler):
    server_version = "LocalQlamMock/0.1"

    @property
    def state(self) -> LocalQlamState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        if self.server.quiet:  # type: ignore[attr-defined]
            return
        super().log_message(format, *args)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"messages": [message]})

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_error(400, "Request body is not valid JSON")
            return None

        if not isinstance(body, dict):
            self._send_error(400, "Request body must be a JSON object")
            return None

        return body

    def _check_auth(self) -> bool:
        if not self.state.require_auth:
            return True
        if self.headers.get("authorization"):
            return True
        self._send_error(401, "Missing Authorization header")
        return False

    def _route_parts(self) -> list[str]:
        return [part for part in urlparse(self.path).path.split("/") if part]

    def do_GET(self) -> None:
        parts = self._route_parts()
        if parts == ["health"]:
            self._send_json(200, {"status": "ok"})
            return

        if not self._check_auth():
            return

        if len(parts) == 3 and parts[0] == "v2" and parts[2] == "tasks":
            self._list_tasks(qpu_mode=parts[1])
            return

        if len(parts) == 4 and parts[0] == "v2" and parts[2] == "tasks":
            self._get_task(qpu_mode=parts[1], task_id=parts[3])
            return

        if (
            len(parts) == 5
            and parts[0] == "v2"
            and parts[2] == "tasks"
            and parts[4] == "results"
        ):
            self._get_results(qpu_mode=parts[1], task_id=parts[3])
            return

        self._send_error(404, f"No mock route for GET {urlparse(self.path).path}")

    def do_POST(self) -> None:
        parts = self._route_parts()
        if not self._check_auth():
            return

        if len(parts) == 3 and parts[0] == "v2" and parts[2] == "tasks":
            self._create_task(qpu_mode=parts[1])
            return

        self._send_error(404, f"No mock route for POST {urlparse(self.path).path}")

    def do_PUT(self) -> None:
        parts = self._route_parts()
        if not self._check_auth():
            return

        if (
            len(parts) == 5
            and parts[0] == "v2"
            and parts[2] == "tasks"
            and parts[4] == "cancel"
        ):
            self._cancel_task(qpu_mode=parts[1], task_id=parts[3])
            return

        self._send_error(404, f"No mock route for PUT {urlparse(self.path).path}")

    def _create_task(self, qpu_mode: str) -> None:
        body = self._read_json()
        if body is None:
            return

        if "programs" not in body or "subtasks" not in body:
            self._send_error(
                400,
                "Local mock only supports inline TaskDefinition bodies with "
                "'programs' and 'subtasks'.",
            )
            return

        task = self.state.create_task(qpu_mode=qpu_mode, definition=body)
        self._send_json(201, task.as_task_response())

    def _list_tasks(self, qpu_mode: str) -> None:
        query = parse_qs(urlparse(self.path).query)
        page = parse_positive_int(query.get("page", [None])[0], 0)
        size = parse_positive_int(query.get("size", [None])[0], 10)

        tasks = [
            task.as_task_response()
            for task in self.state.tasks.values()
            if task.qpu_mode == qpu_mode
        ]
        start = page * size
        end = start + size
        self._send_json(
            200,
            {
                "elements": tasks[start:end],
                "page": page,
                "size": size,
                "total": len(tasks),
            },
        )

    def _get_task(self, qpu_mode: str, task_id: str) -> None:
        task = self.state.tasks.get(task_id)
        if task is None or task.qpu_mode != qpu_mode:
            self._send_error(404, f"Unknown task id: {task_id}")
            return
        self._send_json(200, task.as_task_response())

    def _cancel_task(self, qpu_mode: str, task_id: str) -> None:
        task = self.state.tasks.get(task_id)
        if task is None or task.qpu_mode != qpu_mode:
            self._send_error(404, f"Unknown task id: {task_id}")
            return
        task.task_status = "Cancelled"
        task.modified_date = utc_now()
        self._send_json(202, {"status_code": 202})

    def _get_results(self, qpu_mode: str, task_id: str) -> None:
        task = self.state.tasks.get(task_id)
        if task is None or task.qpu_mode != qpu_mode:
            self._send_error(404, f"Unknown task id: {task_id}")
            return

        query = parse_qs(urlparse(self.path).query)
        page = parse_positive_int(query.get("page", [None])[0], 0)
        size = parse_positive_int(query.get("size", [None])[0], 10)
        shots_page = parse_positive_int(query.get("shots_page", [None])[0], 0)
        shots_size = parse_positive_int(query.get("shots_size", [None])[0], 100)

        subtasks = task.result_subtasks(shots_page=shots_page, shots_size=shots_size)
        start = page * size
        end = start + size
        page_subtasks = subtasks[start:end]

        elements = []
        if page_subtasks:
            elements.append(
                {
                    "task_id": task.task_id,
                    "task_status": task.current_status(),
                    "subtasks": page_subtasks,
                }
            )

        self._send_json(
            200,
            {
                "elements": elements,
                "page": page,
                "size": size,
                "total": len(subtasks),
            },
        )


class LocalQlamServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        state: LocalQlamState,
        quiet: bool,
    ):
        super().__init__(server_address, handler_class)
        self.state = state
        self.quiet = quiet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--bitstring-width",
        type=int,
        default=160,
        help=(
            "Number of SLM-site bits returned for each mock shot result. "
            "The Gemini logical demo expects 160 physical SLM sites."
        ),
    )
    parser.add_argument(
        "--require-auth",
        action="store_true",
        help="Reject requests that do not include an Authorization header.",
    )
    parser.add_argument(
        "--execution-latency-seconds",
        type=float,
        default=0.0,
        help=(
            "Keep newly submitted tasks in PayloadProcessing for this many "
            "seconds before reporting Completed. Useful for exercising SDK "
            "polling and timeout behavior."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress request logs.")
    args = parser.parse_args()

    state = LocalQlamState(
        bitstring_width=max(args.bitstring_width, 1),
        require_auth=args.require_auth,
        execution_latency_seconds=max(args.execution_latency_seconds, 0.0),
    )
    server = LocalQlamServer(
        (args.host, args.port),
        LocalQlamHandler,
        state=state,
        quiet=args.quiet,
    )

    print(f"Local QLAM mock listening at http://{args.host}:{args.port}")
    if state.execution_latency_seconds > 0:
        print(
            "Mock tasks will report PayloadProcessing for "
            f"{state.execution_latency_seconds}s before Completed."
        )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local QLAM mock.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
