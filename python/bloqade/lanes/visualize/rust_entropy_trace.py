"""Utilities for parsing Rust entropy-trace payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RustEntropyTraceStep:
    event: str
    node_id: int
    parent_node_id: int | None
    depth: int
    entropy: int
    unresolved_count: int
    moveset: list[tuple[int, int, int, int, int, int]] | None
    candidate_movesets: list[list[tuple[int, int, int, int, int, int]]]
    candidate_index: int | None
    reason: str | None
    state_seen_node_id: int | None
    no_valid_moves_qubit: int | None
    trigger_node_id: int | None
    configuration: list[tuple[int, int, int, int]]
    parent_configuration: list[tuple[int, int, int, int]] | None
    moveset_score: float | None
    best_buffer_node_ids: list[int] | None


@dataclass(frozen=True)
class RustEntropyTrace:
    root_node_id: int
    best_buffer_size: int
    steps: list[RustEntropyTraceStep]


def _as_lane_tuple(
    lane: list[int] | tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    direction, move_type, zone_id, word_id, site_id, bus_id = lane
    return (
        int(direction),
        int(move_type),
        int(zone_id),
        int(word_id),
        int(site_id),
        int(bus_id),
    )


def _as_loc_tuple(
    loc: list[int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    qid, zone_id, word_id, site_id = loc
    return (int(qid), int(zone_id), int(word_id), int(site_id))


def load_rust_entropy_trace(payload: str) -> RustEntropyTrace:
    raw = json.loads(payload)
    steps: list[RustEntropyTraceStep] = []
    for item in raw.get("steps", []):
        steps.append(
            RustEntropyTraceStep(
                event=str(item["event"]),
                node_id=int(item["node_id"]),
                parent_node_id=(
                    None
                    if item.get("parent_node_id") is None
                    else int(item["parent_node_id"])
                ),
                depth=int(item["depth"]),
                entropy=int(item["entropy"]),
                unresolved_count=int(item["unresolved_count"]),
                moveset=(
                    None
                    if item.get("moveset") is None
                    else [_as_lane_tuple(lane) for lane in item["moveset"]]
                ),
                candidate_movesets=[
                    [_as_lane_tuple(lane) for lane in candidate]
                    for candidate in item.get("candidate_movesets", [])
                ],
                candidate_index=(
                    None
                    if item.get("candidate_index") is None
                    else int(item["candidate_index"])
                ),
                reason=None if item.get("reason") is None else str(item["reason"]),
                state_seen_node_id=(
                    None
                    if item.get("state_seen_node_id") is None
                    else int(item["state_seen_node_id"])
                ),
                no_valid_moves_qubit=(
                    None
                    if item.get("no_valid_moves_qubit") is None
                    else int(item["no_valid_moves_qubit"])
                ),
                trigger_node_id=(
                    None
                    if item.get("trigger_node_id") is None
                    else int(item["trigger_node_id"])
                ),
                configuration=[
                    _as_loc_tuple(loc) for loc in item.get("configuration", [])
                ],
                parent_configuration=(
                    None
                    if item.get("parent_configuration") is None
                    else [_as_loc_tuple(loc) for loc in item["parent_configuration"]]
                ),
                moveset_score=(
                    None
                    if item.get("moveset_score") is None
                    else float(item["moveset_score"])
                ),
                best_buffer_node_ids=(
                    None
                    if item.get("best_buffer_node_ids") is None
                    else [int(node_id) for node_id in item["best_buffer_node_ids"]]
                ),
            )
        )

    return RustEntropyTrace(
        root_node_id=int(raw.get("root_node_id", 0)),
        best_buffer_size=int(raw.get("best_buffer_size", 0)),
        steps=steps,
    )
