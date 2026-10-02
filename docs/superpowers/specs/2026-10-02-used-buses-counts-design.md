# Used-Buses Event Counts Design

## Goal

Report how often each transport bus is used by a compiled Move kernel, in
addition to preserving its bus category and zone identity.

## Public API

`get_used_buses(move_kernel)` returns:

```python
UsedBuses(
    zone={zone_bus_id: event_count},
    word={(zone_id, word_bus_id): event_count},
    site={(zone_id, site_bus_id): event_count},
)
```

The result fields are ordinary dictionaries. Zone-bus IDs are global; word and
site bus IDs are scoped to their zone and therefore use `(zone_id, bus_id)` as
their keys. The helper remains architecture-independent and accepts no
`ArchSpec`.

## Counting Semantics

Each `move.Move` statement is one move event. A bus is counted once if one or
more lanes on that bus occur in that event. Repeated lanes, different
directions, or multiple source locations on the same bus within one event do
not increase that event's count. The same bus in a later move event increments
its count again.

The helper returns dictionaries ordered by sorted keys, so their notebook
representation is deterministic.

## Implementation and Tests

The implementation scans `move.Move` statements, builds a per-event set for
each bus category, and increments aggregate counters from those sets. Focused
tests cover zone qualification, deduplication within an event, repeated use in
later events, sorted output, and a kernel with no moves.
