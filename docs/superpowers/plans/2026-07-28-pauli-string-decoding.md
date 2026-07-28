# Pauli String Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gemini decoding's string-keyed tomography-basis dictionaries with a canonical, immutable Pauli-string key type.

**Architecture:** Add a Lanes-level `PauliString` value object and a `PauliMapping` read-only mapping that normalizes supported user inputs at construction and lookup. Migrate the existing single-qubit X/Y/Z workflow to use canonical keys without expanding it into arbitrary multi-qubit tomography.

**Tech Stack:** Python 3.10, dataclasses, `collections.abc.Mapping`, pytest, pyright.

---

### Task 1: Canonical Pauli value objects

**Files:**
- Create: `python/bloqade/lanes/pauli.py`
- Create: `python/bloqade/gemini/decoding/pauli.py` (compatibility re-export)
- Test: `python/tests/gemini/decoding/test_pauli.py`

- [x] Write failing tests for dense and sparse Pauli-string coercion, validation, equality, and hashability.
- [x] Implement `Pauli` and immutable, canonical `PauliString`.
- [x] Run `uv run pytest python/tests/gemini/decoding/test_pauli.py -q`.

### Task 2: Coercing Pauli mapping

**Files:**
- Modify: `python/bloqade/lanes/pauli.py`
- Test: `python/tests/gemini/decoding/test_pauli.py`

- [x] Write failing tests for normalized construction, lookup, containment, duplicate detection, and canonical key iteration.
- [x] Implement immutable `PauliMapping` backed by canonical `PauliString` keys.
- [x] Run the focused test file.

### Task 3: Migrate the tomography workflow

**Files:**
- Modify: `python/bloqade/gemini/decoding/__init__.py`
- Modify: `python/bloqade/gemini/decoding/kernels.py`
- Modify: `python/bloqade/gemini/decoding/msd.py`
- Modify: `python/bloqade/gemini/decoding/experiments.py`
- Modify: `python/bloqade/gemini/decoding/postselection.py`
- Modify: `python/bloqade/gemini/decoding/tomography.py`
- Test: `python/tests/gemini/decoding/test_msd_utils.py`

- [x] Write failing integration tests showing that the default tomography keys are `PauliString` values while string lookup remains convenient.
- [x] Use `PauliMapping` at the tomography and postselection boundaries.
- [x] Preserve the current single-qubit X/Y/Z validation in `TomographyResult`.
- [x] Run affected decoding tests, formatting, and pyright.
