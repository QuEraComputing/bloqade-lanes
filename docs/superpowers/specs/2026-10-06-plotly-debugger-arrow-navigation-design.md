# Plotly Debugger Arrow Navigation Design

## Goal

Allow users to step through an interactive Plotly move-program debugger with
the four arrow keys, matching the previous/next behavior of the Matplotlib
debuggers.

## Behavior

The Plotly container is focusable only when the debugger is interactive and
has at least two frames. Once it has focus:

- Left Arrow and Up Arrow move to the previous step.
- Right Arrow and Down Arrow move to the next step.
- Navigation clamps at the first and last steps.
- The handler prevents the browser's default scrolling behavior.

The handler ignores key events whose target is a nested control, such as a bus
selector checkbox. Noninteractive/autoplay figures receive no keyboard
navigation support.

## Architecture

Extend the packaged `_arch_interactive.js` controller rather than adding a
second debugger-specific script. That controller already reads
`bloqadePlotlyDebugger.frameNames`, handles slider synchronization, and jumps
to a frame for executed-circuit gate clicks. Arrow navigation reuses its
`jumpToDebuggerStep` path, keeping frames, slider state, circuit highlighting,
and move-path overlays synchronized.

Track the current debugger frame in the controller, updating it for direct
animation events and keyboard-initiated jumps. The event listener is registered
only when the interactive slider exists and there are multiple frames. Set the
container's `tabindex` so it can receive focus, and handle a key only when the
event target is that container itself. That explicit focus guard prevents a
focused nested Plotly control from bubbling an arrow key into debugger
navigation.

## Tests

Extend the Node-based controller fixture to capture `keydown` listeners and
`Plotly.animate` calls. Cover all four arrow keys, forward/backward clamping,
default prevention, focusability, and nested-control event exclusion (including
that the nested event is not default-prevented). Retain the existing
Python-level checks that the debugger exports the controller and its frame
metadata.
