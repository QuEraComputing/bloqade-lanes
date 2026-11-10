# Proposal: shared structural validation in Kirin

Consider upstreaming these compiler utilities from `bloqade.lanes.validation`
into `kirin.validation`:

- `recursion.py`: detect recursive or opaque calls before inlining/unrolling.
- `static_call.py`: reject unresolved calls where a fully expanded program is required.
- `call_site.py`: retain call-site origins so validation errors from inlined kernels
  can be attributed to their callers.

Their implementation depends only on Kirin and Python's standard library.
That is a useful starting point, not evidence that the public API or restrictions
are appropriate for every Kirin consumer. The Kirin code owner should review the
control-flow assumptions, error types, pass interfaces, source-location handling,
and integration with existing validation and inlining facilities.

The implementations and regression tests remain in the Bloqade projects until
that review and a Kirin release provide an agreed replacement. This proposal is
separate from the Gemini package split and does not block it. No upstream API or
release commitment is implied.
