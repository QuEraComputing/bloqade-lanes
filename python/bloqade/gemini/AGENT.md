# Ownership

`bloqade.lanes` owns the compiler — dialects, rewrites, analysis.
`bloqade.gemini` owns the product — interfaces, simulation, user flow.
Gemini uses lanes to implement those; lanes owns the infrastructure that makes it possible.

Import compiler definitions from lanes. Keep product kernel decorators, device
policies, simulation, and decoding here. A deferred lanes import must have a
comment explaining a real optional-dependency or type-only requirement;
working around a lanes-to-Gemini cycle is not acceptable.

Method tables that lanes analyses and pipelines need live in lanes and register
when their lanes module loads; do not duplicate or re-register them here. Keep
Gemini-only policy tables (for example the device-submission extension
rejection) in Gemini.
