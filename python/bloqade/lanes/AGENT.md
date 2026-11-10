# Ownership

`bloqade.lanes` owns the compiler — dialects, rewrites, analysis.
`bloqade.gemini` owns the product — interfaces, simulation, user flow.
Gemini uses lanes to implement those; lanes owns the infrastructure that makes it possible.

Do not import `bloqade.gemini` from lanes, including inside functions or under
`TYPE_CHECKING`. Frontend-specific method tables are supplied by the caller; Gemini kernel
groups load Gemini tables before constructing groups and running analyses.
Validation passes and analysis definitions stay in lanes. Machine-specific architecture targets may live in lanes.
