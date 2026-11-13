# Ownership

`bloqade.lanes` owns the compiler — dialects, rewrites, analysis.
`bloqade.gemini` owns the product — interfaces, simulation, user flow.
Gemini uses lanes to implement those; lanes owns the infrastructure that makes it possible.

Do not import `bloqade.gemini` from lanes, including inside functions or under
`TYPE_CHECKING`. Every method table a lanes analysis, validator, or pipeline needs lives in
lanes and registers when the module that owns it loads: dialect-level tables in
the dialect package, validation tables beside their analysis. Compilation must
not depend on importing a frontend first. Machine-specific architecture targets may live in lanes.
