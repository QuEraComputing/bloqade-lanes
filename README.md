# Welcome to Bloqade Lanes -- A component package of QuEra's Neutral Atom SDK

[![CI](https://github.com/QuEraComputing/bloqade-lanes/actions/workflows/ci.yml/badge.svg)](https://github.com/QuEraComputing/bloqade-circuit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuEraComputing/bloqade-lanes/graph/badge.svg?token=BpHsAYuzdo)](https://codecov.io/gh/QuEraComputing/bloqade-circuit)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/bloqade-lanes.svg?color=%2334D058)](https://pypi.org/project/bloqade-circuit)
[![Documentation](https://img.shields.io/badge/Documentation-6437FF)](https://bloqade.quera.com/)

Bloqade is a Python SDK for neutral atom quantum computing. It provides a set of embedded domain-specific languages (eDSLs) for programming neutral atom quantum computers. Bloqade is designed to be a high-level, user-friendly SDK that abstracts away the complexities of neutral atom quantum computing, allowing users to focus on developing quantum algorithms and compilation strategies for neutral atom quantum computers.

Bloqade-lanes provides the core components for compiling neutral atom quantum circuit programs down to moves. It focuses on the physical layout and movement of atoms along fixed lanes in a neutral atom quantum processor.

> [!IMPORTANT]
>
> This project is in the early stage of development. API and features are subject to change.

## Installation

### Install via `uv` (Recommended)

```py
uv add bloqade-lanes
```

## Documentation

The documentation is available at [https://bloqade.quera.com/latest/](https://bloqade.quera.com/latest/). We are at an early stage of completing the documentation with more details and examples, so comments and contributions are most welcome!

Proposal for the roadmap and feature requests are welcome!

## License

Apache License 2.0 with LLVM Exceptions
