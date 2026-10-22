# AutoQA Hybrid — Python Core + Rust Native Engine

```text
┌──────────────────────────────────────────────┐
│              Python Core Layer               │
│  - Agent Loop & State Management             │
│  - Gemini Vision API Decision Engine         │
│  - Integration Connectors (GitHub / Figma)   │
└──────────────────────┬───────────────────────┘
                       │ PyO3 (maturin)
                       ▼
┌──────────────────────────────────────────────┐
│              Rust Native Core                │
│  - Parallel Visual Diffing (rayon rows)      │
│  - DOM Token Hashing (FNV-1a 64)             │
│  - LCS Mutation Delta Engine                 │
└──────────────────────────────────────────────┘
```

## Building

Requires: Python 3.11+, Rust 1.75+, maturin (`pip install maturin`).

```bash
cd autoqa_hybrid
# Run Rust unit tests
cargo test

# Build + install the extension module & python package into the venv
VIRTUAL_ENV=/path/to/venv maturin develop

# Run the Python test suite
pytest tests -q
```

## Native API

| Function | Purpose |
| :--- | :--- |
| `native.diff_screenshots(before, after, w, h, threshold)` | Row-parallel RGBA pixel diff → count, ratio, bbox |
| `native.similarity_ratio(before, after, threshold)` | Fast similarity ratio |
| `native.hash_tokens(tokens)` | Stable FNV-1a 64 structural hash of a DOM token stream |
| `native.dom_delta(before, after)` | LCS mutation delta → list of keep/replace/remove/insert ops |

The Python layer (`autoqa.agent.AutoQAAgent`) records `dom_hash` on every
captured step and annotates subsequent steps with `dom_delta` mutations.
