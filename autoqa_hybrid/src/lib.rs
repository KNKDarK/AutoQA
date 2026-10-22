//! PyO3 bindings and Python module entrypoint for the AutoQA native core.
//!
//! Built by maturin into the Python package `autoqa._native` (see
//! `pyproject.toml` → `tool.maturin.module-name`).
mod dom_hash;
mod vision_diff;

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// High level module metadata surfaced to Python.
#[pyclass]
#[derive(Clone)]
pub struct NativeInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub version: String,
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("_native_info", NativeInfo {
        name: "autoqa-native-core".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })?;

    vision_diff::register(m)?;
    dom_hash::register(m)?;
    Ok(())
}
