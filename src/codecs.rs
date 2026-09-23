use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) enum MonolithicSimulationMode {
    StateVector,
    Mps,
    Stabilizer,
}

pub(crate) enum DistributedSimulationMode {
    PBlock,
}

pub(crate) struct MpsOptions {
    pub(crate) max_bond_dimension: Option<usize>,
    pub(crate) truncation_threshold: f64,
    pub(crate) max_discarded_weight: f64,
    pub(crate) max_memory_mb: usize,
    pub(crate) max_parallel_shots: usize,
    pub(crate) sample_terminal: bool,
}

pub(crate) struct StatevectorOptions {
    pub(crate) max_memory_mb: usize,
    pub(crate) max_parallel_shots: Option<usize>,
    pub(crate) sample_terminal: bool,
}

pub(crate) fn parse_statevector_options(
    options: Option<&Bound<PyDict>>,
) -> PyResult<StatevectorOptions> {
    let mut result = StatevectorOptions {
        max_memory_mb: 1024,
        max_parallel_shots: None,
        sample_terminal: true,
    };
    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "max_memory_mb" => result.max_memory_mb = value.extract()?,
                "max_parallel_shots" => result.max_parallel_shots = value.extract()?,
                "sample_terminal" => result.sample_terminal = value.extract()?,
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unsupported statevector option {key:?}"
                    )))
                }
            }
        }
    }
    Ok(result)
}

pub(crate) fn parse_monolithic_mode(mode: &str) -> PyResult<MonolithicSimulationMode> {
    match mode.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "state_vector" | "statevector" | "sv" => Ok(MonolithicSimulationMode::StateVector),
        "mps" | "matrix_product_state" => Ok(MonolithicSimulationMode::Mps),
        "stabilizer" | "stab" => Ok(MonolithicSimulationMode::Stabilizer),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported monolithic simulation mode {other:?}; expected 'state_vector' or 'mps'"
        ))),
    }
}

pub(crate) fn parse_distributed_mode(mode: &str) -> PyResult<DistributedSimulationMode> {
    match mode.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "p_block" => Ok(DistributedSimulationMode::PBlock),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported distributed simulation mode {other:?}; expected 'p_block'"
        ))),
    }
}

pub(crate) fn parse_mps_options(options: Option<&Bound<PyDict>>) -> PyResult<MpsOptions> {
    let mut max_bond_dimension = None;
    let mut truncation_threshold: f64 = 1e-12;
    let mut max_discarded_weight = 0.0;
    let mut max_memory_mb = 1024;
    let mut max_parallel_shots = 1;
    let mut sample_terminal = true;

    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "max_discarded_weight" => max_discarded_weight = value.extract()?,
                "max_memory_mb" => max_memory_mb = value.extract()?,
                "max_parallel_shots" => max_parallel_shots = value.extract()?,
                "sample_terminal" => sample_terminal = value.extract()?,
                "max_bond_dimension" => {
                    max_bond_dimension = parse_max_bond_dimension(&value)?;
                }
                "truncation_threshold" => {
                    truncation_threshold = parse_truncation_threshold(&value)?;
                }
                other => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unsupported MPS option {other:?}"
                    )));
                }
            }
        }
    }

    Ok(MpsOptions {
        max_bond_dimension,
        truncation_threshold,
        max_discarded_weight,
        max_memory_mb,
        max_parallel_shots,
        sample_terminal,
    })
}

fn parse_max_bond_dimension(value: &Bound<PyAny>) -> PyResult<Option<usize>> {
    if value.is_none() {
        return Ok(None);
    }

    let max_bond: usize = value.extract()?;
    if max_bond == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_bond_dimension must be positive or None",
        ));
    }
    Ok(Some(max_bond))
}

fn parse_truncation_threshold(value: &Bound<PyAny>) -> PyResult<f64> {
    let truncation_threshold: f64 = value.extract()?;
    if !truncation_threshold.is_finite() || truncation_threshold < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "truncation_threshold must be a finite non-negative float",
        ));
    }
    Ok(truncation_threshold)
}

pub(crate) struct PBlockOptions {
    pub max_memory_mb: usize,
    pub max_block_qubits: Option<usize>,
    pub max_parallel_shots: usize,
    pub sample_terminal: bool,
    pub split_separable: bool,
    pub max_split_qubits: usize,
    pub separation_tolerance: f64,
}
pub(crate) fn parse_pblock_options(options: Option<&Bound<PyDict>>) -> PyResult<PBlockOptions> {
    let mut out = PBlockOptions {
        max_memory_mb: 1024,
        max_block_qubits: None,
        max_parallel_shots: 1,
        sample_terminal: true,
        split_separable: false,
        max_split_qubits: 12,
        separation_tolerance: 0.0,
    };
    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "max_memory_mb" => out.max_memory_mb = value.extract()?,
                "max_block_qubits" => out.max_block_qubits = value.extract()?,
                "max_parallel_shots" => out.max_parallel_shots = value.extract()?,
                "sample_terminal" => out.sample_terminal = value.extract()?,
                "split_separable" => out.split_separable = value.extract()?,
                "max_split_qubits" => out.max_split_qubits = value.extract()?,
                "separation_tolerance" => out.separation_tolerance = value.extract()?,
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unsupported P-block option {key:?}"
                    )))
                }
            }
        }
    }
    Ok(out)
}

pub(crate) struct StabilizerOptions {
    pub max_memory_mb: usize,
    pub max_parallel_shots: usize,
    pub sample_terminal: bool,
    pub clifford_tolerance: f64,
}
pub(crate) fn parse_stabilizer_options(
    options: Option<&Bound<PyDict>>,
) -> PyResult<StabilizerOptions> {
    let mut out = StabilizerOptions {
        max_memory_mb: 1024,
        max_parallel_shots: 1,
        sample_terminal: true,
        clifford_tolerance: 0.0,
    };
    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "max_memory_mb" => out.max_memory_mb = value.extract()?,
                "max_parallel_shots" => out.max_parallel_shots = value.extract()?,
                "sample_terminal" => out.sample_terminal = value.extract()?,
                "clifford_tolerance" => out.clifford_tolerance = value.extract()?,
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unsupported stabilizer option {key:?}"
                    )))
                }
            }
        }
    }
    Ok(out)
}
