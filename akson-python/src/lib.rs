use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::Tensor;
use akson::ContinousFiniteLTISystem;
use akson::DiscreteFiniteLTISystem;
use akson::DiscretePIDRegulator;
use akson::DiscreteFeedbackSystem;
use akson::ContinousFeedbackSystem;
use akson::Regulator;
use akson::DiscreteSystem;

#[pyclass(dict, module = "rust", name = "ContinousFiniteLTISystem")]
pub struct PyContinousFiniteLTISystem(ContinousFiniteLTISystem);

#[pymethods]
impl PyContinousFiniteLTISystem {
    #[new]
    #[pyo3(signature = (state_matrix_a, input_matrix_b, output_matrix_c, feedthrough_matrix_d, initial_state))]
    fn new(
        _py: Python<'_>,
        state_matrix_a: &Bound<'_, PyAny>,
        input_matrix_b: &Bound<'_, PyAny>,
        output_matrix_c: &Bound<'_, PyAny>,
        feedthrough_matrix_d: &Bound<'_, PyAny>,
        initial_state: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self(
            ContinousFiniteLTISystem::new(
                state_matrix_a.extract::<PyTensor>().unwrap().0,
                input_matrix_b.extract::<PyTensor>().unwrap().0,
                output_matrix_c.extract::<PyTensor>().unwrap().0,
                feedthrough_matrix_d.extract::<PyTensor>().unwrap().0,
                initial_state.extract::<PyTensor>().unwrap().0,
            )
            .unwrap(),
        ))
    }

    #[staticmethod]
    fn from_tf(
        _py: Python<'_>,
        numerator: &Bound<'_, PyAny>,
        denominator: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let numerator_tensor = numerator.extract::<PyTensor>()?.0;
        let denominator_tensor = denominator.extract::<PyTensor>()?.0;

        let continous_system = ContinousFiniteLTISystem::from_tf(numerator_tensor, denominator_tensor)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

        Ok(Self(continous_system))
    }

    fn discretize(&self, _py: Python<'_>, t: f64) -> PyResult<PyDiscreteFiniteLTISystem> {
        Ok(PyDiscreteFiniteLTISystem(self.0.discretize(t)))
    }

    fn step_response(&self, duration: f64, step_size: f64) -> PyResult<(PyTensor, PyTensor)> {
        self.0.step_response(duration, step_size)
            .map(|(t, y)| (PyTensor(t), PyTensor(y)))
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass(dict, module = "rust", name = "DiscreteFiniteLTISystem")]
pub struct PyDiscreteFiniteLTISystem(DiscreteFiniteLTISystem);

#[pymethods]
impl PyDiscreteFiniteLTISystem {
    #[new]
    #[pyo3(signature = (state_matrix_a, input_matrix_b, output_matrix_c, feedthrough_matrix_d, initial_state))]
    fn new(
        _py: Python<'_>,
        state_matrix_a: &Bound<'_, PyAny>,
        input_matrix_b: &Bound<'_, PyAny>,
        output_matrix_c: &Bound<'_, PyAny>,
        feedthrough_matrix_d: &Bound<'_, PyAny>,
        initial_state: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self(
            DiscreteFiniteLTISystem::new(
                state_matrix_a.extract::<PyTensor>().unwrap().0,
                input_matrix_b.extract::<PyTensor>().unwrap().0,
                output_matrix_c.extract::<PyTensor>().unwrap().0,
                feedthrough_matrix_d.extract::<PyTensor>().unwrap().0,
                initial_state.extract::<PyTensor>().unwrap().0,
            )
            .unwrap(),
        ))
    }

    fn signal_response(&self, u: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let u = u.extract::<PyTensor>()?.0;
        let y = self.0.signal_response(&u)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        Ok(pyo3_tch::PyTensor(y))
    }

    fn step_response(&self, len: i64) -> PyResult<PyTensor> {
        let y = self.0.step_response(len)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        Ok(pyo3_tch::PyTensor(y))
    }
}

#[pyclass(dict, module = "rust", name = "DiscretePIDRegulator")]
pub struct PyDiscretePIDRegulator(DiscretePIDRegulator);

#[pymethods]
impl PyDiscretePIDRegulator {
    #[new]
    fn new(k: f64, t_i: f64, t_d: f64, t: f64) -> Self {
        Self(DiscretePIDRegulator::new(k, t_i, t_d, t))
    }
}

#[pyclass(unsendable, dict, module = "rust", name = "DiscreteFeedbackSystem")]
pub struct PyDiscreteFeedbackSystem(DiscreteFeedbackSystem);

#[pymethods]
impl PyDiscreteFeedbackSystem {
    #[new]
    fn new(
        regulator: &PyDiscretePIDRegulator,
        system: &PyDiscreteFiniteLTISystem,
    ) -> Self {
        let rust_regulator = Box::new(regulator.0.clone()) as Box<dyn Regulator + Send>;
        let rust_system = Box::new(system.0.clone()) as Box<dyn DiscreteSystem + Send>;
        Self(DiscreteFeedbackSystem::new(
            rust_regulator,
            rust_system
        ))
    }

    fn step(&mut self, reference: f64) -> PyResult<PyTensor> {
        let result = self.0.step(&Tensor::from(reference)).unwrap();
        Ok(PyTensor(result))
    }
}

#[pyclass(unsendable, dict, module = "rust", name = "ContinousFeedbackSystem")]
pub struct PyContinousFeedbackSystem(ContinousFeedbackSystem);

#[pymethods]
impl PyContinousFeedbackSystem {
    #[new]
    fn new(
        regulator: &PyDiscretePIDRegulator,
        system: &PyContinousFiniteLTISystem,
        time_step: f64,
    ) -> PyResult<Self> {
        let rust_regulator = Box::new(regulator.0.clone())
            as Box<dyn Regulator + Send>;

        let rust_system = system.0.clone();

        let feedback = ContinousFeedbackSystem::new(
            rust_regulator,
            rust_system,
            time_step
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self(feedback))
    }

    fn step(&mut self, reference: PyTensor, step_size: Option<PyTensor>) -> PyResult<(PyTensor, PyTensor)> {
        let output = self.0.step(&reference.0, step_size.map(|t| t.0))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((PyTensor(output.0), PyTensor(output.1)))
    }
}

#[pymodule]
#[pyo3(name = "rust")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContinousFiniteLTISystem>()?;
    m.add_class::<PyDiscreteFiniteLTISystem>()?;
    m.add_class::<PyDiscretePIDRegulator>()?;
    m.add_class::<PyDiscreteFeedbackSystem>()?;
    m.add_class::<PyContinousFeedbackSystem>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
