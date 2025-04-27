use tch::{IndexOp, Tensor};
use thiserror::Error;
use anyhow::anyhow;

/// Errors that can occur in ContinousFiniteLTISystem.
#[derive(Error, Debug)]
pub enum ContinousFiniteLTISystemError {
    /// Occurs when the rank of a tensor does not match epxpectations.
    #[error("Rank mismatch: Expected `{object}` to have rank {expected}, got {actual}")]
    RankMismatch {
        expected: usize,
        actual: usize,
        object: String,
    },
    /// Occurs when the shape of a tensor does not match expectations.
    #[error("Shape mismatch: Expected `{object}` to have shape {:?}, got {:?}", *expected, *actual)]
    ShapeMismatch {
        expected: Box<[i64]>,
        actual: Box<[i64]>,
        object: String,
    },
    /// Occurs when invalid argument is passed to a method
    #[error("Invalid argument! {explanation}")]
    InvalidArgument { explanation: String },
}

/// Errors that can occur in DiscreteFiniteLTISystem.
#[derive(Error, Debug)]
pub enum DiscreteFiniteLTISystemError {
    /// Occurs when the rank of a tensor does not match expectations.
    #[error("Rank mismatch: Expected `{object}` to have rank {expected}, got {actual}")]
    RankMismatch {
        expected: usize,
        actual: usize,
        object: String,
    },
    /// Occurs when the shape of a tensor does not match expectations.
    #[error("Shape mismatch: Expected `{object}` to have shape {:?}, got {:?}", *expected, *actual)]
    ShapeMismatch {
        expected: Box<[i64]>,
        actual: Box<[i64]>,
        object: String,
    },
    /// Occurs when invalid argument is passed to a method
    #[error("Invalid argument! {explanation}")]
    InvalidArgument { explanation: String },
}

/// Errors that can occur in DiscretePIDRegulator.
#[derive(Error, Debug)]
pub enum DiscretePIDRegulatorError {
    /// Occurs when the rank of a tensor does not match expectations.
    #[error("Rank mismatch: Expected `{object}` to have rank {expected}, got {actual}")]
    RankMismatch {
        expected: usize,
        actual: usize,
        object: String,
    },
    /// Occurs when the shape of a tensor does not match expectations.
    #[error("Shape mismatch: Expected `{object}` to have shape {:?}, got {:?}", *expected, *actual)]
    ShapeMismatch {
        expected: Box<[i64]>,
        actual: Box<[i64]>,
        object: String,
    },
}

/// A trait representing a discrete-time regulator.
pub trait Regulator: Send {
    ///
    /// # Arguments
    /// * y - Tensor containing past system outputs and current system output
    /// * reference - Reference trajectory
    ///
    /// # Returns
    /// `Result<Tensor>` containing computed system input or an error
    fn compute_control(
        &mut self,
        y: &Tensor,
        reference: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>>;
}

/// A trait representing a discrete-time system.
pub trait DiscreteSystem: Send {
    /// Computes response to signal `u` without modifying internal state
    ///
    /// # Arguments
    /// * u - Tensor containing input signal to the system
    ///
    /// # Returns
    /// `Result<Tensor>` containing output signal from the system
    fn signal_response(&self, u: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>>;

    /// Computes step response without modifying internal state
    ///
    /// # Arguments
    /// * len - Length of the step response to be computed
    ///
    /// # Returns
    /// `Result<Tensor>` containing the system step response
    fn step_response(&self, len: i64) -> Result<Tensor, Box<dyn std::error::Error>>;

    /// Simulates input to the system
    ///
    /// # Arguments
    /// * u - Tensor containing input signal to the system
    ///
    /// # Returns
    /// `Result<Tensor>` containing output signal from the system
    fn simulate_from_current(&mut self, u: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>>;

    /// Returns current output of the system
    fn current_output(&self) -> Tensor;

    /// Returns the kind of [`tch`] tensors in the system
    fn kind(&self) -> tch::Kind;

    /// Returns the [`tch`] device the system resides on
    fn device(&self) -> tch::Device;
}

/// Represents a continuous-time finite dimensional linear time-invariant (LTI) system.
///
/// The system can be created from:
/// - Direct state-space matrices (with [`ContinousFiniteLTISystem::new`])
/// - Coefficients of the proper rational transfer function (with [`ContinousFiniteLTISystem::from_tf`])
///
/// # Internal Representation
/// The system is internally represented in the state-space form:
/// $$
/// \begin{aligned}
/// \frac{dx(t)}{dt} &= A x(t) + B u(t) \quad \text{(State equation)} \\\\
/// y(t) &= C x(t) + D u(t) \quad \text{(Output equation)}
/// \end{aligned}
/// $$
/// Where:
/// - $x \in \mathbb{R}^n$: State vector
/// - $u \in \mathbb{R}$: Input vector
/// - $y \in \mathbb{R}$: Output vector
#[derive(Debug)]
pub struct ContinousFiniteLTISystem {
    /// State matrix A (nxn) governing system dynamics (Immutable)
    state_matrix_a: Tensor,
    /// Input matrix B (nx1) mapping input to state dynamics (Immutable)
    input_matrix_b: Tensor,
    /// Output matrix C (1xn) mapping state to output (Immutable)
    output_matrix_c: Tensor,
    /// Feedthrough matrix D (1x1) mapping input directly to output (Immutable)
    feedthrough_matrix_d: Tensor,
    /// Current state x (nx1) of the system (Mutable)
    state_x: Tensor,
}

/// Represents discrete-time finite dimensional linear time-invariant (LTI) system.
///
/// # Internal Representation
/// The system is internally represented in the state-space form:
/// $$
/// \begin{aligned}
/// x(k) &= Ax(k-1) + Bu(k-1) \quad &&\text{(State equation)} \\\\
/// y(k) &= Ax(k) + Bu(k) \quad &&\text{(Output equation)}
/// \end{aligned}
/// $$
/// Where:
/// - $x \in \mathbb{R}^n$: State vector
/// - $u \in \mathbb{R}$: Input vector
/// - $y \in \mathbb{R}$: Output vector
#[derive(Debug)]
pub struct DiscreteFiniteLTISystem {
    /// State matrix A (nxn) governing system dynamics (Immutable)
    state_matrix_a: Tensor,
    /// Input matrix B (nx1) mapping input to system dynamics (Immutable)
    input_matrix_b: Tensor,
    /// Output matrix C (1xn) mapping state to output (Immutable)
    output_matrix_c: Tensor,
    /// Feedthrough matrix D (1x1) mapping input directly to output (Immutable)
    feedthrough_matrix_d: Tensor,
    /// Current state x (nx1) of the system (Mutable)
    state_x: Tensor,
}

impl Clone for ContinousFiniteLTISystem {
    fn clone(&self) -> Self {
        Self {
            // Shallow clone of immutable matrices
            state_matrix_a: self.state_matrix_a.shallow_clone(),
            input_matrix_b: self.input_matrix_b.shallow_clone(),
            output_matrix_c: self.output_matrix_c.shallow_clone(),
            feedthrough_matrix_d: self.feedthrough_matrix_d.shallow_clone(),

            // Deep copy of mutable state (ensures independence)
            state_x: self.state_x.copy(),
        }
    }
}

impl Clone for DiscreteFiniteLTISystem {
    fn clone(&self) -> Self {
        Self {
            // Shallow clone of immutable matrices
            state_matrix_a: self.state_matrix_a.shallow_clone(),
            input_matrix_b: self.input_matrix_b.shallow_clone(),
            output_matrix_c: self.output_matrix_c.shallow_clone(),
            feedthrough_matrix_d: self.feedthrough_matrix_d.shallow_clone(),

            // Deep copy of mutable state (ensures independence)
            state_x: self.state_x.copy(),
        }
    }
}

/// Represents discrete-time PID regulator.
#[derive(Debug)]
pub struct DiscretePIDRegulator {
    /// Discrete PID coefficient r0
    r_0: f64,
    /// Discrete PID coefficient r1
    r_1: f64,
    /// Discrete PID coefficient r2
    r_2: f64,
    /// Value of e(k-1)
    e_km1: Option<Tensor>,
    /// Value of e(k-2)
    e_km2: Option<Tensor>,
    /// Previous value of the input signal
    u_km1: Option<Tensor>,
}

impl Clone for DiscretePIDRegulator {
    fn clone(&self) -> Self {
        Self {
            r_0: self.r_0,
            r_1: self.r_1,
            r_2: self.r_2,
            e_km1: match self.e_km1 {
                Some(ref e_km1) => Some(e_km1.shallow_clone()),
                None => None,
            },
            e_km2: match self.e_km2 {
                Some(ref e_km2) => Some(e_km2.shallow_clone()),
                None => None,
            },
            u_km1: match self.u_km1 {
                Some(ref u_km1) => Some(u_km1.shallow_clone()),
                None => None,
            },
        }
    }
}

/// Represents setpoint for [`DiscretePIDRegulator`] regulator.
pub struct DiscretePIDReference(pub f64);

/// Represents feedback system with discrete regulator and discrete controlled system.
pub struct DiscreteFeedbackSystem {
    /// Regulator for the system
    regulator: Box<dyn Regulator + Send>,
    /// Controlled system
    system: Box<dyn DiscreteSystem + Send>,
}

impl ContinousFiniteLTISystem {
    pub fn new(
        state_matrix_a: Tensor,
        input_matrix_b: Tensor,
        output_matrix_c: Tensor,
        feedthrough_matrix_d: Tensor,
        state_x: Tensor,
    ) -> Result<Self, ContinousFiniteLTISystemError> {
        let a_shape = state_matrix_a.size();
        let b_shape = input_matrix_b.size();
        let c_shape = output_matrix_c.size();
        let d_shape = feedthrough_matrix_d.size();
        let x_shape = state_x.size();

        // Validate tensor ranks
        for (object, actual_rank, expected_rank) in [
            ("state_matrix_a", a_shape.len(), 2),
            ("input_matrix_b", b_shape.len(), 2),
            ("output_matrix_c", c_shape.len(), 2),
            ("feedthrough_matrix_d", d_shape.len(), 2),
            ("state_x", x_shape.len(), 2),
        ] {
            if actual_rank != expected_rank {
                return Err(ContinousFiniteLTISystemError::RankMismatch {
                    expected: expected_rank,
                    actual: actual_rank,
                    object: object.into(),
                });
            }
        }

        // Validate tensor shapes
        let state_size = x_shape[0];
        for (object, actual_shape, expected_shape) in [
            ("state_matrix_a", a_shape, vec![state_size, state_size]),
            ("input_matrix_b", b_shape, vec![state_size, 1]),
            ("output_matrix_c", c_shape, vec![1, state_size]),
            ("feedthrough_matrix_d", d_shape, vec![1, 1]),
            ("state_x", x_shape, vec![state_size, 1]),
        ] {
            if actual_shape != expected_shape {
                return Err(ContinousFiniteLTISystemError::ShapeMismatch {
                    expected: expected_shape.into(),
                    actual: actual_shape.into(),
                    object: object.into(),
                });
            }
        }

        Ok(Self {
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        })
    }

    fn current_output(&self) -> Tensor {
        self.output_matrix_c.mm(&self.state_x)
    }

    /// Construct the system from transfer function coefficients.
    pub fn from_tf(
        numerator: Tensor,
        denominator: Tensor,
    ) -> Result<Self, ContinousFiniteLTISystemError> {
        // Check that numerator and denominator have the same type and they are on the same device
        if numerator.kind() != denominator.kind() ||
           numerator.device() != denominator.device() {
            return Err(ContinousFiniteLTISystemError::InvalidArgument {
                explanation: String::from("Numerator and denominator must be of the same kind and they must reside on the same device.")
            });
        }

        // Validate ranks of the input tensors
        for (name, actual_rank, expected_rank) in [
            ("numerator", numerator.size().len(), 1),
            ("denominator", denominator.size().len(), 1),
        ] {
            if actual_rank != expected_rank {
                return Err(ContinousFiniteLTISystemError::RankMismatch {
                    actual: actual_rank,
                    expected: expected_rank,
                    object: name.into(),
                });
            }
        }

        // Validate the length of the denominator coefficients tensor
        let denom_coefs_cnt = denominator.size()[0];
        if denom_coefs_cnt < 2 {
            return Err(ContinousFiniteLTISystemError::InvalidArgument {
                explanation: String::from(
                    "ContinousFiniteLTISystem can only be constructed from proper rational transfer functions.",
                ),
            });
        }

        // Validate the length of the numerator coefficients tensor
        let num_coefs_cnt = numerator.size()[0];
        if num_coefs_cnt >= denom_coefs_cnt {
            return Err(ContinousFiniteLTISystemError::InvalidArgument {
                explanation: String::from(
                    "ContinousFiniteLTISystem can only be constructed from proper rational transfer functions.",
                ),
            });
        }

        // Find index of the leading coefficient of the numerator
        let leading_idx_num = {
            let mut it = (0..num_coefs_cnt).rev();
            loop {
                let idx = match it.next() {
                    Some(idx) => idx,
                    None => break None,
                };

                let zero = tch::Tensor::from(0).to_kind(numerator.kind());
                if numerator.get(idx) != zero {
                    break Some(idx);
                }
            }
        };

        // Find index of the leading coefficient of the denominator
        let leading_idx_denom = {
            let mut it = (0..denom_coefs_cnt).rev();
            loop {
                let idx = match it.next() {
                    Some(idx) => idx,
                    None => break None,
                };

                let zero = tch::Tensor::from(0).to_kind(denominator.kind());
                if denominator.get(idx) != zero {
                    break Some(idx);
                }
            }
        };

        let (_leading_idx_num, leading_idx_denom) = match (leading_idx_num, leading_idx_denom) {
            (Some(leading_idx_num), Some(leading_idx_denom))
                if leading_idx_num < leading_idx_denom =>
            {
                (leading_idx_num, leading_idx_denom)
            }
            _ => {
                return Err(ContinousFiniteLTISystemError::InvalidArgument {
                    explanation: String::from(
                        "ContinousFiniteLTISystem can only be constructed from proper rational transfer functions.",
                    ),
                });
            }
        };

        let state_size = leading_idx_denom;
        let tensor_type = (numerator.kind(), numerator.device());

        let leading_denom = denominator.i(leading_idx_denom);
        let a_coefs = denominator / &leading_denom;
        let b_coefs = numerator / &leading_denom;

        let state_matrix_a = {
            let state_matrix_a = Tensor::zeros(
                [state_size, state_size],
                tensor_type
            );

            state_matrix_a.i((1.., ..state_size-1)).copy_(
                &Tensor::eye(
                    state_size-1,
                    tensor_type
                )
            );

            state_matrix_a.i((0, ..)).copy_(
                &-a_coefs.i(..state_size).flip(0)
            );

            state_matrix_a
        };

        let input_matrix_b = {
            let input_matrix_b = Tensor::zeros(
                [state_size, 1],
                tensor_type
            );
            let _ = input_matrix_b.i((0, 0)).copy_(
                &Tensor::ones(
                    [],
                    tensor_type
                )
            );

            input_matrix_b
        };

        let output_matrix_c = b_coefs
            .i(0..state_size)
            .flip(0)
            .reshape([1, -1]);

        let feedthrough_matrix_d = Tensor::zeros(
            [1, 1],
            tensor_type
        );

        let state_x = Tensor::zeros(
            [state_size, 1],
            tensor_type
        );

        Ok(Self {
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        })
    }

    /// Discretizes the system.
    ///
    /// For discretization, the following formulas are used:
    /// $$
    /// \begin{aligned}
    /// A_d &= e^{T \cdot A} \\\\
    /// B_d &= A^{-1}(A_d - I) B \\\\
    /// C_d &= C \\\\
    /// D_d &= D
    /// \end{aligned}
    /// $$
    /// Where $T$ is the discrete system time constant.
    pub fn discretize(&self, t: f64) -> DiscreteFiniteLTISystem {
        let discrete_a = (&self.state_matrix_a * t).linalg_matrix_exp();

        let adm1 = &discrete_a
            - tch::Tensor::eye(
                discrete_a.size()[0],
                (discrete_a.kind(), discrete_a.device()),
            );

        let discrete_b = self
            .state_matrix_a
            .inverse()
            .mm(&adm1)
            .mm(&self.input_matrix_b);

        DiscreteFiniteLTISystem {
            state_matrix_a: discrete_a,
            input_matrix_b: discrete_b,
            output_matrix_c: self.output_matrix_c.shallow_clone(),
            feedthrough_matrix_d: self.feedthrough_matrix_d.shallow_clone(),
            state_x: self.state_x.copy(),
        }
    }

    pub fn get_state_matrix_a(&self) -> &Tensor {
        &self.state_matrix_a
    }

    pub fn get_input_matrix_b(&self) -> &Tensor {
        &self.input_matrix_b
    }

    pub fn get_output_matrix_c(&self) -> &Tensor {
        &self.output_matrix_c
    }

    pub fn get_feedthrough_matrix_d(&self) -> &Tensor {
        &self.feedthrough_matrix_d
    }

    pub fn get_state_x(&self) -> &Tensor {
        &self.state_x
    }

    pub fn set_state_x(&mut self, new_state_x: Tensor) {
        self.state_x = new_state_x;
    }
}

impl DiscreteFiniteLTISystem {
    pub fn new(
        state_matrix_a: Tensor,
        input_matrix_b: Tensor,
        output_matrix_c: Tensor,
        feedthrough_matrix_d: Tensor,
        state_x: Tensor,
    ) -> Result<Self, DiscreteFiniteLTISystemError> {
        let a_shape = state_matrix_a.size();
        let b_shape = input_matrix_b.size();
        let c_shape = output_matrix_c.size();
        let d_shape = feedthrough_matrix_d.size();
        let x_shape = state_x.size();

        // Validate tensor ranks
        for (object, actual_rank, expected_rank) in [
            ("state_matrix_a", a_shape.len(), 2),
            ("input_matrix_b", b_shape.len(), 2),
            ("output_matrix_c", c_shape.len(), 2),
            ("feedthrough_matrix_d", d_shape.len(), 2),
            ("state_x", x_shape.len(), 2),
        ] {
            if actual_rank != expected_rank {
                return Err(DiscreteFiniteLTISystemError::RankMismatch {
                    expected: expected_rank,
                    actual: actual_rank,
                    object: object.into(),
                });
            }
        }

        // Validate tensor shapes
        let state_size = x_shape[0];
        for (object, actual_shape, expected_shape) in [
            ("state_matrix_a", a_shape, vec![state_size, state_size]),
            ("input_matrix_b", b_shape, vec![state_size, 1]),
            ("output_matrix_c", c_shape, vec![1, state_size]),
            ("feedthrough_matrix_d", d_shape, vec![1, 1]),
            ("state_x", x_shape, vec![state_size, 1]),
        ] {
            if actual_shape != expected_shape {
                return Err(DiscreteFiniteLTISystemError::ShapeMismatch {
                    expected: expected_shape.into(),
                    actual: actual_shape.into(),
                    object: object.into(),
                });
            }
        }

        Ok(Self {
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        })
    }

    pub fn get_state_matrix_a(&self) -> &Tensor {
        &self.state_matrix_a
    }

    pub fn get_input_matrix_b(&self) -> &Tensor {
        &self.input_matrix_b
    }

    pub fn get_output_matrix_c(&self) -> &Tensor {
        &self.output_matrix_c
    }

    pub fn get_feedthrough_matrix_d(&self) -> &Tensor {
        &self.feedthrough_matrix_d
    }

    pub fn get_state_x(&self) -> &Tensor {
        &self.state_x
    }

    pub fn set_state_x(&mut self, new_state_x: Tensor) {
        self.state_x = new_state_x
    }
}

impl DiscreteSystem for DiscreteFiniteLTISystem {
    fn signal_response(&self, u: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        self.clone().simulate_from_current(u)
    }

    fn step_response(&self, len: i64) -> Result<Tensor, Box<dyn std::error::Error>> {
        if len <= 0 {
            return Err(Box::new(DiscreteFiniteLTISystemError::InvalidArgument {
                explanation: String::from("Expected the value of `len` to be greater than zero."),
            }));
        }

        let mut x = self.state_x.copy();
        let y = Tensor::zeros([1, len], (x.kind(), x.device()));

        for i in 0..len {
            let y_k = self.output_matrix_c.matmul(&x) + &self.feedthrough_matrix_d;
            y.i((.., i..i + 1)).copy_(&y_k);
            x = self.state_matrix_a.matmul(&x) + &self.input_matrix_b;
        }

        Ok(y)
    }

    fn simulate_from_current(&mut self, u: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let u_shape = u.size();
        if u_shape.len() != 2 {
            return Err(Box::new(DiscreteFiniteLTISystemError::RankMismatch {
                expected: 2,
                actual: u_shape.len(),
                object: String::from("u"),
            }));
        }

        if u_shape[0] != 1 {
            return Err(Box::new(DiscreteFiniteLTISystemError::ShapeMismatch {
                expected: Box::new([1, -1]),
                actual: u_shape.into(),
                object: String::from("u"),
            }));
        }

        let num_steps = u_shape[1];
        let y = Tensor::zeros([1, num_steps], (self.state_x.kind(), self.state_x.device()));
        for i in 0..num_steps {
            let u_k = u.i((.., i..i + 1));
            let y_k =
                self.output_matrix_c.matmul(&self.state_x) + self.feedthrough_matrix_d.matmul(&u_k);
            y.i((.., i..i + 1)).copy_(&y_k);
            self.state_x =
                self.state_matrix_a.matmul(&self.state_x) + self.input_matrix_b.matmul(&u_k);
        }

        Ok(y)
    }

    fn current_output(&self) -> Tensor {
        self.output_matrix_c.mm(&self.state_x)
    }

    fn kind(&self) -> tch::Kind {
        self.state_x.kind()
    }

    fn device(&self) -> tch::Device {
        self.state_x.device()
    }
}

impl DiscretePIDRegulator {
    pub fn new(k: f64, t_i: f64, t_d: f64, t: f64) -> Self {
        Self {
            r_0: k * (1. + t / (2. * t_i) + t_d / t),
            r_1: k * (t / (2. * t_i) - 2. * t_d / t - 1.),
            r_2: k * t_d / t,
            e_km1: None,
            e_km2: None,
            u_km1: None,
        }
    }
}

impl Regulator for DiscretePIDRegulator {
    fn compute_control(
        &mut self,
        y: &Tensor,
        reference: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // TODO: Check that all dimensions of `y` are 1
        // TODO: Check that all dimensions of `reference` are 1

        let e = reference - y;

        let mut result = e.copy() * self.r_0;
        if let Some(ref e_km1) = self.e_km1 {
            result += e_km1 * self.r_1;
        }
        if let Some(ref e_km2) = self.e_km2 {
            result += e_km2 * self.r_2;
        }
        if let Some(ref u_km1) = self.u_km1 {
            result += u_km1;
        }

        self.e_km2 = match self.e_km1 {
            Some(ref e_km1) => Some(e_km1.shallow_clone()),
            None => None,
        };
        self.e_km1 = Some(e);
        self.u_km1 = Some(result.copy());

        Ok(result)
    }
}

impl DiscreteFeedbackSystem {
    pub fn new(
        // FIXME: This is temporary fix, this should be changed when more regulators are added
        regulator: Box<dyn Regulator + Send>,
        system: Box<dyn DiscreteSystem + Send>,
    ) -> Self {
        Self { regulator, system }
    }

    pub fn step(&mut self, reference: &Tensor) -> Result<Tensor, anyhow::Error> {
        let y = self.system.current_output();

        let u = self.regulator.compute_control(&y, reference).map_err(
            |e| anyhow!(format!("Failed to compute control. Error message: {}", e))
        )?;

        self.system.simulate_from_current(&u).map_err(
            |e| anyhow!(format!("Failed to feed input to the system. Error message: {}", e))
        )
    }
}

/// Errors that can occur in ContinousFeedbackSystem.
#[derive(Error, Debug)]
pub enum ContinousFeedbackSystemError {
    /// Occurs when the time step is non-positive.
    #[error("Time step must be positive, got {0}")]
    InvalidTimeStep(f64),
    /// Occurs when matrix operation fails
    #[error("Matrix operation error: {0}")]
    MatrixError(String),
}

/// Represents feedback system with discrete regulator and continuous controlled system.
pub struct ContinousFeedbackSystem {
    /// Discrete-time regulator
    regulator: Box<dyn Regulator + Send>,
    /// Continous system being controlled
    system: ContinousFiniteLTISystem,
    /// Discrete time step for control updates
    time_step: f64,
    /// Current simulation time
    current_time: f64,
}

impl ContinousFeedbackSystem {
    pub fn new(
        regulator: Box<dyn Regulator + Send>,
        system: ContinousFiniteLTISystem,
        time_step: f64,
    ) -> Result<Self, ContinousFeedbackSystemError> {
        if time_step <= 0.0 {
            return Err(ContinousFeedbackSystemError::InvalidTimeStep(time_step));
        }

        Ok(Self {
            regulator,
            system,
            time_step,
            current_time: 0.0,
        })
    }

    pub fn step(&mut self, reference: &Tensor) -> Result<Tensor, anyhow::Error> {
        // 1. Compute control input based on current output
        let y = self.system.current_output();
        let u = self.regulator.compute_control(&y, reference)
            .map_err(|e| anyhow!("Regulator error: {}", e))?
            .reshape(&[1, 1]);  // Ensure 1x1 shape

        let a = self.system.get_state_matrix_a();
        let a_inv = a.inverse(); // Compute inverse once for efficiency
        let b = self.system.get_input_matrix_b();
        let c = self.system.get_output_matrix_c();
        let d = self.system.get_feedthrough_matrix_d();
        let x0 = self.system.get_state_x().shallow_clone();
        let bu = b.matmul(&u);

        // 2. Generate samples during the time_step interval
        let num_samples = 10; // Number of internal samples per step
        let outputs = Tensor::zeros(&[1, num_samples + 1], (a.kind(), a.device()));
        let eye = Tensor::eye(a.size()[0], (a.kind(), a.device()));

        for i in 0..=num_samples {
            let tau = (i as f64 / num_samples as f64) * self.time_step;
            let a_tau = a * tau;
            let e_a_tau = a_tau.linalg_matrix_exp();
            let adm1_tau = &e_a_tau - &eye;
            let integral_term_tau = a_inv.matmul(&adm1_tau);
            let x_tau = e_a_tau.matmul(&x0) + integral_term_tau.matmul(&bu);
            let y_tau = c.matmul(&x_tau) + d.matmul(&u);
            outputs.narrow(1, i as i64, 1).copy_(&y_tau);
        }

        // 3. Update system state to the end of the time_step
        let a_dt = a * self.time_step;
        let e_a_dt = a_dt.linalg_matrix_exp();
        let adm1 = &e_a_dt - &eye;
        let integral_term = a_inv.matmul(&adm1);
        let x_next = e_a_dt.matmul(&x0) + integral_term.matmul(&bu);
        self.system.set_state_x(x_next);
        self.current_time += self.time_step;

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn construction_of_dflti_system_bad_rank0() {
        // Test state_matrix_a with incorrect rank (1 instead of 2)
        let state_matrix_a = Tensor::from_slice(&[1.0, 2.0]);
        let input_matrix_b = Tensor::from_slice(&[3.0]).reshape(&[1, 1]);
        let output_matrix_c = Tensor::from_slice(&[4.0]).reshape(&[1, 1]);
        let feedthrough_matrix_d = Tensor::from_slice(&[5.0]).reshape(&[1, 1]);
        let state_x = Tensor::from_slice(&[6.0]).reshape(&[1, 1]);

        let result = DiscreteFiniteLTISystem::new(
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        );

        match result {
            Err(DiscreteFiniteLTISystemError::RankMismatch {
                expected,
                actual,
                object,
            }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
                assert_eq!(object, "state_matrix_a");
            }
            _ => panic!("Expected RankMismatch error for state_matrix_a"),
        }
    }

    #[test]
    fn construction_of_dflti_system_bad_rank1() {
        // Test input_matrix_b with incorrect rank (1 instead of 2)
        let state_matrix_a = Tensor::eye(1, (tch::Kind::Double, tch::Device::Cpu));
        let input_matrix_b = Tensor::from_slice(&[3.0]);
        let output_matrix_c = Tensor::from_slice(&[4.0]).reshape(&[1, 1]);
        let feedthrough_matrix_d = Tensor::from_slice(&[5.0]).reshape(&[1, 1]);
        let state_x = Tensor::from_slice(&[6.0]).reshape(&[1, 1]);

        let result = DiscreteFiniteLTISystem::new(
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        );

        match result {
            Err(DiscreteFiniteLTISystemError::RankMismatch {
                expected,
                actual,
                object,
            }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
                assert_eq!(object, "input_matrix_b");
            }
            _ => panic!("Expected RankMismatch error for input_matrix_b"),
        }
    }

    #[test]
    fn construction_of_dflti_system_bad_rank2() {
        // Test output_matrix_c with incorrect rank (3 instead of 2)
        let state_matrix_a = Tensor::eye(1, (tch::Kind::Double, tch::Device::Cpu));
        let input_matrix_b = Tensor::from_slice(&[3.0]).reshape(&[1, 1]);
        let output_matrix_c = Tensor::from_slice(&[4.0]).reshape(&[1, 1, 1]);
        let feedthrough_matrix_d = Tensor::from_slice(&[5.0]).reshape(&[1, 1]);
        let state_x = Tensor::from_slice(&[6.0]).reshape(&[1, 1]);

        let result = DiscreteFiniteLTISystem::new(
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        );

        match result {
            Err(DiscreteFiniteLTISystemError::RankMismatch {
                expected,
                actual,
                object,
            }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
                assert_eq!(object, "output_matrix_c");
            }
            _ => panic!("Expected RankMismatch error for output_matrix_c"),
        }
    }

    #[test]
    fn construction_of_dflti_system_bad_shape0() {
        // Test state_matrix_a with incorrect shape (3x3 instead of 2x2)
        let state_x = Tensor::from_slice(&[1.0, 2.0]).reshape(&[2, 1]);
        let state_matrix_a = Tensor::eye(3, (tch::Kind::Double, tch::Device::Cpu));
        let input_matrix_b = Tensor::from_slice(&[3.0, 4.0]).reshape(&[2, 1]);
        let output_matrix_c = Tensor::from_slice(&[5.0, 6.0]).reshape(&[1, 2]);
        let feedthrough_matrix_d = Tensor::from_slice(&[7.0]).reshape(&[1, 1]);

        let result = DiscreteFiniteLTISystem::new(
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        );

        match result {
            Err(DiscreteFiniteLTISystemError::ShapeMismatch {
                expected,
                actual,
                object,
            }) => {
                assert_eq!(&*expected, &[2, 2][..]);
                assert_eq!(&*actual, &[3, 3][..]);
                assert_eq!(object, "state_matrix_a");
            }
            _ => panic!("Expected ShapeMismatch error for state_matrix_a"),
        }
    }

    #[test]
    fn test_dflti_system_cloning() {
        // Create a valid system and clone it, then simulate step responses to check state independence
        let state_x = Tensor::from_slice(&[1.0, 2.0]).reshape(&[2, 1]);
        let mut system = DiscreteFiniteLTISystem::new(
            Tensor::eye(2, (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::from_slice(&[0.5, 0.5]).reshape(&[2, 1]),
            Tensor::from_slice(&[1.0, 1.0]).reshape(&[1, 2]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x.shallow_clone(),
        )
        .unwrap();

        let mut clone = system.clone();

        // Apply different inputs to original and clone
        let u_original = Tensor::from_slice(&[1.0]).reshape(&[1, 1]);
        let u_clone = Tensor::from_slice(&[2.0]).reshape(&[1, 1]);

        // Simulate one step for both systems
        let _ = system.simulate_from_current(&u_original).unwrap();
        let _ = clone.simulate_from_current(&u_clone).unwrap();

        // Check that states have diverged correctly by comparing tensors directly
        let expected_original = Tensor::from_slice(&[1.5, 2.5]).reshape(&[2, 1]);
        let expected_clone = Tensor::from_slice(&[2.0, 3.0]).reshape(&[2, 1]);

        assert!(
            system
                .get_state_x()
                .allclose(&expected_original, 1e-5, 1e-5, false)
        );
        assert!(
            clone
                .get_state_x()
                .allclose(&expected_clone, 1e-5, 1e-5, false)
        );
    }

    #[test]
    fn test_pid_regulator_initial_control() {
        // First call with no previous errors or inputs
        let mut pid = DiscretePIDRegulator::new(1.0, 1.0, 0.5, 0.1);
        let reference = DiscretePIDReference(0.0);
        let y = Tensor::from_slice(&[-2.0]).reshape(&[1, 1]);

        let u = pid.compute_control(&y, &reference).unwrap();
        let expected_u = Tensor::from_slice(&[12.1]).reshape(&[1, 1]);

        assert!(u.allclose(&expected_u, 1e-4, 1e-4, false));
        assert!(pid.e_km1.is_some());
        assert!(pid.e_km2.is_none());
        assert!(pid.u_km1.is_some());
    }

    #[test]
    fn test_cflti_system_valid_construction() {
        // Test valid construction of a continuous system
        let state_matrix_a = Tensor::eye(2, (tch::Kind::Double, tch::Device::Cpu));
        let input_matrix_b = Tensor::from_slice(&[3.0, 4.0]).reshape(&[2, 1]);
        let output_matrix_c = Tensor::from_slice(&[5.0, 6.0]).reshape(&[1, 2]);
        let feedthrough_matrix_d = Tensor::from_slice(&[7.0]).reshape(&[1, 1]);
        let state_x = Tensor::from_slice(&[8.0, 9.0]).reshape(&[2, 1]);

        let system = ContinousFiniteLTISystem::new(
            state_matrix_a.shallow_clone(),
            input_matrix_b.shallow_clone(),
            output_matrix_c.shallow_clone(),
            feedthrough_matrix_d.shallow_clone(),
            state_x.shallow_clone(),
        )
        .unwrap();

        assert_eq!(system.get_state_matrix_a().size(), &[2, 2]);
        assert_eq!(system.get_input_matrix_b().size(), &[2, 1]);
        assert_eq!(system.get_output_matrix_c().size(), &[1, 2]);
        assert_eq!(system.get_feedthrough_matrix_d().size(), &[1, 1]);
        assert_eq!(system.get_state_x().size(), &[2, 1]);
    }

    #[test]
    fn test_cflti_system_invalid_shapes() {
        // Test shape mismatch in continuous system construction
        let state_matrix_a = Tensor::eye(3, (tch::Kind::Double, tch::Device::Cpu)); // 3x3
        let input_matrix_b = Tensor::from_slice(&[1.0, 2.0, 3.0]).reshape(&[3, 1]);
        let output_matrix_c = Tensor::from_slice(&[4.0, 5.0, 6.0]).reshape(&[1, 3]);
        let feedthrough_matrix_d = Tensor::from_slice(&[7.0]).reshape(&[1, 1]);
        let state_x = Tensor::from_slice(&[8.0, 9.0]).reshape(&[2, 1]); // 2x1

        let result = ContinousFiniteLTISystem::new(
            state_matrix_a,
            input_matrix_b,
            output_matrix_c,
            feedthrough_matrix_d,
            state_x,
        );

        match result {
            Err(ContinousFiniteLTISystemError::ShapeMismatch {
                expected,
                actual,
                object,
            }) => {
                assert_eq!(&*expected, &[2, 2][..]);
                assert_eq!(&*actual, &[3, 3][..]);
                assert_eq!(object, "state_matrix_a");
            }
            _ => panic!("Expected ShapeMismatch error for state_matrix_a"),
        }
    }

    #[test]
    fn test_step_response_valid() {
        let state_x = Tensor::from_slice(&[0.0]).reshape(&[1, 1]);
        let system = DiscreteFiniteLTISystem::new(
            Tensor::from_slice(&[0.5]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x,
        )
        .unwrap();

        let step = system.step_response(3).unwrap();
        let expected = Tensor::from_slice(&[0.0, 1.0, 1.5]).reshape(&[1, 3]);
        assert!(step.allclose(&expected, 1e-5, 1e-5, false));
    }

    #[test]
    fn test_step_response_invalid_len() {
        let state_x = Tensor::from_slice(&[0.0]).reshape(&[1, 1]);
        let system = DiscreteFiniteLTISystem::new(
            Tensor::eye(1, (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x,
        )
        .unwrap();

        let result = system.step_response(0);
        assert!(matches!(
            result
                .unwrap_err()
                .downcast_ref::<DiscreteFiniteLTISystemError>(),
            Some(DiscreteFiniteLTISystemError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn test_signal_response_preserves_state() {
        let state_x = Tensor::from_slice(&[0.0]).reshape(&[1, 1]);
        let system = DiscreteFiniteLTISystem::new(
            Tensor::from_slice(&[0.5]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x.shallow_clone(),
        )
        .unwrap();

        let u = Tensor::from_slice(&[1.0, 1.0]).reshape(&[1, 2]);
        let output = system.signal_response(&u).unwrap();
        let expected_output = Tensor::from_slice(&[0.0, 1.0]).reshape(&[1, 2]);
        assert!(output.allclose(&expected_output, 1e-5, 1e-5, false));
        assert!(system.get_state_x().allclose(&state_x, 1e-5, 1e-5, false));
    }

    #[test]
    fn test_step_response_nonzero_initial_state() {
        // Test step response with non-zero initial state
        let state_x = Tensor::from_slice(&[1.0]).reshape(&[1, 1]);
        let system = DiscreteFiniteLTISystem::new(
            Tensor::from_slice(&[0.5]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x,
        )
        .unwrap();

        let step = system.step_response(3).unwrap();
        let expected = Tensor::from_slice(&[1.0, 1.5, 1.75]).reshape(&[1, 3]);
        assert!(step.allclose(&expected, 1e-5, 1e-5, false));
    }

    #[test]
    fn test_simulate_from_current_with_feedthrough() {
        // Test system with non-zero D matrix (direct feedthrough)
        let state_x = Tensor::from_slice(&[0.0]).reshape(&[1, 1]);
        let mut system = DiscreteFiniteLTISystem::new(
            Tensor::from_slice(&[0.5]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[0.1]).reshape(&[1, 1]),
            state_x.shallow_clone(),
        )
        .unwrap();

        let u = Tensor::from_slice(&[1.0, 1.0]).reshape(&[1, 2]);
        let output = system.simulate_from_current(&u).unwrap();

        // First output is D*u since initial state is zero
        // Second output is C*(A*x + B*u) + D*u
        let expected_output = Tensor::from_slice(&[0.1, 1.1]).reshape(&[1, 2]);
        assert!(output.allclose(&expected_output, 1e-5, 1e-5, false));
    }

    #[test]
    fn test_empty_input_simulation() {
        // Test with empty input tensor
        let state_x = Tensor::from_slice(&[0.0]).reshape(&[1, 1]);
        let mut system = DiscreteFiniteLTISystem::new(
            Tensor::eye(1, (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[1.0]).reshape(&[1, 1]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x,
        )
        .unwrap();

        let u = Tensor::zeros(&[1, 0], (tch::Kind::Double, tch::Device::Cpu));
        let output = system.simulate_from_current(&u).unwrap();

        assert_eq!(output.size(), &[1, 0]); // Should return empty tensor with same time dimension
    }

    #[test]
    fn test_multi_output_system() {
        // Test system with multiple outputs (though our implementation currently only supports single output)
        // This test verifies the shape handling even though C matrix is 1xn
        let state_x = Tensor::from_slice(&[1.0, 2.0]).reshape(&[2, 1]);
        let system = DiscreteFiniteLTISystem::new(
            Tensor::eye(2, (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::from_slice(&[1.0, 0.0]).reshape(&[2, 1]),
            Tensor::from_slice(&[1.0, 1.0]).reshape(&[1, 2]),
            Tensor::from_slice(&[0.0]).reshape(&[1, 1]),
            state_x,
        )
        .unwrap();

        let u = Tensor::from_slice(&[1.0, 1.0]).reshape(&[1, 2]);
        let output = system.signal_response(&u).unwrap();

        // Output is C*x + D*u for each step
        let expected_output = Tensor::from_slice(&[3.0, 4.0]).reshape(&[1, 2]);
        assert!(output.allclose(&expected_output, 1e-5, 1e-5, false));
    }
}
