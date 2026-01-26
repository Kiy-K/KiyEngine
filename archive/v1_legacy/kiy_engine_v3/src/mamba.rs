// src/mamba.rs

use ndarray::{s, Array1, Array2, Array3, Axis};

// Constants based on the spec
pub const D_MODEL: usize = 384;
pub const D_STATE: usize = 16;
pub const D_CONV: usize = 4;
pub const EXPANSION_FACTOR: usize = 2;
pub const D_INNER: usize = D_MODEL * EXPANSION_FACTOR;

/// Represents the state of a Mamba block that needs to be tracked during search.
/// This includes the last few inputs for the convolution and the SSM state.
#[derive(Clone)]
pub struct MambaState {
    pub conv_state: Array2<f32>, // Shape: (D_INNER, D_CONV - 1)
    pub ssm_state: Array2<f32>,  // Shape: (D_INNER, D_STATE)
}

impl MambaState {
    /// Creates a new, zero-initialized MambaState.
    pub fn new() -> Self {
        Self {
            conv_state: Array2::zeros((D_INNER, D_CONV - 1)),
            ssm_state: Array2::zeros((D_INNER, D_STATE)),
        }
    }
}

/// The weights for a single Mamba block.
/// These are immutable and loaded at startup.
pub struct MambaBlock {
    // Input projection weights
    pub in_proj_w: Array2<f32>, // Shape: (2 * D_INNER, D_MODEL)

    // Convolutional layer weights
    pub conv1d_w: Array3<f32>, // Shape: (D_INNER, 1, D_CONV)
    pub conv1d_b: Array1<f32>, // Shape: (D_INNER)

    // SSM parameters
    // Note: D_T is a placeholder for the time-varying component from the input,
    // which is projected along with the state. It corresponds to D_INNER.
    pub x_proj_w: Array2<f32>, // Shape: (D_INNER + 2 * D_STATE, D_INNER)
    pub dt_proj_w: Array1<f32>, // Shape: (D_INNER)
    pub dt_proj_b: Array1<f32>, // Shape: (D_INNER)

    // A and B are not stored directly, but derived from ssm_params
    pub a_log: Array2<f32>, // Shape: (D_INNER, D_STATE)
    pub d: Array1<f32>,     // Shape: (D_INNER)

    // Output projection weights
    pub out_proj_w: Array2<f32>, // Shape: (D_MODEL, D_INNER)
}

/// SiLU activation function
fn silu(x: &mut Array1<f32>) {
    x.mapv_inplace(|v| v / (1.0 + (-v).exp()));
}

impl MambaBlock {
    /// Performs the forward pass for a single Mamba block on a single token.
    ///
    /// # Arguments
    /// * `x` - The input tensor of shape (D_MODEL).
    /// * `state` - The mutable state for this block, containing conv_state and ssm_state.
    ///
    /// # Returns
    /// The output tensor of shape (D_MODEL).
    pub fn forward(&self, x: &Array1<f32>, state: &mut MambaState) -> Array1<f32> {
        // 1. Input Projection
        let xz = self.in_proj_w.dot(x);
        let (x_inner, z) = xz.view().split_at(Axis(0), D_INNER);

        // Make mutable copies
        let mut x_inner = x_inner.to_owned();
        let mut z = z.to_owned();

        // 2. 1D Convolution
        // Update conv state: shift old values and add new input
        let source_slice = state.conv_state.slice(s![.., 1..]).to_owned();
        state.conv_state.slice_mut(s![.., 0..D_CONV-2]).assign(&source_slice);
        state.conv_state.slice_mut(s![.., D_CONV-2]).assign(&x_inner);

        // Perform convolution
        let _conv_weights_flat = self.conv1d_w.view().to_shape(D_INNER * D_CONV).unwrap();
        let _conv_state_flat = state.conv_state.view().to_shape(D_INNER * (D_CONV - 1)).unwrap();

        // Perform the 1D convolution
        let mut conv_input = Array2::<f32>::zeros((D_INNER, D_CONV));
        conv_input.slice_mut(s![.., 0..D_CONV - 1]).assign(&state.conv_state);
        conv_input.slice_mut(s![.., D_CONV - 1]).assign(&x_inner);

        // Reshape weights for broadcasting
        let conv_weights_reshaped = self.conv1d_w.to_shape((D_INNER, D_CONV)).unwrap();

        // Element-wise multiplication and sum along the convolution dimension
        x_inner = (&conv_input * &conv_weights_reshaped).sum_axis(Axis(1));
        x_inner = x_inner + &self.conv1d_b;

        // 3. Activation
        silu(&mut x_inner);

        // 4. SSM (S6) - Selective Scan
        let (y, new_ssm_state) = self.ssm(&x_inner, &state.ssm_state);
        state.ssm_state = new_ssm_state;

        // 5. Gating
        silu(&mut z);
        let output = y * z;

        // 6. Output Projection
        self.out_proj_w.dot(&output)
    }

    /// The core Selective Scan (S6) mechanism.
    ///
    /// # Arguments
    /// * `x` - The input tensor of shape (D_INNER).
    /// * `ssm_state` - The current SSM state of shape (D_INNER, D_STATE).
    ///
    /// # Returns
    /// A tuple of (output, new_ssm_state).
    fn ssm(&self, x: &Array1<f32>, ssm_state: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
        // --- Project input x to get dt, B, C ---
        // This assumes the simplification that B and C are vectors of size D_STATE,
        // broadcast across the D_INNER dimension.
        let ssm_params = self.x_proj_w.dot(x);
        let b = ssm_params.slice(s![D_INNER..D_INNER + D_STATE]);
        let c = ssm_params.slice(s![D_INNER + D_STATE..]);

        // --- Discretize dt ---
        // dt' = dt_proj(x)
        let dt_proj = x * &self.dt_proj_w + &self.dt_proj_b;
        // dt = softplus(dt')
        let mut dt = dt_proj;
        dt.mapv_inplace(|v| (1.0 + v.exp()).ln()); // Correct Softplus implementation

        // --- Discretize A and B (State Space Model update) ---
        // A_bar = exp(dt * A)
        // B_bar = dt * B
        let dt_reshaped = dt.to_shape((D_INNER, 1)).unwrap();

        let a = self.a_log.mapv(|v| v.exp());
        let a_bar = (&dt_reshaped * &a).mapv(|v| v.exp());

        let b_reshaped = b.to_shape((1, D_STATE)).unwrap();
        let b_bar = &dt_reshaped * &b_reshaped;

        // --- Update SSM state ---
        // h_new = A_bar * h_old + B_bar * x
        let x_reshaped = x.to_shape((D_INNER, 1)).unwrap();
        let new_ssm_state = &a_bar * ssm_state + &b_bar * &x_reshaped;

        // --- Calculate output ---
        // y = C * h_new
        let y = (&new_ssm_state * &c).sum_axis(Axis(1));

        // Add residual D connection
        let y = y + x * &self.d;

        (y, new_ssm_state)
    }
}
