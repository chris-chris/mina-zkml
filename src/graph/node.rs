use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use tract_onnx::{prelude::{tract_linalg::{self, mmm::MatMatMul}, TypedFact, TypedOp}, tract_core, tract_hir};
use log::{trace, warn};

/// A node's input is a tensor from another node's output
pub type Outlet = (usize, usize);

/// Represents the supported operations in the computation graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SupportedOp {
    /// Linear operations (like matrix multiplication, addition)
    Linear {
        /// Operation type
        op_type: String,
        /// Operation parameters
        params: BTreeMap<String, f32>,
    },
    /// Non-linear operations (like ReLU, Sigmoid)
    Nonlinear {
        /// Operation type
        op_type: String,
        /// Operation parameters
        params: BTreeMap<String, f32>,
    },
    /// Input nodes
    Input {
        /// Name of the input
        name: String,
        /// Shape of the input
        shape: Vec<usize>,
    },
    /// Constant nodes
    Constant {
        /// The constant values
        values: Vec<f32>,
        /// Shape of the constant
        shape: Vec<usize>,
    },
    /// Unknown/unsupported operations
    Unknown
}

impl SupportedOp {
    /// Returns the operation type as a string
    pub fn as_string(&self) -> String {
        match self {
            SupportedOp::Linear { op_type, .. } => format!("Linear({})", op_type),
            SupportedOp::Nonlinear { op_type, .. } => format!("Nonlinear({})", op_type),
            SupportedOp::Input { name, .. } => format!("Input({})", name),
            SupportedOp::Constant { .. } => "Constant".to_string(),
            SupportedOp::Unknown => "Unknown".to_string(),
        }
    }

    /// Checks if the operation is an input
    pub fn is_input(&self) -> bool {
        matches!(self, SupportedOp::Input { .. })
    }

    /// Checks if the operation is a constant
    pub fn is_constant(&self) -> bool {
        matches!(self, SupportedOp::Constant { .. })
    }
}

/// A single node in the computation graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomNode {
    /// Operation type and parameters
    pub opkind: SupportedOp,
    /// The node's unique identifier
    pub idx: usize,
    /// The indices of the node's inputs
    pub inputs: Vec<Outlet>,
    /// Dimensions of output
    pub out_dims: Vec<usize>,
    /// Number of times this node's output is used
    pub num_uses: usize,
    // Output scale
    pub out_scale: i32
}

impl CustomNode {
    /// Creates a new Node from a tract ONNX node
    pub fn new(
        node: tract_onnx::prelude::Node<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, CustomNode>,
        idx: usize,
    ) -> Result<Self, String> {
        trace!("Creating node {:?}", node);
        
        // Calculate number of uses
        let num_uses = std::cmp::max(
            node.outputs
                .iter()
                .map(|outlet| outlet.successors.len())
                .sum::<usize>(),
            1, // minimum of 1 for outputs
        );

        // Process inputs
        let inputs: Vec<Outlet> = node.inputs
            .iter()
            .map(|i| (i.node, i.slot))
            .collect();

        // Get output dimensions
        let out_dims = match node.outputs.get(0) {
            Some(output) => output.fact.shape.as_concrete()
                .ok_or("Could not determine concrete output shape")?
                .to_vec(),
            None => vec![1], // Default to scalar output if no output fact is available
        };

        println!("Node: {:?}", node);
        // Determine operation type
        let opkind = SupportedOp::Unknown;

        Ok(CustomNode {
            opkind,
            idx,
            inputs,
            out_dims,
            num_uses,
            out_scale: 0,
        })
    }
}

impl PartialEq for CustomNode {
    fn eq(&self, other: &CustomNode) -> bool {
        self.idx == other.idx
            && self.inputs == other.inputs
            && self.out_dims == other.out_dims
            && self.opkind.as_string() == other.opkind.as_string()
    }
}

impl fmt::Display for CustomNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node {}: {} (inputs: {:?}, out_dims: {:?}, uses: {})",
            self.idx,
            self.opkind.as_string(),
            self.inputs,
            self.out_dims,
            self.num_uses
        )
    }
}