use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    wires::Wire,
};
use mina_curves::pasta::Fp;

use crate::graph::model::{Model, NodeType, OperationType};

pub mod operations;
pub mod wiring;

/// Represents a node in the ZK proof circuit
#[derive(Debug)]
pub struct ZkNode {
    /// Operation type
    pub op_type: OperationType,
    /// Starting row in the circuit
    pub start_row: usize,
    /// Number of rows used by this node
    pub num_rows: usize,
    /// Gates implementing this node's operation
    pub gates: Vec<CircuitGate<Fp>>,
}

/// Generates ZK proof circuit from model
pub struct ZkProofGenerator {
    /// Current row in the circuit
    current_row: usize,
    /// List of nodes in the circuit
    nodes: Vec<ZkNode>,
}

impl ZkProofGenerator {
    /// Create a new ZK proof generator
    pub fn new() -> Self {
        ZkProofGenerator {
            current_row: 0,
            nodes: Vec::new(),
        }
    }

    /// Generate ZK proof circuit from model
    pub fn generate_circuit(&mut self, model: &Model) -> Vec<CircuitGate<Fp>> {
        let mut gates = Vec::new();

        // Process nodes in topological order
        for node in model.graph.nodes.values() {
            match node {
                NodeType::Node(n) => {
                    // Convert operation to circuit gates
                    let node_gates = match n.op_type {
                        OperationType::Input => vec![],
                        OperationType::MatMul => {
                            let m = n.out_dims[0];
                            let n_dim = n.out_dims[1];
                            let k = if !n.inputs.is_empty() {
                                n.out_dims[1]
                            } else {
                                0
                            };
                            self.generate_matmul_gates(m, n_dim, k)
                        }
                        OperationType::Relu => self.generate_relu_gates(),
                        OperationType::Sigmoid => self.generate_sigmoid_gates(),
                        OperationType::Add => self.generate_add_gates(),
                        OperationType::EinSum => self.generate_einsum_gates(),
                        OperationType::Max => self.generate_max_gates(),
                        OperationType::Const => vec![],
                        OperationType::RmAxis => vec![], // Shape operation, no gates needed
                        OperationType::Reshape => vec![], // Shape operation, no gates needed
                    };

                    // Add gates to circuit
                    gates.extend(node_gates);
                }
                NodeType::SubGraph { .. } => {
                    // Subgraphs not supported yet
                }
            }
        }

        gates
    }

    /// Generate gates for matrix multiplication
    fn generate_matmul_gates(&mut self, m: usize, n: usize, k: usize) -> Vec<CircuitGate<Fp>> {
        let mut gates = Vec::new();
        let start_row = self.current_row;

        // For each output element
        for i in 0..m {
            for j in 0..n {
                // For each element in dot product
                for l in 0..k {
                    // Multiplication gate
                    gates.push(CircuitGate::new(
                        GateType::ForeignFieldMul,
                        [Wire::new(self.current_row, 0); 7],
                        vec![],
                    ));
                    self.current_row += 1;

                    // Addition gate (except for first element)
                    if l > 0 {
                        gates.push(CircuitGate::new(
                            GateType::ForeignFieldAdd,
                            [Wire::new(self.current_row, 0); 7],
                            vec![],
                        ));
                        self.current_row += 1;
                    }
                }
            }
        }

        // Record node info
        self.nodes.push(ZkNode {
            op_type: OperationType::MatMul,
            start_row,
            num_rows: self.current_row - start_row,
            gates: gates.clone(),
        });

        gates
    }

    /// Generate gates for ReLU activation
    fn generate_relu_gates(&mut self) -> Vec<CircuitGate<Fp>> {
        let mut gates = Vec::new();
        let start_row = self.current_row;

        // Range check gate
        gates.push(CircuitGate::new(
            GateType::RangeCheck0,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Generic gate for max(0,x)
        gates.push(CircuitGate::new(
            GateType::Generic,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Record node info
        self.nodes.push(ZkNode {
            op_type: OperationType::Relu,
            start_row,
            num_rows: 2,
            gates: gates.clone(),
        });

        gates
    }

    /// Generate gates for sigmoid activation
    fn generate_sigmoid_gates(&mut self) -> Vec<CircuitGate<Fp>> {
        let mut gates = Vec::new();
        let start_row = self.current_row;

        // Generic gate for sigmoid computation
        gates.push(CircuitGate::new(
            GateType::Generic,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Record node info
        self.nodes.push(ZkNode {
            op_type: OperationType::Sigmoid,
            start_row,
            num_rows: 1,
            gates: gates.clone(),
        });

        gates
    }

    /// Generate gates for addition
    fn generate_add_gates(&mut self) -> Vec<CircuitGate<Fp>> {
        let mut gates = Vec::new();
        let start_row = self.current_row;

        // Addition gate
        gates.push(CircuitGate::new(
            GateType::ForeignFieldAdd,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Record node info
        self.nodes.push(ZkNode {
            op_type: OperationType::Add,
            start_row,
            num_rows: 1,
            gates: gates.clone(),
        });

        gates
    }

    /// Generate gates for EinSum operation
    fn generate_einsum_gates(&mut self) -> Vec<CircuitGate<Fp>> {
        // Similar to MatMul for now
        let mut gates = Vec::new();
        let start_row = self.current_row;

        // Generic gate for EinSum computation
        gates.push(CircuitGate::new(
            GateType::Generic,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Record node info
        self.nodes.push(ZkNode {
            op_type: OperationType::EinSum,
            start_row,
            num_rows: 1,
            gates: gates.clone(),
        });

        gates
    }

    /// Generate gates for max operation
    fn generate_max_gates(&mut self) -> Vec<CircuitGate<Fp>> {
        // Similar to ReLU
        let mut gates = Vec::new();
        let start_row = self.current_row;

        // Range check gate
        gates.push(CircuitGate::new(
            GateType::RangeCheck0,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Generic gate for max computation
        gates.push(CircuitGate::new(
            GateType::Generic,
            [Wire::new(self.current_row, 0); 7],
            vec![],
        ));
        self.current_row += 1;

        // Record node info
        self.nodes.push(ZkNode {
            op_type: OperationType::Max,
            start_row,
            num_rows: 2,
            gates: gates.clone(),
        });

        gates
    }
}

impl Default for ZkProofGenerator {
    fn default() -> Self {
        Self::new()
    }
}
