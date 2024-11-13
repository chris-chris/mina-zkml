use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    wires::Wire,
};
use mina_curves::pasta::Fp;
use tract_onnx::prelude::*;
use std::borrow::Cow;

use crate::graph::model::{SerializableNode, OperationType};
use anyhow::Result;

/// Maps ONNX operations to Kimchi circuit gates
#[derive(Debug)]
pub enum OnnxOperation {
    /// Matrix multiplication (Gemm/MatMul)
    MatMul {
        m: usize,  // Number of rows in first matrix
        n: usize,  // Number of columns in second matrix
        k: usize,  // Number of columns in first matrix/rows in second matrix
    },
    /// ReLU activation
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Addition operation
    Add,
    /// EinSum operation (used for matrix operations)
    EinSum,
    /// Max operation (used in ReLU)
    Max,
    /// Constant value
    Const,
}

impl OnnxOperation {
    /// Convert ONNX operation to Kimchi circuit gates
    pub fn to_circuit_gates(&self, start_row: usize) -> Result<Vec<CircuitGate<Fp>>> {
        match self {
            OnnxOperation::MatMul { m, n, k } => {
                let mut gates = Vec::new();
                let mut current_row = start_row;

                // For each output element (m x n matrix)
                for _i in 0..*m {
                    for _j in 0..*n {
                        // For each element in the dot product (k elements)
                        for _l in 0..*k {
                            // Multiplication gate
                            let mul_gate = CircuitGate::new(
                                GateType::ForeignFieldMul,
                                [Wire::new(current_row, 0); 7],
                                vec![],
                            );
                            gates.push(mul_gate);
                            current_row += 1;

                            // Addition gate (except for the first element)
                            if _l > 0 {
                                let add_gate = CircuitGate::new(
                                    GateType::ForeignFieldAdd,
                                    [Wire::new(current_row, 0); 7],
                                    vec![],
                                );
                                gates.push(add_gate);
                                current_row += 1;
                            }
                        }
                    }
                }
                Ok(gates)
            }

            OnnxOperation::Relu | OnnxOperation::Max => {
                // ReLU implemented using range check and generic gates
                let mut gates = Vec::new();
                
                // Range check for input
                let range_check = CircuitGate::new(
                    GateType::RangeCheck0,
                    [Wire::new(start_row, 0); 7],
                    vec![],
                );
                gates.push(range_check);

                // Generic gate for max(0, x) logic
                let generic = CircuitGate::new(
                    GateType::Generic,
                    [Wire::new(start_row + 1, 0); 7],
                    vec![],
                );
                gates.push(generic);

                Ok(gates)
            }

            OnnxOperation::Sigmoid => {
                // Sigmoid implemented using generic gates for the sigmoid function
                let mut gates = Vec::new();
                
                // Generic gate for sigmoid computation
                let generic = CircuitGate::new(
                    GateType::Generic,
                    [Wire::new(start_row, 0); 7],
                    vec![],
                );
                gates.push(generic);

                Ok(gates)
            }

            OnnxOperation::Add | OnnxOperation::EinSum => {
                // Addition operation
                let mut gates = Vec::new();
                
                // Generic gate for addition
                let add_gate = CircuitGate::new(
                    GateType::ForeignFieldAdd,
                    [Wire::new(start_row, 0); 7],
                    vec![],
                );
                gates.push(add_gate);

                Ok(gates)
            }

            OnnxOperation::Const => {
                // Const nodes don't need any gates as they're just values
                Ok(vec![])
            }
        }
    }
}

/// Attempts to identify the ONNX operation type from a serialized node
pub fn identify_operation(node: &SerializableNode) -> Option<OnnxOperation> {
    match node.op_type {
        OperationType::Input => None,
        OperationType::Const => Some(OnnxOperation::Const),
        OperationType::MatMul => {
            if node.inputs.len() == 2 {
                let m = node.out_dims[0];
                let n = node.out_dims[1];
                let k = if node.inputs.len() == 2 {
                    // For MatMul, k is the inner dimension
                    node.out_dims[1]  // This should be derived from input dimensions
                } else {
                    0
                };
                Some(OnnxOperation::MatMul { m, n, k })
            } else {
                None
            }
        },
        OperationType::Relu => Some(OnnxOperation::Relu),
        OperationType::Sigmoid => Some(OnnxOperation::Sigmoid),
        OperationType::Add => Some(OnnxOperation::Add),
        OperationType::EinSum => Some(OnnxOperation::EinSum),
        OperationType::Max => Some(OnnxOperation::Max),
    }
}

/// Identifies the operation type from a tract node
pub fn identify_tract_operation(node: &TypedNode) -> Option<OperationType> {
    // Check operation type based on the node's operation name
    match node.op.name() {
        name if name == Cow::from("Const") => {
            println!("Found Const operation");
            Some(OperationType::Const)
        },
        name if name == Cow::from("MatMul") || name == Cow::from("Gemm") || name == Cow::from("EinSum") => {
            println!("Found matrix operation: {}", name);
            if name == Cow::from("EinSum") {
                Some(OperationType::EinSum)
            } else {
                Some(OperationType::MatMul)
            }
        },
        name if name == Cow::from("Relu") || name == Cow::from("Max") => {
            println!("Found ReLU/Max operation: {}", name);
            if name == Cow::from("Max") {
                Some(OperationType::Max)
            } else {
                Some(OperationType::Relu)
            }
        },
        name if name == Cow::from("Sigmoid") => {
            println!("Found Sigmoid operation: {}", name);
            Some(OperationType::Sigmoid)
        },
        name if name == Cow::from("Add") => {
            println!("Found Add operation: {}", name);
            Some(OperationType::Add)
        },
        name => {
            if node.inputs.is_empty() {
                println!("Found Input operation");
                Some(OperationType::Input)
            } else {
                println!("Unknown operation: {}", name);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_gate_generation() {
        let op = OnnxOperation::MatMul { m: 2, n: 2, k: 2 };
        let gates = op.to_circuit_gates(0).unwrap();
        assert!(gates.len() > 0);
    }

    #[test]
    fn test_relu_gate_generation() {
        let op = OnnxOperation::Relu;
        let gates = op.to_circuit_gates(0).unwrap();
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].typ, GateType::RangeCheck0);
        assert_eq!(gates[1].typ, GateType::Generic);
    }

    #[test]
    fn test_sigmoid_gate_generation() {
        let op = OnnxOperation::Sigmoid;
        let gates = op.to_circuit_gates(0).unwrap();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].typ, GateType::Generic);
    }

    #[test]
    fn test_operation_identification() {
        // Test MatMul identification
        let matmul_node = SerializableNode {
            op_type: OperationType::MatMul,
            inputs: vec![(0, 0), (1, 0)],
            out_dims: vec![2, 2],
            out_scale: 1,
            id: 0,
            weights: None,
            bias: None,
        };
        match identify_operation(&matmul_node) {
            Some(OnnxOperation::MatMul { m, n, k }) => {
                assert_eq!(m, 2);
                assert_eq!(n, 2);
                assert_eq!(k, 2);
            }
            _ => panic!("Expected MatMul operation"),
        }

        // Test ReLU identification
        let relu_node = SerializableNode {
            op_type: OperationType::Relu,
            inputs: vec![(0, 0)],
            out_dims: vec![4],
            out_scale: 1,
            id: 0,
            weights: None,
            bias: None,
        };
        match identify_operation(&relu_node) {
            Some(OnnxOperation::Relu) => (),
            _ => panic!("Expected ReLU operation"),
        }

        // Test Sigmoid identification
        let sigmoid_node = SerializableNode {
            op_type: OperationType::Sigmoid,
            inputs: vec![(0, 0)],
            out_dims: vec![4],
            out_scale: 1,
            id: 0,
            weights: None,
            bias: None,
        };
        match identify_operation(&sigmoid_node) {
            Some(OnnxOperation::Sigmoid) => (),
            _ => panic!("Expected Sigmoid operation"),
        }

        // Test Input node (should return None)
        let input_node = SerializableNode {
            op_type: OperationType::Input,
            inputs: vec![],
            out_dims: vec![4],
            out_scale: 1,
            id: 0,
            weights: None,
            bias: None,
        };
        assert!(identify_operation(&input_node).is_none());

        // Test Const node
        let const_node = SerializableNode {
            op_type: OperationType::Const,
            inputs: vec![],
            out_dims: vec![4],
            out_scale: 1,
            id: 0,
            weights: Some(vec![1.0, 2.0, 3.0, 4.0]),
            bias: None,
        };
        match identify_operation(&const_node) {
            Some(OnnxOperation::Const) => (),
            _ => panic!("Expected Const operation"),
        }
    }
}