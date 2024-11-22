use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    wires::Wire,
};
use ark_ff::Zero;
use mina_curves::pasta::Fp;

use crate::graph::model::{Model, NodeType, OperationType};

const MIN_DOMAIN_SIZE: usize = 256;
const ZK_ROWS: usize = 50;  // Increased ZK rows for safety

pub struct ModelCircuitBuilder {
    current_row: usize,
}

impl ModelCircuitBuilder {
    pub fn new() -> Self {
        Self { current_row: 0 }
    }

    pub fn build_circuit(&mut self, model: &Model) -> Vec<CircuitGate<Fp>> {
        let mut gates = Vec::new();

        // Calculate total number of public inputs
        let num_public = model.graph.inputs.iter().map(|&idx| {
            if let NodeType::Node(node) = &model.graph.nodes[&idx] {
                node.out_dims.iter().product()
            } else {
                0
            }
        }).sum();

        // Add public input gates
        for i in 0..num_public {
            gates.push(CircuitGate {
                typ: GateType::Generic,
                wires: self.create_wires(i),
                coeffs: vec![Fp::from(1u64)],
            });
            self.current_row += 1;
        }

        // Process each node in topological order
        let mut intermediate_rows = std::collections::HashMap::new();
        for (idx, node) in &model.graph.nodes {
            if let NodeType::Node(node) = node {
                match node.op_type {
                    OperationType::MatMul => {
                        let output_size = node.out_dims.iter().product();
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: self.create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }
                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    },
                    OperationType::Relu => {
                        let output_size = node.out_dims.iter().product();
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: self.create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }
                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    },
                    OperationType::Add => {
                        let output_size = node.out_dims.iter().product();
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: self.create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }
                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    },
                    _ => {}
                }
            }
        }

        // Calculate required domain size
        let required_size = std::cmp::max(
            MIN_DOMAIN_SIZE,
            self.next_power_of_two(self.current_row + ZK_ROWS)
        );

        // Add padding gates
        while gates.len() < required_size - ZK_ROWS {
            gates.push(CircuitGate {
                typ: GateType::Zero,
                wires: self.create_wires(self.current_row),
                coeffs: vec![Fp::zero()],
            });
            self.current_row += 1;
        }

        // Add zero-knowledge rows
        for i in 0..ZK_ROWS {
            gates.push(CircuitGate {
                typ: GateType::Zero,
                wires: self.create_wires(self.current_row + i),
                coeffs: vec![Fp::zero()],
            });
        }

        gates
    }

    fn next_power_of_two(&self, n: usize) -> usize {
        let mut v = n;
        v -= 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v += 1;
        v
    }

    fn create_wires(&self, row: usize) -> [Wire; 7] {
        // Create a simple linear connection pattern
        [
            Wire::new(row, 0),     // Current row, main wire
            Wire::new(row, 1),     // Current row, auxiliary wire 1
            Wire::new(row, 2),     // Current row, auxiliary wire 2
            Wire::new(row, 3),     // Current row, auxiliary wire 3
            Wire::new(row, 4),     // Current row, auxiliary wire 4
            Wire::new(row, 5),     // Current row, auxiliary wire 5
            Wire::new(row, 6),     // Current row, auxiliary wire 6
        ]
    }
}
