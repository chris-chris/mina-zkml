use ark_ff::Zero;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use kimchi::circuits::{
    gate::{CircuitGate, GateType},
    wires::Wire,
};
use mina_curves::pasta::Fp;

use crate::graph::model::{Model, NodeType, OperationType};

/// Number of coefficients columns in the circuit
pub const COLUMNS: usize = 15;

/// Number of permutable columns in the circuit
pub const PERMUTS: usize = 7;

/// Wire indices for all columns
pub const WIRES: [usize; COLUMNS] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

/// Minimum domain size required for the circuit
pub const MIN_DOMAIN_SIZE: usize = 4096;

/// Builder for creating arithmetic circuits from computational models
///
/// The `ModelCircuitBuilder` constructs a circuit representation of a computational
/// model by translating each operation into corresponding circuit gates. It handles
/// the layout of gates, padding, and zero-knowledge rows required for the proof system.
pub struct ModelCircuitBuilder {
    /// Current row index in the circuit being built
    current_row: usize,
}

impl Default for ModelCircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelCircuitBuilder {
    /// Creates a new `ModelCircuitBuilder` instance
    pub fn new() -> Self {
        Self { current_row: 0 }
    }

    /// Calculates the strict lower bound for the number of zero-knowledge rows
    ///
    /// # Arguments
    /// * `num_chunks` - Number of chunks the domain is divided into
    ///
    /// # Returns
    /// The minimum number of zero-knowledge rows required
    fn zk_rows_strict_lower_bound(num_chunks: usize) -> usize {
        (2 * (PERMUTS + 1) * num_chunks - 2) / PERMUTS
    }

    /// Calculates the next power of two greater than or equal to `n`
    ///
    /// # Arguments
    /// * `n` - The input number
    ///
    /// # Returns
    /// The smallest power of two >= `n`
    fn next_power_of_two(n: usize) -> usize {
        if n.is_power_of_two() {
            n
        } else {
            let mut v = n;
            v -= 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            #[cfg(target_pointer_width = "64")]
            {
                v |= v >> 32;
            }
            v += 1;
            v
        }
    }

    /// Creates wire connections for a given row
    ///
    /// # Arguments
    /// * `row` - The row index to create wires for
    ///
    /// # Returns
    /// Array of `Wire` structs for the specified row
    fn create_wires(row: usize) -> [Wire; PERMUTS] {
        [
            Wire { row, col: 0 }, // Current row, main wire
            Wire { row, col: 1 }, // Current row, auxiliary wire 1
            Wire { row, col: 2 }, // Current row, auxiliary wire 2
            Wire { row, col: 3 }, // Current row, auxiliary wire 3
            Wire { row, col: 4 }, // Current row, auxiliary wire 4
            Wire { row, col: 5 }, // Current row, auxiliary wire 5
            Wire { row, col: 6 }, // Current row, auxiliary wire 6
        ]
    }

    /// Calculates the domain size and number of zero-knowledge rows
    ///
    /// # Arguments
    /// * `circuit_size` - The size of the circuit being built
    ///
    /// # Returns
    /// A tuple containing (domain_size, zk_rows)
    fn calculate_domain_params(circuit_size: usize) -> (usize, usize) {
        let lookup_domain_size = 0; // We don't use lookup tables yet
        let circuit_lower_bound = std::cmp::max(circuit_size, lookup_domain_size + 1);
        let get_domain_size_lower_bound = |zk_rows: usize| circuit_lower_bound + zk_rows;

        // Start with minimum values
        let zk_rows = 3;
        let domain_size_lower_bound = get_domain_size_lower_bound(zk_rows);

        // Calculate initial domain size
        let mut domain_size = match Radix2EvaluationDomain::<Fp>::new(domain_size_lower_bound) {
            Some(domain) => std::cmp::max(MIN_DOMAIN_SIZE, domain.size()),
            None => std::cmp::max(
                MIN_DOMAIN_SIZE,
                Self::next_power_of_two(domain_size_lower_bound),
            ),
        };

        // Calculate number of chunks and required zk_rows
        let num_chunks = domain_size.div_ceil(MIN_DOMAIN_SIZE);
        let min_zk_rows = Self::zk_rows_strict_lower_bound(num_chunks) + 1;
        let zk_rows = std::cmp::max(min_zk_rows, (16 * num_chunks + 5) / 7);

        // Ensure domain size is large enough
        let domain_size_lower_bound = get_domain_size_lower_bound(zk_rows);
        if domain_size < domain_size_lower_bound {
            domain_size = match Radix2EvaluationDomain::<Fp>::new(domain_size_lower_bound) {
                Some(domain) => std::cmp::max(MIN_DOMAIN_SIZE, domain.size()),
                None => std::cmp::max(
                    MIN_DOMAIN_SIZE,
                    Self::next_power_of_two(domain_size_lower_bound),
                ),
            };
        }

        (domain_size, zk_rows)
    }

    /// Builds a circuit from a computational model
    ///
    /// # Arguments
    /// * `model` - The computational model to translate into a circuit
    ///
    /// # Returns
    /// A tuple containing:
    /// - Vector of circuit gates
    /// - Domain size
    /// - Number of zero-knowledge rows
    pub fn build_circuit(&mut self, model: &Model) -> (Vec<CircuitGate<Fp>>, usize, usize) {
        let mut gates = Vec::new();
        // Calculate total number of public inputs
        let num_public: usize = model
            .graph
            .inputs
            .iter()
            .map(|&idx| {
                if let NodeType::Node(node) = &model.graph.nodes[&idx] {
                    node.out_dims.iter().product::<usize>()
                } else {
                    0
                }
            })
            .sum::<usize>();

        // Calculate initial circuit size (without padding)
        let mut circuit_size = num_public;

        // Calculate space needed for operations
        for node in model.graph.nodes.values() {
            if let NodeType::Node(node) = node {
                match node.op_type {
                    OperationType::MatMul => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Relu => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Add => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Conv => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::MaxPool => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Gather => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Softmax => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Reduce => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::AddAxis => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::Cast => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::TypedBinOp => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    OperationType::ElementWiseOp => {
                        let output_size = node.out_dims.iter().product::<usize>();
                        circuit_size += output_size;
                    }
                    _ => {}
                }
            }
        }

        // Calculate domain size and zk_rows
        let (domain_size, zk_rows) = Self::calculate_domain_params(circuit_size);

        // Add gates for public inputs
        for i in 0..num_public {
            gates.push(CircuitGate {
                typ: GateType::Generic,
                wires: Self::create_wires(i),
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
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Relu => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Add => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Conv => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::MaxPool => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Gather => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Softmax => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Reduce => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::AddAxis => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::Cast => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::TypedBinOp => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    OperationType::ElementWiseOp => {
                        let output_size: usize = node.out_dims.iter().product();

                        // Add computation gates
                        for i in 0..output_size {
                            gates.push(CircuitGate {
                                typ: GateType::Generic,
                                wires: Self::create_wires(self.current_row + i),
                                coeffs: vec![Fp::from(1u64)],
                            });
                        }

                        intermediate_rows.insert(*idx, self.current_row);
                        self.current_row += output_size;
                    }
                    _ => {}
                }
            }
        }

        // Add padding gates until we reach domain_size - zk_rows
        while gates.len() < domain_size - zk_rows {
            gates.push(CircuitGate {
                typ: GateType::Zero,
                wires: Self::create_wires(self.current_row),
                coeffs: vec![Fp::zero()],
            });
            self.current_row += 1;
        }

        // Add zero-knowledge rows
        for i in 0..zk_rows {
            gates.push(CircuitGate {
                typ: GateType::Zero,
                wires: Self::create_wires(self.current_row + i),
                coeffs: vec![Fp::zero()],
            });
        }

        // Ensure we have at least 2 gates (required by o1js/proof-systems)
        while gates.len() < 2 {
            gates.push(CircuitGate {
                typ: GateType::Zero,
                wires: Self::create_wires(self.current_row),
                coeffs: vec![Fp::zero()],
            });
            self.current_row += 1;
        }

        (gates, domain_size, zk_rows)
    }
}
