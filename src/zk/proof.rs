use ark_ff::{UniformRand, Zero};
use ark_poly::EvaluationDomain;
use groupmap::GroupMap;
use kimchi::{
    circuits::{constraints::ConstraintSystem, wires::COLUMNS},
    proof::ProverProof,
    prover_index::ProverIndex,
    verifier_index::VerifierIndex,
};
use mina_curves::pasta::{Fp, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use poly_commitment::{commitment::CommitmentCurve, ipa::SRS, SRS as _};
use rand::{rngs::ThreadRng, thread_rng};
use serde::{Deserialize, Serialize};
use std::{array, sync::Arc};

use super::wiring::ModelCircuitBuilder;
use super::ZkOpeningProof;
use crate::graph::model::{Model, Visibility};

type SpongeParams = PlonkSpongeConstantsKimchi;
type BaseSponge = DefaultFqSponge<VestaParameters, SpongeParams>;
type ScalarSponge = DefaultFrSponge<Fp, SpongeParams>;

/// Result type containing model output (if public) and its proof
#[derive(Clone, Serialize, Deserialize)]
pub struct ProverOutput {
    pub output: Option<Vec<Vec<f32>>>, // Only Some if output visibility is Public
    pub proof: ProverProof<Vesta, ZkOpeningProof>,
    pub prover_index: ProverIndex<Vesta, ZkOpeningProof>,
    pub verifier_index: VerifierIndex<Vesta, ZkOpeningProof>,
}

/// Creates prover and verifier indices for a model
#[derive(Clone, Serialize, Deserialize)]
pub struct ProverSystem {
    pub prover_index: ProverIndex<Vesta, ZkOpeningProof>,
    pub verifier_index: VerifierIndex<Vesta, ZkOpeningProof>,
    #[serde(skip)]
    model: Model,
    domain_size: usize,
    zk_rows: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct VerifierSystem {
    pub verifier_index: VerifierIndex<Vesta, ZkOpeningProof>,
}

type WitnessOutput = ([Vec<Fp>; COLUMNS], Vec<Vec<f32>>);

impl ProverSystem {
    /// Create a new prover system from a model
    pub fn new(model: &Model) -> Self {
        // Convert model to circuit gates
        let mut builder = ModelCircuitBuilder::new();
        let (gates, domain_size, zk_rows) = builder.build_circuit(model);

        // Calculate total number of public inputs and outputs based on visibility
        let num_public_inputs = if model.visibility.input == Visibility::Public {
            model
                .graph
                .inputs
                .iter()
                .map(|&idx| {
                    if let crate::graph::model::NodeType::Node(node) = &model.graph.nodes[&idx] {
                        node.out_dims.iter().product::<usize>()
                    } else {
                        0usize
                    }
                })
                .sum::<usize>()
        } else {
            0
        };

        let num_public_outputs = if model.visibility.output == Visibility::Public {
            model
                .graph
                .outputs
                .iter()
                .map(|&(node, _)| {
                    if let crate::graph::model::NodeType::Node(node) = &model.graph.nodes[&node] {
                        node.out_dims.iter().product::<usize>()
                    } else {
                        0usize
                    }
                })
                .sum::<usize>()
        } else {
            0
        };

        let total_public = num_public_inputs + num_public_outputs;

        println!("Number of public inputs: {}", num_public_inputs);
        println!("Number of public outputs: {}", num_public_outputs);
        println!("Required domain size: {}", domain_size);

        // Create constraint system with our domain size
        let cs = ConstraintSystem::create(gates)
            .public(total_public)
            .max_poly_size(Some(domain_size))
            .build()
            .expect("Failed to create constraint system");

        println!("Constraint system domain size: {}", cs.domain.d1.size());

        // Create SRS with our domain size
        println!("Using SRS size: {}", domain_size);
        let srs = SRS::<Vesta>::create(domain_size);
        let srs = Arc::new(srs);

        // Create prover index
        let prover_index = ProverIndex::create(cs.clone(), Fp::zero(), srs);

        // Create verifier index
        let verifier_index = prover_index.verifier_index();

        Self {
            prover_index,
            verifier_index,
            model: model.clone(),
            domain_size,
            zk_rows,
        }
    }

    /// Convert f32 to field element with overflow protection
    fn f32_to_field(value: f32) -> Fp {
        const SCALE: f32 = 1_000_000.0; // Increased precision
        const EPSILON: f32 = 1e-6; // Small number threshold
        const MAX_SAFE_VALUE: f32 = ((u64::MAX as f64) / 1_000_000.0) as f32;

        if value.abs() < EPSILON {
            return Fp::zero();
        }

        // Check for overflow
        if value.abs() > MAX_SAFE_VALUE {
            println!(
                "Warning: Value {} exceeds safe range, clamping to ±{}",
                value, MAX_SAFE_VALUE
            );
            let clamped = if value < 0.0 {
                -MAX_SAFE_VALUE
            } else {
                MAX_SAFE_VALUE
            };
            if clamped < 0.0 {
                -Fp::from((-clamped * SCALE) as u64)
            } else {
                Fp::from((clamped * SCALE) as u64)
            }
        } else if value < 0.0 {
            -Fp::from((-value * SCALE) as u64)
        } else {
            Fp::from((value * SCALE) as u64)
        }
    }

    /// Create witness for the circuit
    fn create_witness(&self, inputs: &[Vec<f32>]) -> Result<WitnessOutput, String> {
        // First execute the model to get outputs
        let outputs = self
            .model
            .graph
            .execute(inputs)
            .map_err(|e| format!("Failed to execute model: {:?}", e))?;

        // Calculate initial witness size (without padding)
        let mut witness_size = 0;
        let mut current_pos = 0;

        // Convert inputs to field elements if public
        let mut witness = array::from_fn(|_| vec![Fp::zero(); self.domain_size]);

        if self.model.visibility.input == Visibility::Public {
            let public_inputs: Vec<Fp> = inputs
                .iter()
                .flat_map(|input| input.iter().map(|&x| Self::f32_to_field(x)))
                .collect();

            witness_size += public_inputs.len();

            // Place public inputs at the start
            for (i, &value) in public_inputs.iter().enumerate() {
                for item in witness.iter_mut().take(COLUMNS) {
                    item[i] = value;
                }
            }
            current_pos += public_inputs.len();
        }

        // Convert outputs to field elements if public
        if self.model.visibility.output == Visibility::Public {
            let public_outputs: Vec<Fp> = outputs
                .iter()
                .flat_map(|output| output.iter().map(|&x| Self::f32_to_field(x)))
                .collect();

            witness_size += public_outputs.len();

            // Place public outputs after inputs
            for (i, &value) in public_outputs.iter().enumerate() {
                for item in witness.iter_mut().take(COLUMNS) {
                    item[current_pos + i] = value;
                }
            }
            current_pos += public_outputs.len();
        }

        // Add space for intermediate computations
        for node in self.model.graph.nodes.values() {
            if let crate::graph::model::NodeType::Node(node) = node {
                match node.op_type {
                    crate::graph::model::OperationType::MatMul => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    }
                    crate::graph::model::OperationType::Relu => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    }
                    crate::graph::model::OperationType::Add => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    }
                    _ => {}
                }
            }
        }

        // Ensure witness size is strictly less than domain_size - zk_rows
        assert!(
            witness_size < self.domain_size - self.zk_rows,
            "Witness size {} must be strictly less than domain size {} minus zk_rows {}",
            witness_size,
            self.domain_size,
            self.zk_rows
        );

        // Process each node in topological order
        let mut intermediate_values = std::collections::HashMap::new();

        for (idx, node) in &self.model.graph.nodes {
            if let crate::graph::model::NodeType::Node(node) = node {
                match node.op_type {
                    crate::graph::model::OperationType::MatMul => {
                        let input_size = node.inputs[0].1;
                        let output_size = node.out_dims.iter().product();

                        // Get input values
                        let input_values = if let Some((input_idx, _)) = node.inputs.first() {
                            intermediate_values
                                .get(input_idx)
                                .map(|&row| (0..input_size).map(|i| witness[0][row + i]).collect())
                                .unwrap_or_else(|| (0..input_size).map(|i| witness[0][i]).collect())
                        } else {
                            vec![Fp::zero(); input_size]
                        };

                        // Compute matrix multiplication
                        if let Some(weights) = &node.weights {
                            for i in 0..output_size {
                                let mut sum = Fp::zero();
                                for j in 0..input_size {
                                    let weight = Self::f32_to_field(weights[i * input_size + j]);
                                    sum += weight * input_values[j];
                                }
                                // Set the result in all columns
                                for item in witness.iter_mut().take(COLUMNS) {
                                    item[current_pos + i] = sum;
                                }
                            }
                            intermediate_values.insert(*idx, current_pos);
                            current_pos += output_size;
                        }
                    }
                    crate::graph::model::OperationType::Add => {
                        if let (Some((left_idx, _)), Some((right_idx, _))) =
                            (node.inputs.first(), node.inputs.get(1))
                        {
                            if let (Some(&left_row), Some(&right_row)) = (
                                intermediate_values.get(left_idx),
                                intermediate_values.get(right_idx),
                            ) {
                                let size = node.out_dims.iter().product();
                                for i in 0..size {
                                    let result =
                                        witness[0][left_row + i] + witness[0][right_row + i];
                                    // Set the result in all columns
                                    for item in witness.iter_mut().take(COLUMNS) {
                                        item[current_pos + i] = result;
                                    }
                                }
                                intermediate_values.insert(*idx, current_pos);
                                current_pos += size;
                            }
                        }
                    }
                    crate::graph::model::OperationType::Relu
                    | crate::graph::model::OperationType::Max => {
                        if let Some((input_idx, _)) = node.inputs.first() {
                            if let Some(&input_row) = intermediate_values.get(input_idx) {
                                let size = node.out_dims.iter().product();
                                for i in 0..size {
                                    let x = witness[0][input_row + i];
                                    let result = if x == Fp::zero() { Fp::zero() } else { x };
                                    // Set the result in all columns
                                    for item in witness.iter_mut().take(COLUMNS) {
                                        item[current_pos + i] = result;
                                    }
                                }
                                intermediate_values.insert(*idx, current_pos);
                                current_pos += size;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Add random values for zero-knowledge rows at the end
        let mut rng = thread_rng();
        for item in witness.iter_mut().take(COLUMNS) {
            for i in item
                .iter_mut()
                .take(self.domain_size)
                .skip(self.domain_size - self.zk_rows)
            {
                *i = <Fp as UniformRand>::rand(&mut rng);
            }
        }

        // Pad remaining rows with zeros
        for item in witness.iter_mut().take(COLUMNS) {
            for i in item
                .iter_mut()
                .take(self.domain_size - self.zk_rows)
                .skip(current_pos)
            {
                *i = Fp::zero();
            }
        }

        Ok((witness, outputs))
    }

    /// Generate model output and create a proof
    pub fn prove(&self, inputs: &[Vec<f32>]) -> Result<ProverOutput, String> {
        // Create witness and get outputs
        let (witness, outputs) = self.create_witness(inputs)?;

        // Setup group map
        let group_map = <Vesta as CommitmentCurve>::Map::setup();

        // Create proof
        let mut rng = thread_rng();
        let proof = ProverProof::create::<BaseSponge, ScalarSponge, ThreadRng>(
            &group_map,
            witness,
            &[],
            &self.prover_index,
            &mut rng,
        )
        .map_err(|e| format!("Failed to create proof: {:?}", e))?;

        Ok(ProverOutput {
            output: if self.model.visibility.output == Visibility::Public {
                Some(outputs)
            } else {
                None
            },
            proof,
            prover_index: self.prover_index.clone(),
            verifier_index: self.verifier_index.clone(),
        })
    }

    /// Get the verifier system for this prover
    pub fn verifier(&self) -> VerifierSystem {
        VerifierSystem {
            verifier_index: self.verifier_index.clone(),
        }
    }
}

impl VerifierSystem {
    /// Create a new verifier system from a verifier index
    pub fn new(verifier_index: VerifierIndex<Vesta, ZkOpeningProof>) -> Self {
        Self { verifier_index }
    }

    /// Convert f32 to field element with overflow protection
    fn f32_to_field(value: f32) -> Fp {
        const SCALE: f32 = 1_000_000.0; // Match ProverSystem scale
        const EPSILON: f32 = 1e-6; // Small number threshold
        const MAX_SAFE_VALUE: f32 = ((u64::MAX as f64) / 1_000_000.0) as f32;

        if value.abs() < EPSILON {
            return Fp::zero();
        }

        // Check for overflow
        if value.abs() > MAX_SAFE_VALUE {
            println!(
                "Warning: Value {} exceeds safe range, clamping to ±{}",
                value, MAX_SAFE_VALUE
            );
            let clamped = if value < 0.0 {
                -MAX_SAFE_VALUE
            } else {
                MAX_SAFE_VALUE
            };
            if clamped < 0.0 {
                -Fp::from((-clamped * SCALE) as u64)
            } else {
                Fp::from((clamped * SCALE) as u64)
            }
        } else if value < 0.0 {
            -Fp::from((-value * SCALE) as u64)
        } else {
            Fp::from((value * SCALE) as u64)
        }
    }

    /// Verify a proof with optional public inputs/outputs
    pub fn verify(
        &self,
        proof: &ProverProof<Vesta, ZkOpeningProof>,
        public_inputs: Option<&[Vec<f32>]>,
        public_outputs: Option<&[Vec<f32>]>,
    ) -> Result<bool, String> {
        let mut public_values = Vec::new();

        // Add public inputs if provided
        if let Some(inputs) = public_inputs {
            println!("Processing public inputs: {:?}", inputs);
            for (i, input) in inputs.iter().enumerate() {
                println!("Processing input {}: {:?}", i, input);
                for (j, &x) in input.iter().enumerate() {
                    let field_val = Self::f32_to_field(x);
                    println!(
                        "Converting input {},{} = {} to field: {:?}",
                        i, j, x, field_val
                    );
                    public_values.push(field_val);
                }
            }
        }

        // Add public outputs if provided
        if let Some(outputs) = public_outputs {
            println!("Processing public outputs: {:?}", outputs);
            for (i, output) in outputs.iter().enumerate() {
                println!("Processing output {}: {:?}", i, output);
                for (j, &x) in output.iter().enumerate() {
                    let field_val = Self::f32_to_field(x);
                    println!(
                        "Converting output {},{} = {} to field: {:?}",
                        i, j, x, field_val
                    );
                    public_values.push(field_val);
                }
            }
        }

        if !public_values.is_empty() {
            println!("Verifying with {} public values", public_values.len());
            if log::log_enabled!(log::Level::Debug) {
                println!("Public values: {:?}", public_values);
            }
        }

        // Setup group map
        let group_map = <Vesta as CommitmentCurve>::Map::setup();

        // Verify proof with public values
        let result = kimchi::verifier::verify::<Vesta, BaseSponge, ScalarSponge, ZkOpeningProof>(
            &group_map,
            &self.verifier_index,
            proof,
            &public_values,
        );

        match result {
            Ok(_) => {
                println!("Proof verification successful");
                Ok(true)
            }
            Err(e) => {
                println!("Proof verification failed: {:?}", e);
                Err(format!("Failed to verify proof: {:?}", e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{RunArgs, VarVisibility};
    use std::collections::HashMap;

    #[test]
    fn test_proof_system_public() {
        // Create a simple model (perceptron) with public visibility
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        let run_args = RunArgs { variables };

        let visibility = VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        };

        let model = Model::new("models/simple_perceptron.onnx", &run_args, &visibility)
            .expect("Failed to load model");

        // Create prover system
        let prover = ProverSystem::new(&model);

        // Create sample input
        let input = vec![vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];

        // Generate output and proof
        let prover_output = prover.prove(&input).expect("Failed to create proof");
        let outputs = prover_output.output.expect("Output should be public");

        // Get verifier from prover
        let verifier = prover.verifier();

        // Verify the proof
        let result = verifier
            .verify(&prover_output.proof, Some(&input), Some(&outputs))
            .expect("Failed to verify proof");

        assert!(result, "Proof verification failed");
    }

    #[test]
    fn test_proof_system_private() {
        // Create a model with private input/output
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        let run_args = RunArgs { variables };

        let visibility = VarVisibility {
            input: Visibility::Private,
            output: Visibility::Private,
        };

        let model = Model::new("models/simple_perceptron.onnx", &run_args, &visibility)
            .expect("Failed to load model");

        // Create prover system
        let prover = ProverSystem::new(&model);

        // Create sample input
        let input = vec![vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];

        // Generate proof
        let prover_output = prover.prove(&input).expect("Failed to create proof");
        assert!(prover_output.output.is_none(), "Output should be private");

        // Get verifier from prover
        let verifier = prover.verifier();

        // Verify the proof without public values
        let result = verifier
            .verify(&prover_output.proof, None, None)
            .expect("Failed to verify proof");

        assert!(result, "Proof verification failed");
    }
}
