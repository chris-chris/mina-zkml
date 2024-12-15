use crate::graph::model::*;
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
pub struct ProofSystem {
    pub prover_index: ProverIndex<Vesta, ZkOpeningProof>,
    pub verifier_index: VerifierIndex<Vesta, ZkOpeningProof>,
    #[serde(skip)]
    model: Model,
    domain_size: usize,
    zk_rows: usize,
}

type WitnessOutput = ([Vec<Fp>; COLUMNS], Vec<Vec<f32>>);

impl ProofSystem {
    /// Create a new proof system from a model
    pub fn new(model: &Model) -> Self {
        // Convert model to circuit gates
        let mut builder = ModelCircuitBuilder::new();
        let (gates, domain_size, zk_rows) = builder.build_circuit(model);

        // Calculate total number of public inputs and outputs
        let num_public_inputs = model
            .graph
            .inputs
            .iter()
            .map(|&idx| {
                if let NodeType::Node(node) = &model.graph.nodes[&idx] {
                    node.out_dims.iter().product::<usize>()
                } else {
                    0usize
                }
            })
            .sum::<usize>();

        let num_public_outputs = model
            .graph
            .outputs
            .iter()
            .map(|&(node, _)| {
                if let NodeType::Node(node) = &model.graph.nodes[&node] {
                    node.out_dims.iter().product::<usize>()
                } else {
                    0usize
                }
            })
            .sum::<usize>();

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

    /// Convert f32 to field element
    fn f32_to_field(value: f32) -> Fp {
        if value < 0.0 {
            -Fp::from((-value * 1000.0) as u64)
        } else {
            Fp::from((value * 1000.0) as u64)
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
            if let NodeType::Node(node) = node {
                println!(
                    "Calculate space needed for operations node.out_dims: {:?}",
                    node.out_dims
                );
                match node.op_type {
                    OperationType::MatMul => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    }
                    OperationType::Relu => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    }
                    OperationType::Add => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    }
                    OperationType::Conv => {
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
            if let NodeType::Node(node) = node {
                match node.op_type {
                    OperationType::MatMul => {
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
                        if let Some(op_params) = &node.op_params {
                            for i in 0..output_size {
                                let mut sum = Fp::zero();
                                for j in 0..input_size {
                                    let param = Self::f32_to_field(op_params[i * input_size + j]);
                                    sum += param * input_values[j];
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
                    OperationType::Add => {
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
                    OperationType::Relu | OperationType::Max => {
                        if let Some((input_idx, _)) = node.inputs.first() {
                            if let Some(&input_row) = intermediate_values.get(input_idx) {
                                let size = node.out_dims.iter().product();
                                for i in 0..size {
                                    let x = witness[0][input_row + i];
                                    let result = if x == Fp::zero() { Fp::zero() } else { x };
                                    // Set the result in all columns
                                    for item in witness.iter_mut().take(COLUMNS) {
                                        item[current_row + i] = result;
                                    }
                                }
                                intermediate_values.insert(*idx, current_row);
                                current_row += size;
                            }
                        }
                    }
                    OperationType::Conv => {
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

    /// Verify a proof
    pub fn verify(
        &self,
        proof: &ProverProof<Vesta, ZkOpeningProof>,
        public_inputs: Option<&[Vec<f32>]>,
        public_outputs: Option<&[Vec<f32>]>,
    ) -> Result<bool, String> {
        let mut public_values = Vec::new();

        // Add public inputs if provided and input visibility is public
        if self.model.visibility.input == Visibility::Public {
            if let Some(inputs) = public_inputs {
                public_values.extend(
                    inputs
                        .iter()
                        .flat_map(|input| input.iter().map(|&x| Self::f32_to_field(x))),
                );
            } else {
                return Err("Public inputs required for verification".to_string());
            }
        }

        // Add public outputs if provided and output visibility is public
        if self.model.visibility.output == Visibility::Public {
            if let Some(outputs) = public_outputs {
                public_values.extend(
                    outputs
                        .iter()
                        .flat_map(|output| output.iter().map(|&x| Self::f32_to_field(x))),
                );
            } else {
                return Err("Public outputs required for verification".to_string());
            }
        }

        // Setup group map
        let group_map = <Vesta as CommitmentCurve>::Map::setup();

        // Verify proof
        kimchi::verifier::verify::<Vesta, BaseSponge, ScalarSponge, ZkOpeningProof>(
            &group_map,
            &self.verifier_index,
            proof,
            &public_values,
        )
        .map(|_| true)
        .map_err(|e| format!("Failed to verify proof: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use {RunArgs, VarVisibility, Visibility};

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

        // Create proof system
        let proof_system = ProofSystem::new(&model);

        // Create sample input
        let input = vec![vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];

        // Generate output and proof
        let prover_output = proof_system.prove(&input).expect("Failed to create proof");
        let outputs = prover_output.output.expect("Output should be public");

        // Verify the proof
        let result = proof_system
            .verify(&prover_output.proof, Some(&input), Some(&outputs))
            .expect("Failed to verify proof");

        assert!(result);
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

        // Create proof system
        let proof_system = ProofSystem::new(&model);

        // Create sample input
        let input = vec![vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];

        // Generate proof
        let prover_output = proof_system.prove(&input).expect("Failed to create proof");
        assert!(prover_output.output.is_none(), "Output should be private");

        // Verify the proof without public values
        let result = proof_system
            .verify(&prover_output.proof, None, None)
            .expect("Failed to verify proof");

        assert!(result);
    }
}
