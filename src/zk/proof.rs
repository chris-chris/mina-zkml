use kimchi::{
    circuits::{
        constraints::ConstraintSystem,
        domains::EvaluationDomains,
        wires::COLUMNS,
    },
    prover_index::ProverIndex,
    verifier_index::VerifierIndex,
    proof::ProverProof,
};
use mina_curves::pasta::{Fp, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use ark_ff::{Zero, UniformRand};
use ark_poly::EvaluationDomain;
use std::{sync::Arc, array};
use poly_commitment::{
    commitment::CommitmentCurve,
    ipa::SRS,
    SRS as _,
};
use groupmap::GroupMap;
use rand::{thread_rng, rngs::ThreadRng};

use super::wiring::ModelCircuitBuilder;
use crate::graph::model::Model;
use super::ZkOpeningProof;

type SpongeParams = PlonkSpongeConstantsKimchi;
type BaseSponge = DefaultFqSponge<VestaParameters, SpongeParams>;
type ScalarSponge = DefaultFrSponge<Fp, SpongeParams>;

/// Creates prover and verifier indices for a model
pub struct ProofSystem {
    pub prover_index: ProverIndex<Vesta, ZkOpeningProof>,
    pub verifier_index: VerifierIndex<Vesta, ZkOpeningProof>,
    domain: EvaluationDomains<Fp>,
    model: Model,
    domain_size: usize,
    zk_rows: usize,
}

impl ProofSystem {
    /// Create a new proof system from a model
    pub fn new(model: &Model) -> Self {
        // Convert model to circuit gates
        let mut builder = ModelCircuitBuilder::new();
        let (gates, domain_size, zk_rows) = builder.build_circuit(model);

        // Calculate total number of public inputs
        let num_public = model.graph.inputs.iter().map(|&idx| {
            if let crate::graph::model::NodeType::Node(node) = &model.graph.nodes[&idx] {
                node.out_dims.iter().product()
            } else {
                0
            }
        }).sum();

        println!("Number of public inputs: {}", num_public);
        println!("Required domain size: {}", domain_size);

        // Create constraint system with our domain size
        let cs = ConstraintSystem::create(gates)
            .public(num_public) // Set public input size
            .max_poly_size(Some(domain_size)) // Set max poly size to match domain size
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
            domain: cs.domain,
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
    fn create_witness(&self, inputs: &[Vec<f32>]) -> [Vec<Fp>; COLUMNS] {
        // Calculate initial witness size (without padding)
        let mut witness_size = 0;
        
        // Add space for public inputs
        let public_inputs: Vec<Fp> = inputs.iter()
            .flat_map(|input| input.iter().map(|&x| Self::f32_to_field(x)))
            .collect();
        witness_size += public_inputs.len();

        // Add space for intermediate computations
        for node in self.model.graph.nodes.values() {
            if let crate::graph::model::NodeType::Node(node) = node {
                match node.op_type {
                    crate::graph::model::OperationType::MatMul => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    },
                    crate::graph::model::OperationType::Relu => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    },
                    crate::graph::model::OperationType::Add => {
                        witness_size += node.out_dims.iter().product::<usize>();
                    },
                    _ => {}
                }
            }
        }

        println!("Creating witness with {} public inputs", public_inputs.len());

        // Ensure witness size is strictly less than domain_size - zk_rows
        assert!(witness_size < self.domain_size - self.zk_rows,
            "Witness size {} must be strictly less than domain size {} minus zk_rows {}",
            witness_size,
            self.domain_size,
            self.zk_rows
        );

        // Create witness arrays
        let mut witness = array::from_fn(|_| vec![Fp::zero(); self.domain_size]);

        // Place public inputs in witness
        for (i, &value) in public_inputs.iter().enumerate() {
            // Set the value in all columns for public inputs
            for col in 0..COLUMNS {
                witness[col][i] = value;
            }
        }

        // Process each node in topological order starting after public inputs
        let mut current_row = public_inputs.len();
        let mut intermediate_values = std::collections::HashMap::new();

        for (idx, node) in &self.model.graph.nodes {
            if let crate::graph::model::NodeType::Node(node) = node {
                match node.op_type {
                    crate::graph::model::OperationType::MatMul => {
                        let input_size = node.inputs[0].1;
                        let output_size = node.out_dims.iter().product();
                        
                        // Get input values
                        let input_values = if let Some((input_idx, _)) = node.inputs.first() {
                            intermediate_values.get(input_idx)
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
                                for col in 0..COLUMNS {
                                    witness[col][current_row + i] = sum;
                                }
                            }
                            intermediate_values.insert(*idx, current_row);
                            current_row += output_size;
                        }
                    },
                    crate::graph::model::OperationType::Relu => {
                        if let Some((input_idx, _)) = node.inputs.first() {
                            if let Some(&input_row) = intermediate_values.get(input_idx) {
                                let size = node.out_dims.iter().product();
                                for i in 0..size {
                                    let x = witness[0][input_row + i];
                                    let result = if x == Fp::zero() { Fp::zero() } else { x };
                                    // Set the result in all columns
                                    for col in 0..COLUMNS {
                                        witness[col][current_row + i] = result;
                                    }
                                }
                                intermediate_values.insert(*idx, current_row);
                                current_row += size;
                            }
                        }
                    },
                    crate::graph::model::OperationType::Add => {
                        if let (Some((left_idx, _)), Some((right_idx, _))) = (node.inputs.get(0), node.inputs.get(1)) {
                            if let (Some(&left_row), Some(&right_row)) = (intermediate_values.get(left_idx), intermediate_values.get(right_idx)) {
                                let size = node.out_dims.iter().product();
                                for i in 0..size {
                                    let result = witness[0][left_row + i] + witness[0][right_row + i];
                                    // Set the result in all columns
                                    for col in 0..COLUMNS {
                                        witness[col][current_row + i] = result;
                                    }
                                }
                                intermediate_values.insert(*idx, current_row);
                                current_row += size;
                            }
                        }
                    },
                    _ => {}
                }
            }
        }

        // Add random values for zero-knowledge rows at the end
        let mut rng = thread_rng();
        for col in 0..COLUMNS {
            for i in (self.domain_size - self.zk_rows)..self.domain_size {
                witness[col][i] = <Fp as UniformRand>::rand(&mut rng);
            }
        }

        // Pad remaining rows with zeros
        for col in 0..COLUMNS {
            for i in current_row..(self.domain_size - self.zk_rows) {
                witness[col][i] = Fp::zero();
            }
        }

        witness
    }

    /// Create a proof for model execution
    pub fn create_proof(
        &self,
        inputs: &[Vec<f32>],
    ) -> Result<ProverProof<Vesta, ZkOpeningProof>, String> {
        // Create witness with public inputs properly placed
        let witness = self.create_witness(inputs);

        // Convert inputs to public inputs
        let public_inputs: Vec<Fp> = inputs.iter()
            .flat_map(|input| input.iter().map(|&x| Self::f32_to_field(x)))
            .collect();

        println!("Creating proof with {} public inputs", public_inputs.len());

        // Setup group map
        let group_map = <Vesta as CommitmentCurve>::Map::setup();

        // Create proof with low-level API
        let mut rng = thread_rng();
        ProverProof::create::<BaseSponge, ScalarSponge, ThreadRng>(
            &group_map,
            witness,
            &[],  // No runtime tables
            &self.prover_index,
            &mut rng,
        ).map_err(|e| format!("Failed to create proof: {:?}", e))
    }

    /// Verify a proof
    pub fn verify_proof(
        &self,
        proof: &ProverProof<Vesta, ZkOpeningProof>,
        inputs: &[Vec<f32>],
    ) -> Result<bool, String> {
        // Convert inputs to public inputs
        let public_inputs: Vec<Fp> = inputs.iter()
            .flat_map(|input| input.iter().map(|&x| Self::f32_to_field(x)))
            .collect();

        println!("Verifying proof with {} public inputs", public_inputs.len());

        // Setup group map
        let group_map = <Vesta as CommitmentCurve>::Map::setup();

        // Verify proof
        kimchi::verifier::verify::<Vesta, BaseSponge, ScalarSponge, ZkOpeningProof>(
            &group_map,
            &self.verifier_index,
            proof,
            &public_inputs,
        ).map(|_| true)
        .map_err(|e| format!("Failed to verify proof: {:?}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{RunArgs, VarVisibility, Visibility};
    use std::collections::HashMap;

    #[test]
    fn test_proof_system() {
        // Create a simple model (perceptron)
        let mut variables = HashMap::new();
        variables.insert("batch_size".to_string(), 1);
        let run_args = RunArgs { variables };

        let visibility = VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        };

        let model = Model::new(
            "models/simple_perceptron.onnx",
            &run_args,
            &visibility,
        ).expect("Failed to load model");

        // Create proof system
        let proof_system = ProofSystem::new(&model);

        // Create sample input - pad to match expected size [1, 10]
        let input = vec![vec![
            1.0, 0.5, 0.3, 0.8, 0.2,  // Original values
            0.0, 0.0, 0.0, 0.0, 0.0   // Padding to reach size 10
        ]];

        println!("Creating proof with input size: {}", input[0].len());

        // Create and verify proof
        let proof = proof_system.create_proof(&input).expect("Failed to create proof");
        let result = proof_system.verify_proof(&proof, &input).expect("Failed to verify proof");

        assert!(result);
    }
}
