use anyhow::Result;
use base64::prelude::*;
use kimchi::verifier_index::VerifierIndex;
use mina_curves::pasta::{Fp, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use mina_zkml::{
    graph::model::{Model, RunArgs, VarVisibility, Visibility},
    zk::proof::{ProverSystem, VerifierSystem},
    zk::ZkOpeningProof,
};
use poly_commitment::{commitment::CommitmentCurve, ipa::SRS, SRS as _};
use groupmap::GroupMap;
use std::{collections::HashMap, sync::Arc};

type SpongeParams = PlonkSpongeConstantsKimchi;
type BaseSponge = DefaultFqSponge<VestaParameters, SpongeParams>;
type ScalarSponge = DefaultFrSponge<Fp, SpongeParams>;

fn main() -> Result<()> {
    // 1. Create a model with some sample data
    let run_args = RunArgs {
        variables: HashMap::from([("batch_size".to_string(), 1)]),
    };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    println!("Loading model...");
    let model = Model::new("models/simple_perceptron.onnx", &run_args, &visibility)?;

    // 2. Create prover system and generate a proof
    println!("Creating prover system...");
    let prover = ProverSystem::new(&model);
    
    // Sample input data
    let input = vec![vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
    
    println!("Generating proof...");
    let prover_output = prover.prove(&input).map_err(|e| {
        anyhow::anyhow!("Proof generation failed: {}", e)
    })?;

    // 3. Get verifier and serialize its index
    let verifier = prover.verifier();
    
    println!("Serializing verifier index...");
    let mut buf = Vec::new();
    rmp_serde::encode::write_named(&mut buf, &verifier.verifier_index)
        .map_err(|e| anyhow::anyhow!("Failed to serialize verifier index: {}", e))?;
    
    // Convert to base64 for storage/transmission
    let serialized = BASE64_STANDARD.encode(&buf);
    println!("Serialized verifier index (base64):\n{}\n", serialized);

    // 4. Simulate loading the verifier index from storage
    println!("Deserializing verifier index...");
    let bytes = BASE64_STANDARD.decode(&serialized)
        .map_err(|e| anyhow::anyhow!("Failed to decode base64: {}", e))?;
    
    // Create SRS for the loaded verifier
    let srs = SRS::<Vesta>::create(4096);
    let srs = Arc::new(srs);
    
    // Deserialize verifier index with proper SRS
    let mut loaded_verifier_index: VerifierIndex<Vesta, ZkOpeningProof> = rmp_serde::decode::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize verifier index: {}", e))?;
    
    // Set the SRS and other skipped fields in the loaded verifier index
    loaded_verifier_index.srs = Arc::clone(&srs);
    loaded_verifier_index.linearization = verifier.verifier_index.linearization.clone();
    loaded_verifier_index.powers_of_alpha = verifier.verifier_index.powers_of_alpha.clone();
    loaded_verifier_index.w = verifier.verifier_index.w.clone();
    loaded_verifier_index.permutation_vanishing_polynomial_m = verifier.verifier_index.permutation_vanishing_polynomial_m.clone();

    // Create new verifier from loaded index
    let loaded_verifier = VerifierSystem::new(loaded_verifier_index);

    // 5. Verify the proof using the loaded verifier
    println!("Verifying proof with loaded verifier...");
    
    // Setup group map for verification
    let group_map = <Vesta as CommitmentCurve>::Map::setup();

    // Convert inputs and outputs to field elements for verification
    let mut public_values = Vec::new();

    // Add public inputs
    for input_vec in &input {
        for &x in input_vec {
            const SCALE: f32 = 1_000_000.0;
            let scaled = (x * SCALE) as i64;
            let field_element = if scaled < 0 {
                -Fp::from((-scaled) as u64)
            } else {
                Fp::from(scaled as u64)
            };
            public_values.push(field_element);
        }
    }

    // Add public outputs
    if let Some(outputs) = &prover_output.output {
        for output_vec in outputs {
            for &x in output_vec {
                const SCALE: f32 = 1_000_000.0;
                let scaled = (x * SCALE) as i64;
                let field_element = if scaled < 0 {
                    -Fp::from((-scaled) as u64)
                } else {
                    Fp::from(scaled as u64)
                };
                public_values.push(field_element);
            }
        }
    }

    // Verify the proof with the loaded verifier
    kimchi::verifier::verify::<Vesta, BaseSponge, ScalarSponge, ZkOpeningProof>(
        &group_map,
        &loaded_verifier.verifier_index,
        &prover_output.proof,
        &public_values,
    ).map_err(|e| anyhow::anyhow!("Proof verification failed: {:?}", e))?;

    println!("✅ Proof verification successful!");
    if let Some(output) = prover_output.output {
        println!("\nModel output: {:?}", output);
    }

    Ok(())
}
