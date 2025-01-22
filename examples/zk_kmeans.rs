use mina_zkml::graph::model::{Model, RunArgs, VarVisibility, Visibility};
use mina_zkml::zk::proof::ProverSystem;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup model
    let mut variables = HashMap::new();
    variables.insert("batch_size".to_string(), 1);
    let run_args = RunArgs { variables };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let model = Model::new("models/kmeans.onnx", &run_args, &visibility)?;

    // 2. Create prover system
    println!("Creating prover system...");
    let prover = ProverSystem::new(&model);
    let verifier = prover.verifier();

    println!("\n=== Test Case 1: Valid Proof for First Image ===");
    // Load first image
    let input_vec1 = [vec![1.0f32, 1.0f32]];

    // Generate output and proof for first image
    let prover_output1 = prover.prove(&input_vec1)?;
    let output1 = prover_output1
        .output
        .as_ref()
        .expect("Output should be public");
    println!("First image prediction:");

    // Verify proof for first image
    let is_valid1 = verifier.verify(&prover_output1.proof, Some(&input_vec1), Some(output1))?;
    println!(
        "Verification result: {}",
        if is_valid1 {
            "✓ Valid"
        } else {
            "✗ Invalid"
        }
    );

    Ok(())
}
