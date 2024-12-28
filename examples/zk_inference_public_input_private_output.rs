//! Example demonstrating zk inference with public input and private output

use mina_zkml::{
    graph::model::{Model, RunArgs, VarVisibility, Visibility},
    zk::proof::ProverSystem,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Public Input + Private Output ===");
    test_scenario(VarVisibility {
        input: Visibility::Public,
        output: Visibility::Private,
    })?;

    Ok(())
}

fn test_scenario(visibility: VarVisibility) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the model
    println!("Loading model...");
    let mut variables = HashMap::new();
    variables.insert("batch_size".to_string(), 1);
    let run_args = RunArgs { variables };

    let model = Model::new("models/simple_perceptron.onnx", &run_args, &visibility)?;

    // 2. Create prover system
    println!("Creating prover system...");
    let prover = ProverSystem::new(&model);
    let verifier = prover.verifier();

    // 3. Create sample input (with proper padding to size 10)
    let input = vec![vec![
        1.0, 0.5, -0.3, 0.8, -0.2, // Original values
        0.0, 0.0, 0.0, 0.0, 0.0, // Padding to reach size 10
    ]];

    // 4. Generate output and proof
    println!("Generating output and proof...");
    let prover_output = prover.prove(&input)?;

    println!("Model output is private");

    // 5. Verify the proof
    println!("Verifying proof...");
    let input_for_verify = Some(&input[..]); // Input is public
    let output_for_verify = None; // Output is private

    // Convert input to correct format for verification
    let formatted_input: Vec<Vec<f32>> = input
        .iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();

    let is_valid = verifier.verify(
        &prover_output.proof,
        Some(&formatted_input),
        output_for_verify,
    )?;

    println!("\nResults:");
    println!("Model execution successful: ✓");
    println!("Proof creation successful: ✓");
    println!(
        "Proof verification: {}",
        if is_valid { "✓ Valid" } else { "✗ Invalid" }
    );

    Ok(())
}
