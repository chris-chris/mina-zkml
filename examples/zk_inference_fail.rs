use mina_zkml::{
    graph::model::{Model, RunArgs, VarVisibility, Visibility},
    zk::proof::ProverSystem,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the model
    println!("Loading model...");
    let mut variables = HashMap::new();
    variables.insert("batch_size".to_string(), 1);
    let run_args = RunArgs { variables };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

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
    let output = prover_output
        .output
        .as_ref()
        .expect("Output should be public");
    println!("Model output: {:?}", output);

    // 5. Create modified output (simulating malicious behavior)
    let mut modified_output = output.clone();
    modified_output[0][0] += 1.0; // Modify first output value

    // 6. Try to verify with modified output (should fail)
    println!("Verifying proof with modified output...");
    let is_valid = verifier.verify(&prover_output.proof, Some(&input), Some(&modified_output))?;

    println!("\nResults:");
    println!("Model execution successful: ✓");
    println!("Proof creation successful: ✓");
    println!(
        "Modified output verification: {}",
        if !is_valid {
            "✗ Invalid (Expected)"
        } else {
            "✓ Valid (Unexpected!)"
        }
    );

    Ok(())
}
