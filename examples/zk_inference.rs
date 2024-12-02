use mina_zkml::{
    graph::model::{Model, RunArgs, VarVisibility, Visibility},
    zk::proof::ProofSystem,
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

    // 2. Create proof system
    println!("Creating proof system...");
    let proof_system = ProofSystem::new(&model);

    // 3. Create sample input (with proper padding to size 10)
    let input = vec![vec![
        1.0, 0.5, -0.3, 0.8, -0.2, // Original values
        0.0, 0.0, 0.0, 0.0, 0.0, // Padding to reach size 10
    ]];

    // 4. Generate output and proof
    println!("Generating output and proof...");
    let prover_output = proof_system.prove(&input)?;
    println!("Model output: {:?}", prover_output.output);

    // 5. Verify the proof with output and proof
    println!("Verifying proof...");
    let is_valid = proof_system.verify(&prover_output.output, &prover_output.proof)?;

    println!("\nResults:");
    println!("Model execution successful: ✓");
    println!("Proof creation successful: ✓");
    println!(
        "Proof verification: {}",
        if is_valid { "✓ Valid" } else { "✗ Invalid" }
    );

    // 6. Demonstrate invalid verification with modified output
    println!("\nTesting invalid case with modified output...");
    let mut modified_output = prover_output.output.clone();
    modified_output[0][0] += 1.0; // Modify first output value

    let is_valid_modified = proof_system.verify(&modified_output, &prover_output.proof)?;
    println!(
        "Modified output verification: {}",
        if !is_valid_modified {
            "✗ Invalid (Expected)"
        } else {
            "✓ Valid (Unexpected!)"
        }
    );

    Ok(())
}
