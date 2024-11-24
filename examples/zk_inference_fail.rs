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

    let model = Model::new(
        "models/simple_perceptron.onnx",
        &run_args,
        &visibility,
    )?;

    // 2. Create proof system
    println!("Creating proof system...");
    let proof_system = ProofSystem::new(&model);

    // 3. Create sample input (with proper padding to size 10)
    let input = vec![vec![
        1.0, 0.5, -0.3, 0.8, -0.2,  // Original values
        0.0, 0.0, 0.0, 0.0, 0.0     // Padding to reach size 10
    ]];

    // 4. Execute model normally to get actual output
    println!("Executing model...");
    let actual_output = model.graph.execute(&input)?;
    println!("Actual model output: {:?}", actual_output);

    // 5. Create zero-knowledge proof
    println!("Creating proof...");
    let proof = proof_system.create_proof(&input)?;

    // 6. Create incorrect output (random values)
    let incorrect_output = vec![vec![0.5, 0.7, 0.9]];  // Random values different from actual output
    println!("Incorrect output to verify against: {:?}", incorrect_output);

    // 7. Try to verify the proof with incorrect output
    println!("Attempting to verify proof with incorrect output...");
    let is_valid = proof_system.verify_proof(&proof, &input, &incorrect_output)?;

    println!("\nResults:");
    println!("Model execution successful: ✓");
    println!("Proof creation successful: ✓");
    println!("Proof verification: {}", if is_valid { "✓ Valid (UNEXPECTED!)" } else { "✗ Invalid (Expected)" });
    println!("Actual output: {:?}", actual_output);
    println!("Incorrect output used: {:?}", incorrect_output);
    println!("\nVerification failed as expected because the provided output doesn't match the actual model computation.");

    Ok(())
}
