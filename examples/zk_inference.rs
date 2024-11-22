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

    // 3. Create sample input
    let input = vec![vec![1.0, 0.5, -0.3, 0.8, -0.2]];

    // 4. Execute model normally
    println!("Executing model...");
    let output = model.graph.execute(&input)?;
    println!("Model output: {:?}", output);

    // 5. Create zero-knowledge proof
    println!("Creating proof...");
    let proof = proof_system.create_proof(&input)?;

    // 6. Verify the proof
    println!("Verifying proof...");
    let is_valid = proof_system.verify_proof(&proof, &input)?;

    println!("\nResults:");
    println!("Model execution successful: ✓");
    println!("Proof creation successful: ✓");
    println!("Proof verification: {}", if is_valid { "✓ Valid" } else { "✗ Invalid" });

    Ok(())
}
