use mina_zkml::{
    graph::model::{Model, RunArgs, VarVisibility, Visibility},
    zk::proof::ProofSystem,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test all visibility scenarios
    println!("\n=== Testing Public Model + Public Data ===");
    test_scenario(
        VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    )?;

    println!("\n=== Testing Private Model + Public Data ===");
    test_scenario(
        VarVisibility {
            input: Visibility::Public,
            output: Visibility::Private,
        },
    )?;

    println!("\n=== Testing Public Model + Private Data ===");
    test_scenario(
        VarVisibility {
            input: Visibility::Private,
            output: Visibility::Public,
        },
    )?;

    Ok(())
}

fn test_scenario(visibility: VarVisibility) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the model
    println!("Loading model...");
    let mut variables = HashMap::new();
    variables.insert("batch_size".to_string(), 1);
    let run_args = RunArgs { variables };

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
    
    // Print output if public
    if let Some(output) = &prover_output.output {
        println!("Model output (public): {:?}", output);
    } else {
        println!("Model output is private");
    }

    // 5. Verify the proof
    println!("Verifying proof...");
    let input_for_verify = if visibility.input == Visibility::Public {
        Some(&input[..])
    } else {
        None
    };
    
    let output_for_verify = if visibility.output == Visibility::Public {
        prover_output.output.as_deref()
    } else {
        None
    };

    let is_valid = proof_system.verify(
        &prover_output.proof,
        input_for_verify,
        output_for_verify,
    )?;

    println!("\nResults:");
    println!("Model execution successful: ✓");
    println!("Proof creation successful: ✓");
    println!(
        "Proof verification: {}",
        if is_valid { "✓ Valid" } else { "✗ Invalid" }
    );

    // 6. Test invalid case only for public output
    if visibility.output == Visibility::Public {
        println!("\nTesting invalid case with modified output...");
        if let Some(output) = prover_output.output {
            let mut modified_output = output.clone();
            modified_output[0][0] += 1.0; // Modify first output value

            let is_valid_modified = proof_system.verify(
                &prover_output.proof,
                input_for_verify,
                Some(&modified_output),
            )?;

            println!(
                "Modified output verification: {}",
                if !is_valid_modified {
                    "✗ Invalid (Expected)"
                } else {
                    "✓ Valid (Unexpected!)"
                }
            );
        }
    }

    Ok(())
}
