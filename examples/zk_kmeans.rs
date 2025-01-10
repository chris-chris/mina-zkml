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

    // println!("\n=== Test Case 2: Valid Proof for Second Image ===");
    // // Load second image
    // let input2 = preprocess_image("models/data/1085.png")?;
    // let input_vec2 = vec![input2];

    // // Generate output and proof for second image
    // let prover_output2 = prover.prove(&input_vec2)?;
    // let output2 = prover_output2
    //     .output
    //     .as_ref()
    //     .expect("Output should be public");
    // println!("Second image prediction:");
    // print_prediction_info(&output2[0]);

    // // Verify proof for second image
    // let is_valid2 = verifier.verify(&prover_output2.proof, Some(&input_vec2), Some(output2))?;
    // println!(
    //     "Verification result: {}",
    //     if is_valid2 {
    //         "✓ Valid"
    //     } else {
    //         "✗ Invalid"
    //     }
    // );

    // println!("\n=== Test Case 3: Invalid Proof - Completely Wrong Outputs ===");
    // // Create fake output with opposite predictions
    // let mut fake_output1 = output1.clone();
    // for i in 0..10 {
    //     fake_output1[0][i] = -fake_output1[0][i]; // Invert all logits
    // }
    // println!("Attempted fake prediction:");
    // print_prediction_info(&fake_output1[0]);

    // // Try to verify with wrong outputs
    // let is_valid3 = verifier.verify(
    //     &prover_output1.proof,
    //     Some(&input_vec1),
    //     Some(&fake_output1),
    // )?;
    // println!(
    //     "Verification result: {}",
    //     if is_valid3 {
    //         "✓ Valid (UNEXPECTED!)"
    //     } else {
    //         "✗ Invalid (Expected)"
    //     }
    // );

    // println!("\n=== Test Case 4: Invalid Proof - Slightly Modified Outputs ===");
    // // Create fake output with small perturbations
    // let mut fake_output2 = output2.clone();
    // for i in 0..10 {
    //     fake_output2[0][i] += 0.1; // Add small perturbation to each logit
    // }
    // println!("Attempted fake prediction (with small perturbations):");
    // print_prediction_info(&fake_output2[0]);

    // // Try to verify with slightly modified outputs
    // let is_valid4 = verifier.verify(
    //     &prover_output2.proof,
    //     Some(&input_vec2),
    //     Some(&fake_output2),
    // )?;
    // println!(
    //     "Verification result: {}",
    //     if is_valid4 {
    //         "✓ Valid (UNEXPECTED!)"
    //     } else {
    //         "✗ Invalid (Expected)"
    //     }
    // );

    // println!("\n=== Summary ===");
    // println!(
    //     "1. First valid case (1052.png): {}",
    //     if is_valid1 {
    //         "✓ Valid"
    //     } else {
    //         "✗ Invalid"
    //     }
    // );
    // println!(
    //     "2. Second valid case (1085.png): {}",
    //     if is_valid2 {
    //         "✓ Valid"
    //     } else {
    //         "✗ Invalid"
    //     }
    // );
    // println!(
    //     "3. Invalid case (inverted logits): {}",
    //     if !is_valid3 {
    //         "✓ Failed as expected"
    //     } else {
    //         "✗ Unexpectedly passed"
    //     }
    // );
    // println!(
    //     "4. Invalid case (small perturbations): {}",
    //     if !is_valid4 {
    //         "✓ Failed as expected"
    //     } else {
    //         "✗ Unexpectedly passed"
    //     }
    // );

    // println!("\nThis demonstrates that the zero-knowledge proof system:");
    // println!("- Successfully verifies correct model executions");
    // println!("- Detects both large and small output manipulations");
    // println!("- Works consistently across different input images");

    // Ok(())
}
