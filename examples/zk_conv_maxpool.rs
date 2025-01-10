use mina_zkml::graph::model::*;
use mina_zkml::graph::utilities::*;
use mina_zkml::zk::proof::ProverSystem;
use std::collections::{BTreeMap, HashMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Simple Conv + MaxPool Model Setup
    println!("Setting up a simple Conv + MaxPool model...");
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 1, 4, 4]));

    // Weight node (id: 1)
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![1, 1, 3, 3],
            vec![
                1.0, 0.0, -1.0, // Kernel row 1
                1.0, 0.0, -1.0, // Kernel row 2
                1.0, 0.0, -1.0, // Kernel row 3
            ],
        ),
    );

    // Bias node (id: 2)
    nodes.insert(2, create_const_node(2, vec![1], vec![0.0])); // No bias

    // Conv node (id: 3)
    let mut conv_attributes = HashMap::new();
    conv_attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    conv_attributes.insert("strides".to_string(), vec![1, 1]);
    conv_attributes.insert("padding".to_string(), vec![0, 0, 0, 0]);
    conv_attributes.insert("dilations".to_string(), vec![1, 1]);

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 1, 2, 2],
            conv_attributes,
        ),
    );

    // MaxPool node (id: 4)
    let mut maxpool_attributes = HashMap::new();
    maxpool_attributes.insert("kernel_shape".to_string(), vec![2, 2]); // Kernel: 2x2
    maxpool_attributes.insert("strides".to_string(), vec![2, 2]); // Strides: 2
    maxpool_attributes.insert("padding".to_string(), vec![0, 0, 0, 0]); // No padding

    nodes.insert(
        4,
        create_max_pool_node(4, vec![(3, 0)], vec![1, 1, 1, 1], maxpool_attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(4, 0)], // Final output from MaxPool
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // 2. Create proof system
    println!("Creating proof system...");
    let proof_system = ProverSystem::new(&model);

    // 3. Input Tensor
    let input_tensor = vec![
        1.0, 2.0, 3.0, 4.0, // Row 1
        5.0, 6.0, 7.0, 8.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
        13.0, 14.0, 15.0, 16.0, // Row 4
    ];

    let input_vec = vec![input_tensor];

    // 4. Generate proof
    println!("Generating proof...");
    let prover_output = proof_system.prove(&input_vec)?;
    let output = prover_output
        .output
        .as_ref()
        .expect("Output should be public");
    println!("Prediction:");
    println!("Output: {:?}", prover_output.output);

    // 5. Verify proof
    let is_valid =
        proof_system
            .verifier()
            .verify(&prover_output.proof, Some(&input_vec), Some(output))?;
    println!(
        "Verification result: {}",
        if is_valid { "✓ Valid" } else { "✗ Invalid" }
    );

    Ok(())
}
