use mina_zkml::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use mina_zkml::graph::utilities::*;
use mina_zkml::zk::proof::ProofSystem;
use std::collections::{BTreeMap, HashMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Simple Conv Model Setup
    println!("Setting up a simple Conv model...");
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
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![1, 1]);
    attributes.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attributes.insert("dilations".to_string(), vec![1, 1]);

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 1, 2, 2],
            attributes,
        ),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
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
    let proof_system = ProofSystem::new(&model);

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
    println!("Prediction:");
    println!("Output: {:?}", prover_output.output);

    // 5. Verify proof
    let is_valid = proof_system.verify(&prover_output.output, &prover_output.proof)?;
    println!(
        "Verification result: {}",
        if is_valid { "✓ Valid" } else { "✗ Invalid" }
    );

    Ok(())
}
