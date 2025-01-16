use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_typedbin_sub_basic() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![1, 4])); // Shape: [1, 4]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Operation along axis 1

    nodes.insert(
        2,
        create_typedbin_node(2, vec![(0, 0), (1, 0)], vec![1, 4], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0, 1],
        outputs: vec![(2, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensors
    let input_tensor_a = vec![10.0, 20.0, 30.0, 40.0]; // Input A
    let input_tensor_b = vec![1.0, 2.0, 3.0, 4.0]; // Input B

    // Execute the graph
    let result = model
        .graph
        .execute(&[input_tensor_a, input_tensor_b])
        .unwrap();

    // Expected output: A - B
    let expected_output = vec![9.0, 18.0, 27.0, 36.0];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_typedbin_sub_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3])); // Shape: [2, 3]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![2, 3])); // Shape: [2, 3]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Operation along axis 1

    nodes.insert(
        2,
        create_typedbin_node(2, vec![(0, 0), (1, 0)], vec![2, 3], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0, 1],
        outputs: vec![(2, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensors
    let input_tensor_a = vec![
        10.0, 20.0, 30.0, // Row 1
        40.0, 50.0, 60.0, // Row 2
    ]; // Input A
    let input_tensor_b = vec![
        1.0, 2.0, 3.0, // Row 1
        4.0, 5.0, 6.0, // Row 2
    ]; // Input B

    // Execute the graph
    let result = model
        .graph
        .execute(&[input_tensor_a, input_tensor_b])
        .unwrap();

    // Expected output: A - B
    let expected_output = vec![
        9.0, 18.0, 27.0, // Row 1
        36.0, 45.0, 54.0, // Row 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
