use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_typedbinop_add_basic() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![1, 4])); // Shape: [1, 4]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("bin_op_idx".to_string(), vec![0]); // Add operation

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

    // Expected output: A + B
    let expected_output = vec![11.0, 22.0, 33.0, 44.0];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_typedbinop_add_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 2, 3])); // Shape: [2, 2, 3]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![2, 2, 3])); // Shape: [2, 2, 3]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("bin_op_idx".to_string(), vec![0]); // Add operation

    nodes.insert(
        2,
        create_typedbin_node(2, vec![(0, 0), (1, 0)], vec![2, 2, 3], attributes),
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
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Group 1
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Group 2
    ];
    let input_tensor_b = vec![
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // Group 1
        70.0, 80.0, 90.0, 100.0, 110.0, 120.0, // Group 2
    ];

    // Execute the graph
    let result = model
        .graph
        .execute(&[input_tensor_a, input_tensor_b])
        .unwrap();

    // Expected output: A + B
    let expected_output = vec![
        11.0, 22.0, 33.0, 44.0, 55.0, 66.0, // Group 1
        77.0, 88.0, 99.0, 110.0, 121.0, 132.0, // Group 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_typedbinop_sub_basic() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![1, 4])); // Shape: [1, 4]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("bin_op_idx".to_string(), vec![1]);

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
fn test_typedbinop_sub_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3])); // Shape: [2, 3]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![2, 3])); // Shape: [2, 3]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("bin_op_idx".to_string(), vec![1]); // Operation along axis 1

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

#[test]
fn test_typedbinop_pow_basic() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![1, 4])); // Shape: [1, 4]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("bin_op_idx".to_string(), vec![4]); // Pow operation

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
    let input_tensor_a = vec![2.0, 3.0, 4.0, 5.0]; // Base (A)
    let input_tensor_b = vec![2.0, 2.0, 2.0, 2.0]; // Exponent (B)

    // Execute the graph
    let result = model
        .graph
        .execute(&[input_tensor_a, input_tensor_b])
        .unwrap();

    // Expected output: A^B
    let expected_output = vec![4.0, 9.0, 16.0, 25.0];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_typedbinop_pow_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node A (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 2, 2])); // Shape: [2, 2, 2]

    // Input node B (id: 1)
    nodes.insert(1, create_input_node(1, vec![2, 2, 2])); // Shape: [2, 2, 2]

    // TypedBin node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("bin_op_idx".to_string(), vec![4]); // Pow operation

    nodes.insert(
        2,
        create_typedbin_node(2, vec![(0, 0), (1, 0)], vec![2, 2, 2], attributes),
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
        2.0, 3.0, 4.0, 5.0, // Group 1
        6.0, 7.0, 8.0, 9.0, // Group 2
    ];
    let input_tensor_b = vec![
        2.0, 2.0, 2.0, 2.0, // Group 1
        2.0, 2.0, 2.0, 2.0, // Group 2
    ];

    // Execute the graph
    let result = model
        .graph
        .execute(&[input_tensor_a, input_tensor_b])
        .unwrap();

    // Expected output: A^B
    let expected_output = vec![
        4.0, 9.0, 16.0, 25.0, // Group 1
        36.0, 49.0, 64.0, 81.0, // Group 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
