use crate::graph::model::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_maxpool_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1, 1, 4, 4], // Shape: [Batch: 1, Channels: 1, Height: 4, Width: 4]
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // MaxPool node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![2, 2]); // Kernel: 2x2
    attributes.insert("strides".to_string(), vec![2, 2]); // Strides: 2
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]); // No padding

    let maxpool_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![1, 1, 2, 2], // Output: [Batch: 1, Channels: 1, Height: 2, Width: 2]
        out_scale: 1,
        id: 1,
        op_type: OperationType::MaxPool,
        op_params: None,
        attributes,
    };
    nodes.insert(1, NodeType::Node(maxpool_node));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(1, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensor
    let input_tensor = vec![
        1.0, 3.0, 2.0, 4.0, // Row 1
        5.0, 6.0, 8.0, 7.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
        13.0, 15.0, 14.0, 16.0, // Row 4
    ]; // Shape: [1, 1, 4, 4]

    // Execute the graph
    let result = model.graph.execute(&[input_tensor.clone()]).unwrap();

    // Expected output
    let expected_output = vec![
        6.0, 8.0, // Max of 2x2 blocks from Rows 1-2
        15.0, 16.0, // Max of 2x2 blocks from Rows 3-4
    ]; // Shape: [1, 1, 2, 2]

    // Assert that the result matches the expected output
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_maxpool_stride_3_kernel_3() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1, 1, 6, 6], // Shape: [Batch: 1, Channels: 1, Height: 6, Width: 6]
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // MaxPool node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]); // Kernel: 3x3
    attributes.insert("strides".to_string(), vec![3, 3]); // Strides: 3
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]); // No padding

    let maxpool_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![1, 2, 2, 1], // Output: [Batch: 1, Channels: 2, Height: 2, Width: 1]
        out_scale: 1,
        id: 1,
        op_type: OperationType::MaxPool,
        op_params: None,
        attributes,
    };
    nodes.insert(1, NodeType::Node(maxpool_node));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(1, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensor
    let input_tensor = vec![
        1.0, -2.0, 3.0, 0.5, -1.0, 2.0, // Channel 1, Row 1
        4.0, 5.0, 0.0, -3.0, 8.0, 6.0, // Channel 1, Row 2
        -7.0, 10.0, 9.0, -2.0, 4.0, 1.0, // Channel 1, Row 3
        3.0, 2.0, 6.0, 0.0, -1.0, -4.0, // Channel 1, Row 4
        5.0, -6.0, 9.0, 10.0, 7.0, -8.0, // Channel 1, Row 5
        1.0, -9.0, -5.0, -3.0, 0.0, 4.0, // Channel 1, Row 6
    ];

    let result = model.graph.execute(&[input_tensor.clone()]).unwrap();

    // Expected output
    let expected_output = vec![10.0, 8.0, 9.0, 10.0];

    // Assert that the result matches the expected output
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_maxpool_stride_4_kernel_4_with_two_channels() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1, 2, 6, 6], // Shape: [Batch: 1, Channels: 2, Height: 6, Width: 6]
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // MaxPool node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![4, 4]); // Kernel: 4x4
    attributes.insert("strides".to_string(), vec![4, 4]); // Strides: 4
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]); // No padding

    let maxpool_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![1, 2, 1, 1], // Output: [Batch: 1, Channels: 2, Height: 1, Width: 1]
        out_scale: 1,
        id: 1,
        op_type: OperationType::MaxPool,
        op_params: None,
        attributes,
    };
    nodes.insert(1, NodeType::Node(maxpool_node));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(1, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensor
    let input_tensor = vec![
        // Channel 1
        4.0, -2.0, 3.0, 8.0, -1.0, 6.0, // Row 1
        7.0, 5.0, -3.0, -6.0, 2.0, 4.0, // Row 2
        1.0, 10.0, -9.0, 0.0, 11.0, -8.0, // Row 3
        13.0, -7.0, 14.0, -2.0, 5.0, -4.0, // Row 4
        3.0, 2.0, 1.0, -8.0, 6.0, -5.0, // Row 5
        -1.0, 12.0, -6.0, 7.0, -3.0, 9.0, // Row 6
        // Channel 2
        -5.0, 1.0, 7.0, 10.0, -4.0, 2.0, // Row 1
        8.0, 6.0, -2.0, 3.0, 11.0, -8.0, // Row 2
        0.0, 15.0, -9.0, 4.0, -6.0, 14.0, // Row 3
        5.0, -1.0, 12.0, 22.0, -7.0, 8.0, // Row 4
        13.0, -3.0, -10.0, 6.0, -11.0, 0.0, // Row 5
        2.0, -12.0, 7.0, 16.0, -13.0, 3.0, // Row 6
    ];

    let result = model.graph.execute(&[input_tensor.clone()]).unwrap();

    // Expected output
    let expected_output = vec![
        14.0, // Max value for the 4x4 block in Channel 1
        22.0, // Max value for the 4x4 block in Channel 2
    ];

    // Assert that the result matches the expected output
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
