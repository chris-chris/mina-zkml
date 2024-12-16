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
