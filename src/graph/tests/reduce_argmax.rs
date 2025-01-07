use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_reduce_argmax_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4]));

    // ArgMax node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Reduce along axis 1

    nodes.insert(
        1,
        create_argmax_node(1, vec![(0, 0)], vec![1, 1], attributes),
    );

    // Graph setup
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
        10.0, 20.0, 15.0, 25.0, // Single row
    ];

    // Execute the graph
    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output: max indices along axis 1
    let expected_output = vec![3.0]; // Index of 25.0

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_reduce_argmax_axis() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 4]));

    // ArgMax node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Reduce along axis 1

    nodes.insert(
        1,
        create_argmax_node(1, vec![(0, 0)], vec![2, 1], attributes),
    );

    // Graph setup
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
        10.0, 20.0, 15.0, 25.0, // Row 1
        5.0, 50.0, 35.0, 10.0, // Row 2
    ];

    // Execute the graph
    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output: max indices along axis 1
    let expected_output = vec![3.0, 1.0]; // Indices of 25.0 and 50.0

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_reduce_argmax_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 2, 2]));

    // ArgMax node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![2]); // Reduce along axis 2

    nodes.insert(
        1,
        create_argmax_node(1, vec![(0, 0)], vec![2, 2], attributes),
    );

    // Graph setup
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
        10.0, 20.0, // Row 1, Column 1
        15.0, 25.0, // Row 1, Column 2
        5.0, 50.0, // Row 2, Column 1
        35.0, 10.0, // Row 2, Column 2
    ];

    // Execute the graph
    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output: max indices along axis 2
    let expected_output = vec![
        1.0, 1.0, // Indices of 20.0 and 25.0 in each sub-array
        1.0, 0.0, // Indices of 50.0 and 35.0 in each sub-array
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
