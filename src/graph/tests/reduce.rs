use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

const REDUCE_ARGMAX_FALSE_IDX: i32 = 1;
const REDUCE_ARGMIN_FALSE_IDX: i32 = 3;
const REDUCE_SUM_FALSE_IDX: i32 = 7;

#[test]
fn test_reduce_argmax_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4]));

    // Reduce node (id: 1)

    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Reduce along axis 1
    attributes.insert("reducer".to_string(), vec![REDUCE_ARGMAX_FALSE_IDX]); // Reduce along axis 1

    nodes.insert(
        1,
        create_reduce_node(1, vec![(0, 0)], vec![1, 1], attributes),
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
fn test_reduce_argmax_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 2, 2]));

    // Reduce node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![2]); // Reduce along axis 2
    attributes.insert("reducer".to_string(), vec![REDUCE_ARGMAX_FALSE_IDX]); // Reduce along axis 1

    nodes.insert(
        1,
        create_reduce_node(1, vec![(0, 0)], vec![2, 2], attributes),
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

#[test]
fn test_reduce_argmin_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4]));

    // Reduce node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Reduce along axis 1
    attributes.insert("reducer".to_string(), vec![REDUCE_ARGMIN_FALSE_IDX]); // Use ArgMin reducer

    nodes.insert(
        1,
        create_reduce_node(1, vec![(0, 0)], vec![1, 1], attributes),
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
        10.0, 20.0, 15.0, 5.0, // Single row
    ];

    // Execute the graph
    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output: min indices along axis 1
    let expected_output = vec![3.0]; // Index of 5.0

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_reduce_argmin_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 2, 2]));

    // Reduce node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![2]); // Reduce along axis 2
    attributes.insert("reducer".to_string(), vec![REDUCE_ARGMIN_FALSE_IDX]); // Use ArgMin reducer

    nodes.insert(
        1,
        create_reduce_node(1, vec![(0, 0)], vec![2, 2], attributes),
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
        15.0, 5.0, // Row 1, Column 2
        50.0, 5.0, // Row 2, Column 1
        10.0, 35.0, // Row 2, Column 2
    ];

    // Execute the graph
    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output: min indices along axis 2
    let expected_output = vec![
        0.0, 1.0, // Indices of 10.0 and 5.0 in each sub-array
        1.0, 0.0, // Indices of 5.0 and 10.0 in each sub-array
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_reduce_sum_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Reduce node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Reduce along axis 1
    attributes.insert("reducer".to_string(), vec![REDUCE_SUM_FALSE_IDX]); // Use Sum reducer

    nodes.insert(
        1,
        create_reduce_node(1, vec![(0, 0)], vec![1, 1], attributes),
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

    // Expected output: Sum along axis 1
    let expected_output = vec![70.0]; // 10 + 20 + 15 + 25

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_reduce_sum_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 2, 2])); // Shape: [2, 2, 2]

    // Reduce node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![2]); // Reduce along axis 2
    attributes.insert("reducer".to_string(), vec![REDUCE_SUM_FALSE_IDX]); // Use Sum reducer

    nodes.insert(
        1,
        create_reduce_node(1, vec![(0, 0)], vec![2, 2], attributes),
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

    // Expected output: Sum along axis 2
    let expected_output = vec![
        30.0, 40.0, // Sums for Row 1
        55.0, 45.0, // Sums for Row 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
