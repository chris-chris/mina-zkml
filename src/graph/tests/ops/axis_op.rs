use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_add_axis_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![3, 4])); // Shape: [3, 4]

    // AddAxis node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![1]); // Add axis at dimension 1

    nodes.insert(
        1,
        create_add_axis_node(1, vec![(0, 0)], vec![3, 1, 4], attributes),
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

    // Data tensor
    let data_tensor = vec![
        1.0, 2.0, 3.0, 4.0, // Row 1
        5.0, 6.0, 7.0, 8.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor.clone()]).unwrap();

    // Expected output: Shape transformed to [3, 1, 4], values unchanged
    let expected_output = vec![
        1.0, 2.0, 3.0, 4.0, // Row 1
        5.0, 6.0, 7.0, 8.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_add_axis_multi_dimension() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3, 4])); // Shape: [2, 3, 4]

    // AddAxis node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![2]); // Add axis at dimension 2

    nodes.insert(
        1,
        create_add_axis_node(1, vec![(0, 0)], vec![2, 3, 4], attributes),
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

    // Data tensor
    let data_tensor = vec![
        // First batch
        1.0, 2.0, 3.0, 4.0, // Row 1
        5.0, 6.0, 7.0, 8.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
        // Second batch
        13.0, 14.0, 15.0, 16.0, // Row 1
        17.0, 18.0, 19.0, 20.0, // Row 2
        21.0, 22.0, 23.0, 24.0, // Row 3
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor.clone()]).unwrap();

    // Expected output: Shape transformed to [2, 3, 1, 4], values unchanged
    let expected_output = vec![
        // First batch
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Second batch
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
