use tract_onnx::tract_hir::ops::math::exp;

use crate::graph::errors::GraphError;
use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_gather_1d() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Indices node (id: 1)
    nodes.insert(
        1,
        create_const_node(1, vec![2], vec![1.0, 3.0]), // Indices: [1, 3]
    );

    // Gather node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![1]); // Gather along axis 1

    nodes.insert(
        2,
        create_gather_node(2, vec![(0, 0), (1, 0)], vec![1, 2], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(2, 0)],
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
        10.0, 20.0, 30.0, 40.0, // Single row
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor]).unwrap();

    // Expected output: Gathered values at indices [1, 3]
    let expected_output = vec![20.0, 40.0];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_gather_2d() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3])); // Shape: [2, 3]

    // Indices node (id: 1)
    nodes.insert(
        1,
        create_const_node(1, vec![2], vec![0.0, 2.0]), // Indices: [0, 2]
    );

    // Gather node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![1]); // Gather along axis 1

    nodes.insert(
        2,
        create_gather_node(2, vec![(0, 0), (1, 0)], vec![2, 2], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(2, 0)],
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
        1.0, 2.0, 3.0, // Row 1
        4.0, 5.0, 6.0, // Row 2
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor]).unwrap();

    // Expected output: Gathered values at indices [0, 2]
    let expected_output = vec![
        1.0, 3.0, // Row 1
        4.0, 6.0, // Row 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_gather_3d() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3, 4])); // Shape: [2, 3, 4]

    // Indices node (id: 1)
    nodes.insert(
        1,
        create_const_node(1, vec![2], vec![1.0, 3.0]), // Indices: [1, 3]
    );

    // Gather node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![2]); // Gather along axis 2

    nodes.insert(
        2,
        create_gather_node(2, vec![(0, 0), (1, 0)], vec![2, 3, 2], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(2, 0)],
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
        1.0, 2.0, 3.0, 4.0, // Row 1, Column 1
        5.0, 6.0, 7.0, 8.0, // Row 1, Column 2
        9.0, 10.0, 11.0, 12.0, // Row 1, Column 3
        13.0, 14.0, 15.0, 16.0, // Row 2, Column 1
        17.0, 18.0, 19.0, 20.0, // Row 2, Column 2
        21.0, 22.0, 23.0, 24.0, // Row 2, Column 3
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor]).unwrap();

    // Expected output: Gather along axis 2 for indices [1, 3]
    let expected_output = vec![
        2.0, 4.0, // Row 1, Column 1
        6.0, 8.0, // Row 1, Column 2
        10.0, 12.0, // Row 1, Column 3
        14.0, 16.0, // Row 2, Column 1
        18.0, 20.0, // Row 2, Column 2
        22.0, 24.0, // Row 2, Column 3
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_gather_4d() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3, 4, 2])); // Shape: [2, 3, 4, 2]

    // Indices node (id: 1)
    nodes.insert(
        1,
        create_const_node(1, vec![3], vec![2.0, 3.0]), // Indices: [2, 3]
    );

    // Gather node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![2]); // Gather along axis 2

    nodes.insert(
        2,
        create_gather_node(2, vec![(0, 0), (1, 0)], vec![2, 3, 2, 2], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(2, 0)],
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
        // Row Group 1
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Column 1
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // Column 2
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, // Column 3
        // Row Group 2
        25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, // Column 1
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, // Column 2
        41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, // Column 3
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor]).unwrap();

    // Expected output: Gathered values at indices [2, 3] along axis 2
    let expected_output = vec![
        5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0, 21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0,
        37.0, 38.0, 39.0, 40.0, 45.0, 46.0, 47.0, 48.0,
    ];

    println!(
        "len result: {:?}, len expected: {:?}, ",
        result[0].len(),
        expected_output.len()
    );

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_gather_invalid_indices() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Indices node (id: 1)
    nodes.insert(
        1,
        create_const_node(1, vec![2], vec![4.0, 5.0]), // Invalid indices: [4, 5]
    );

    // Gather node (id: 2)
    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), vec![1]); // Gather along axis 1

    nodes.insert(
        2,
        create_gather_node(2, vec![(0, 0), (1, 0)], vec![1, 2], attributes),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(2, 0)],
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
        10.0, 20.0, 30.0, 40.0, // Single row
    ];

    // Execute the graph and expect an error
    let result = model.graph.execute(&[data_tensor]);

    assert!(result.is_err());
    if let Err(GraphError::InvalidInput(err)) = result {
        assert_eq!(err, "Gather: indices do not match data shape");
    } else {
        panic!("Expected GraphError::InvalidInput");
    }
}
