use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_softmax_basic() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Softmax node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1]); // Softmax along axis 1

    nodes.insert(
        1,
        create_softmax_node(1, vec![(0, 0)], vec![1, 4], attributes),
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
        1.0, 2.0, 3.0, 4.0, // Single row
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor.clone()]).unwrap();

    // Expected output: Softmax values along axis 1
    let exp_values: Vec<f32> = data_tensor.iter().map(|&x| x.exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    let expected_output: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp).collect();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), expected_output.len());
    result[0]
        .iter()
        .zip(expected_output.iter())
        .for_each(|(res, exp)| assert!((res - exp).abs() < 1e-6));
}

#[test]
fn test_softmax_multiple_axes() {
    let mut nodes = BTreeMap::new();

    // Data node (id: 0)
    nodes.insert(0, create_input_node(0, vec![2, 3, 4])); // Shape: [2, 3, 4]

    // Softmax node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("axes".to_string(), vec![1, 2]); // Softmax along axes 1 and 2

    nodes.insert(
        1,
        create_softmax_node(1, vec![(0, 0)], vec![2, 3, 4], attributes),
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
        // Row group 1
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Row group 2
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];

    // Execute the graph
    let result = model.graph.execute(&[data_tensor.clone()]).unwrap();

    // Expected output: Softmax values across axes 1 and 2
    let exp_values: Vec<f32> = data_tensor.iter().map(|&x| x.exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    let expected_output: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp).collect();
    println!("expected_output: {:?}", expected_output);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), expected_output.len());
    result[0]
        .iter()
        .zip(expected_output.iter())
        .for_each(|(res, exp)| assert!((res - exp).abs() < 1e-6));
}
