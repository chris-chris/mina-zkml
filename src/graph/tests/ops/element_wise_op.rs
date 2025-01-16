use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_elementwise_square_basic() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 4])); // Shape: [1, 4]

    // Elementwise node (id: 1)
    let mut attributes = HashMap::new();
    attributes.insert("element_wise_op_idx".to_string(), vec![3]); // Square operation index

    nodes.insert(
        1,
        create_elementwise_node(1, vec![(0, 0)], vec![1, 4], attributes),
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
    let input_tensor = vec![2.0, 3.0, 4.0, 5.0]; // Input values

    // Execute the graph
    let result = model.graph.execute(&[input_tensor.clone()]).unwrap();

    // Expected output: Square of input values
    let expected_output = input_tensor.iter().map(|x| x * x).collect::<Vec<f32>>();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), expected_output.len());
    result[0]
        .iter()
        .zip(expected_output.iter())
        .for_each(|(res, exp)| assert!((res - exp).abs() < 1e-6));
}
