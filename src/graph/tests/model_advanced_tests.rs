use crate::graph::{
    errors::GraphError,
    model::{
        Model, NodeType, OperationType, ParsedNodes, SerializableNode, VarVisibility, Visibility,
    },
};
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_matrix_dimension_mismatch() {
    let mut nodes = BTreeMap::new();

    // Input nodes (id: 0, 1)
    let input_node1 = SerializableNode {
        inputs: vec![],
        out_dims: vec![2, 3], // 2x3 matrix
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node1));

    let input_node2 = SerializableNode {
        inputs: vec![],
        out_dims: vec![4, 2], // 4x2 matrix (incompatible dimensions)
        out_scale: 1,
        id: 1,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(input_node2));

    // MatMul node (id: 2)
    let matmul_node = SerializableNode {
        inputs: vec![(0, 0), (1, 0)],
        out_dims: vec![2, 2], // Result should be 2x2
        out_scale: 1,
        id: 2,
        op_type: OperationType::MatMul,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(2, NodeType::Node(matmul_node));

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

    // Test with incompatible matrix dimensions
    let input1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
    let input2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4x2 matrix
    let result = model.graph.execute(&[input1, input2]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_relu_edge_cases() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![4], // 1D vector of size 4
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // ReLU node (id: 1)
    let relu_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![4],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Relu,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(relu_node));

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

    // Test with very large negative numbers
    let input = vec![-1e10, -1e5, 1e5, 1e10];
    let result = model.graph.execute(&[input]).unwrap();
    assert_eq!(result[0], vec![0.0, 0.0, 1e5, 1e10]);

    // Test with zeros and small numbers
    let input = vec![-1e-10, 0.0, 1e-10, 1.0];
    let result = model.graph.execute(&[input]).unwrap();
    assert_eq!(result[0], vec![0.0, 0.0, 1e-10, 1.0]);

    // Test with special values
    let input = vec![f32::NEG_INFINITY, -0.0, 0.0, f32::INFINITY];
    let result = model.graph.execute(&[input]).unwrap();
    assert_eq!(result[0], vec![0.0, 0.0, 0.0, f32::INFINITY]);
}

#[test]
fn test_sigmoid_edge_cases() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![4],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // Sigmoid node (id: 1)
    let sigmoid_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![4],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Sigmoid,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(sigmoid_node));

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

    // Test with extreme values
    let input = vec![-1000.0, -20.0, 20.0, 1000.0];
    let result = model.graph.execute(&[input]).unwrap();
    assert!(
        (result[0][0] - 0.0).abs() < 1e-6,
        "Failed on extreme negative value"
    );
    assert!(
        (result[0][1] - 0.0).abs() < 1e-6,
        "Failed on large negative value"
    );
    assert!(
        (result[0][2] - 1.0).abs() < 1e-6,
        "Failed on large positive value"
    );
    assert!(
        (result[0][3] - 1.0).abs() < 1e-6,
        "Failed on extreme positive value"
    );

    // Test with zeros and small numbers
    let input = vec![-1e-10, 0.0, 1e-10, 1.0];
    let result = model.graph.execute(&[input]).unwrap();
    // For very small numbers (< 1e-7), we expect 0.0 due to numerical stability
    assert_eq!(result[0][0], 0.5, "Failed on small negative number");
    assert_eq!(result[0][1], 0.5, "Failed on zero");
    assert_eq!(result[0][2], 0.5, "Failed on small positive number");
    assert!(
        (result[0][3] - 0.7310586).abs() < 1e-6,
        "Failed on regular number"
    );

    // Test with special values
    let input = vec![f32::NEG_INFINITY, -0.0, 0.0, f32::INFINITY];
    let result = model.graph.execute(&[input]).unwrap();
    assert_eq!(
        result[0],
        vec![0.0, 0.5, 0.5, 1.0],
        "Failed on special values"
    );
}

#[test]
fn test_reshape_edge_cases() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![6],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // Reshape node (id: 1) - reshape to 2x3
    let reshape_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![2, 3],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Reshape,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(reshape_node));

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

    // Test with exact size match
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = model.graph.execute(&[input]).unwrap();
    assert_eq!(result[0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
