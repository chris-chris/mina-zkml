use super::*;
use crate::graph::{
    model::{Model, NodeType, ParsedNodes, RunArgs, SerializableNode, VarVisibility, Visibility, OperationType},
    errors::GraphError,
};
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_matmul_operation() {
    // Create a graph with MatMul operation
    let mut nodes = BTreeMap::new();
    
    // Input nodes (id: 0, 1)
    let input_node1 = SerializableNode {
        inputs: vec![],
        out_dims: vec![2, 2], // 2x2 matrix
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
    };
    nodes.insert(0, NodeType::Node(input_node1));

    let input_node2 = SerializableNode {
        inputs: vec![],
        out_dims: vec![2, 2], // 2x2 matrix
        out_scale: 1,
        id: 1,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
    };
    nodes.insert(1, NodeType::Node(input_node2));

    // MatMul node (id: 2)
    let matmul_node = SerializableNode {
        inputs: vec![(0, 0), (1, 0)],
        out_dims: vec![2, 2], // Result is 2x2
        out_scale: 1,
        id: 2,
        op_type: OperationType::MatMul,
        weights: None,
        bias: None,
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

    // Test execution with 2x2 matrices
    // First matrix: [[1, 2], [3, 4]]
    let input1 = vec![1.0, 2.0, 3.0, 4.0];
    // Second matrix: [[5, 6], [7, 8]]
    let input2 = vec![5.0, 6.0, 7.0, 8.0];
    
    let result = model.graph.execute(&[input1.clone(), input2.clone()]).unwrap();

    // Expected result: [[19, 22], [43, 50]]
    // First row: 1*5 + 2*7 = 19, 1*6 + 2*8 = 22
    // Second row: 3*5 + 4*7 = 43, 3*6 + 4*8 = 50
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], vec![19.0, 22.0, 43.0, 50.0]);

    // Test invalid input dimensions
    let invalid_input1 = vec![1.0, 2.0, 3.0]; // 3 elements instead of 4
    let result = model.graph.execute(&[invalid_input1.clone(), input2]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));

    let result = model.graph.execute(&[input1, invalid_input1]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_relu_operation() {
    // Create a graph with ReLU operation
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

    // Test execution with various inputs
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let result = model.graph.execute(&[input]).unwrap();

    // Expected result: [0.0, 0.0, 1.0, 2.0]
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], vec![0.0, 0.0, 1.0, 2.0]);

    // Test with invalid input shape
    let invalid_input = vec![-1.0, 0.0, 1.0]; // 3 elements instead of 4
    let result = model.graph.execute(&[invalid_input]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_sigmoid_operation() {
    // Create a graph with Sigmoid operation
    let mut nodes = BTreeMap::new();
    
    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![3],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
    };
    nodes.insert(0, NodeType::Node(input_node));

    // Sigmoid node (id: 1)
    let sigmoid_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![3],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Sigmoid,
        weights: None,
        bias: None,
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

    // Test execution with various inputs
    let input = vec![-2.0, 0.0, 2.0];
    let result = model.graph.execute(&[input]).unwrap();

    // Expected result: sigmoid values for [-2.0, 0.0, 2.0]
    // sigmoid(x) = 1 / (1 + e^(-x))
    assert_eq!(result.len(), 1);
    assert!((result[0][0] - 0.119).abs() < 0.001); // sigmoid(-2) ≈ 0.119
    assert!((result[0][1] - 0.5).abs() < 0.001);   // sigmoid(0) = 0.5
    assert!((result[0][2] - 0.881).abs() < 0.001); // sigmoid(2) ≈ 0.881

    // Test with invalid input shape
    let invalid_input = vec![-2.0, 0.0]; // 2 elements instead of 3
    let result = model.graph.execute(&[invalid_input]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_error_handling() {
    // Create a graph with invalid input shapes
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
    };
    nodes.insert(0, NodeType::Node(input_node));

    // MatMul node with invalid input dimensions (id: 1)
    let matmul_node = SerializableNode {
        inputs: vec![(0, 0)], // MatMul needs two inputs
        out_dims: vec![2, 2],
        out_scale: 1,
        id: 1,
        op_type: OperationType::MatMul,
        weights: None,
        bias: None,
    };
    nodes.insert(1, NodeType::Node(matmul_node));

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

    // Test execution with invalid input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = model.graph.execute(&[input]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_const_operation() {
    // Create a graph with Const operation
    let mut nodes = BTreeMap::new();
    
    // Const node (id: 0)
    let const_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![3],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Const,
        weights: Some(vec![1.0, 2.0, 3.0]),
        bias: None,
    };
    nodes.insert(0, NodeType::Node(const_node));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![],
        outputs: vec![(0, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Test execution - Const nodes should output their weights
    let result = model.graph.execute(&[]).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
}
