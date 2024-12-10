use crate::graph::{
    errors::GraphError,
    model::{
        Model, NodeType, OperationType, ParsedNodes, SerializableNode, VarVisibility, Visibility,
    },
};
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_matmul_operation() {
    let mut nodes = BTreeMap::new();

    // Input nodes (id: 0, 1)
    let input_node1 = SerializableNode {
        inputs: vec![],
        out_dims: vec![2], // 2 elements vector
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
        out_dims: vec![2, 2], // 2x2 matrix
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
        out_dims: vec![2], // Result is 2 elements
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

    // Test execution with vector-matrix multiplication
    // Vector: [1, 2]
    let input1 = vec![1.0, 2.0];
    // Matrix: [[5, 6], [7, 8]]
    let input2 = vec![5.0, 6.0, 7.0, 8.0];

    let result = model
        .graph
        .execute(&[input1.clone(), input2.clone()])
        .unwrap();

    // Expected result: [17, 23]
    // First element: 1 * 5 + 2 * 6 = 17 (Transpose the vector for multiplication, same as pytorch)
    // Second element: 1 * 7 + 2 * 8 = 23
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], vec![17.0, 23.0]);

    // Test invalid input dimensions
    let invalid_input1 = vec![1.0, 2.0, 3.0]; // 3 elements instead of 2
    let result = model.graph.execute(&[invalid_input1, input2]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_relu_operation() {
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

    // Test execution with various inputs
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let result = model.graph.execute(&[input]).unwrap();

    // Expected result: [0.0, 0.0, 1.0, 2.0]
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], vec![0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid_operation() {
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
        attributes: HashMap::new(),
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

    // Test execution with various inputs
    let input = vec![-2.0, 0.0, 2.0];
    let result = model.graph.execute(&[input]).unwrap();

    // Expected result: sigmoid values for [-2.0, 0.0, 2.0]
    // sigmoid(x) = 1 / (1 + e^(-x))
    assert_eq!(result.len(), 1);
    assert!((result[0][0] - 0.119).abs() < 0.001); // sigmoid(-2) ≈ 0.119
    assert!((result[0][1] - 0.5).abs() < 0.001); // sigmoid(0) = 0.5
    assert!((result[0][2] - 0.881).abs() < 0.001); // sigmoid(2) ≈ 0.881

    // Test with invalid input shape
    let invalid_input = vec![-2.0, 0.0]; // 2 elements instead of 3
    let result = model.graph.execute(&[invalid_input]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_add_operation() {
    let mut nodes = BTreeMap::new();

    // Input nodes (id: 0, 1)
    let input_node1 = SerializableNode {
        inputs: vec![],
        out_dims: vec![3],
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
        out_dims: vec![3],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(input_node2));

    // Add node (id: 2)
    let add_node = SerializableNode {
        inputs: vec![(0, 0), (1, 0)],
        out_dims: vec![3],
        out_scale: 1,
        id: 2,
        op_type: OperationType::Add,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(2, NodeType::Node(add_node));

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

    // Test normal addition
    let input1 = vec![1.0, 2.0, 3.0];
    let input2 = vec![4.0, 5.0, 6.0];
    let result = model.graph.execute(&[input1, input2]).unwrap();
    assert_eq!(result[0], vec![5.0, 7.0, 9.0]);

    // Test with mismatched dimensions
    let input1 = vec![1.0, 2.0, 3.0];
    let input2 = vec![4.0, 5.0];
    let result = model.graph.execute(&[input1, input2]);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));
}

#[test]
fn test_einsum_operation() {
    let mut nodes = BTreeMap::new();

    // Input nodes (id: 0, 1)
    let input_node1 = SerializableNode {
        inputs: vec![],
        out_dims: vec![2], // 2 elements vector
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
        out_dims: vec![2, 2], // 2x2 matrix
        out_scale: 1,
        id: 1,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(input_node2));

    // EinSum node (id: 2)
    let einsum_node = SerializableNode {
        inputs: vec![(0, 0), (1, 0)],
        out_dims: vec![2], // Result is 2 elements
        out_scale: 1,
        id: 2,
        op_type: OperationType::EinSum,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(2, NodeType::Node(einsum_node));

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

    // Test EinSum operation (vector-matrix multiplication in this case)
    let input1 = vec![1.0, 2.0]; // 2-element vector
    let input2 = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
    let result = model.graph.execute(&[input1, input2]).unwrap();

    // Expected result: [17.0, 23.0]
    assert_eq!(result[0], vec![17.0, 23.0]);
}

#[test]
fn test_reshape_operation() {
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

    // Reshape node (id: 1)
    let reshape_node = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![2, 3], // Reshape 6 elements to 2x3 matrix
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

    // Test reshaping
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = model.graph.execute(&[input]).unwrap();
    assert_eq!(result[0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_const_operation() {
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
        attributes: HashMap::new(),
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
    assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_cyclic_dependency() {
    let mut nodes = BTreeMap::new();

    // Create a cycle: node 0 -> node 1 -> node 2 -> node 0
    let node0 = SerializableNode {
        inputs: vec![(2, 0)], // Creates cycle
        out_dims: vec![1],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Add,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(node0));

    let node1 = SerializableNode {
        inputs: vec![(0, 0)],
        out_dims: vec![1],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Add,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(node1));

    let node2 = SerializableNode {
        inputs: vec![(1, 0)],
        out_dims: vec![1],
        out_scale: 1,
        id: 2,
        op_type: OperationType::Add,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(2, NodeType::Node(node2));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![],
        outputs: vec![(2, 0)],
    };

    // Execute should fail due to cyclic dependency
    let result = graph.execute(&[]);
    assert!(matches!(result, Err(GraphError::CyclicDependency)));
}

#[test]
fn test_invalid_output_slot() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(0, 1)], // Invalid output slot (node 0 only has slot 0)
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    let result = model.graph.execute(&[vec![1.0]]);
    assert!(matches!(result, Err(GraphError::InvalidOutputSlot(1))));
}

#[test]
fn test_missing_node() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(0, NodeType::Node(input_node));

    // Add node referencing non-existent node (id: 1)
    let add_node = SerializableNode {
        inputs: vec![(0, 0), (2, 0)], // Node 2 doesn't exist
        out_dims: vec![1],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Add,
        weights: None,
        bias: None,
        attributes: HashMap::new(),
    };
    nodes.insert(1, NodeType::Node(add_node));

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

    let result = model.graph.execute(&[vec![1.0]]);
    assert!(matches!(result, Err(GraphError::MissingNode(2))));
}
