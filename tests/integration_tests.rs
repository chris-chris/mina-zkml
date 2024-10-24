use kimchi::graph::{
    model::*,
    errors::GraphError,
    scales::*,
};
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_model_load_and_scale_integration() {
    // Setup model loading parameters
    let run_args = RunArgs {
        variables: HashMap::from([
            ("N".to_string(), 1),
            ("C".to_string(), 3),
            ("H".to_string(), 224),
            ("W".to_string(), 224),
            ("batch_size".to_string(), 1),
            ("sequence_length".to_string(), 128),
        ]),
    };
    
    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    // Load the model
    let model_path = "models/resnet101-v1-7.onnx";
    let model = Model::new(model_path, &run_args, &visibility).unwrap();

    // Verify model structure
    assert!(model.graph.num_inputs() > 0);
    assert!(!model.graph.outputs.is_empty());

    // Test output scales
    let output_scales = model.graph.get_output_scales().unwrap();
    assert!(!output_scales.is_empty());

    // Create and test scales for each node
    for (_, node_type) in model.graph.nodes.iter() {
        match node_type {
            NodeType::Node(node) => {
                // Create scales for the node
                let input_scale = Scale::new(1);
                let params_scale = Scale::new(2);
                let output_scale = Scale::new(node.out_scale);
                let rebase_multiplier = 2.0;

                let var_scales = VarScales::new(
                    input_scale,
                    params_scale,
                    output_scale,
                    rebase_multiplier,
                );

                // Test scale operations
                assert_eq!(var_scales.input.value(), 1);
                assert_eq!(var_scales.params.value(), 2);
                assert_eq!(var_scales.output.value(), node.out_scale);

                // Test multiplier conversions
                let test_value = 10.0;
                let rebased = var_scales.rebase(test_value);
                let unrebased = var_scales.unrebase(rebased);
                assert!((test_value - unrebased).abs() < std::f64::EPSILON);
            }
            NodeType::SubGraph { out_scales, .. } => {
                // Test subgraph scales
                assert!(!out_scales.is_empty());
                for &scale in out_scales {
                    let scale_obj = Scale::new(scale);
                    assert_eq!(scale_obj.value(), scale);
                }
            }
        }
    }
}

#[test]
fn test_model_graph_traversal() {
    // Setup model loading parameters
    let run_args = RunArgs {
        variables: HashMap::from([
            ("N".to_string(), 1),
            ("C".to_string(), 3),
            ("H".to_string(), 224),
            ("W".to_string(), 224),
            ("batch_size".to_string(), 1),
            ("sequence_length".to_string(), 128),
        ]),
    };
    
    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    // Load the model
    let model_path = "models/resnet101-v1-7.onnx";
    let model = Model::new(model_path, &run_args, &visibility).unwrap();

    // Verify graph connectivity
    let mut visited_nodes = HashMap::new();
    let mut queue = std::collections::VecDeque::new();

    // Start from both input nodes and output nodes to ensure complete traversal
    for &input_node in &model.graph.inputs {
        queue.push_back(input_node);
    }

    for &(output_node, _) in &model.graph.outputs {
        queue.push_back(output_node);
    }

    // Traverse the graph
    while let Some(node_id) = queue.pop_front() {
        if visited_nodes.contains_key(&node_id) {
            continue;
        }

        visited_nodes.insert(node_id, true);

        if let Some(node_type) = model.graph.nodes.get(&node_id) {
            // Check node connections
            match node_type {
                NodeType::Node(node) => {
                    // Add input nodes to queue
                    for &(input_node, _) in &node.inputs {
                        if !visited_nodes.contains_key(&input_node) {
                            queue.push_back(input_node);
                        }
                    }
                }
                NodeType::SubGraph { inputs, .. } => {
                    // Add subgraph input nodes to queue
                    for input in inputs {
                        if !visited_nodes.contains_key(&input.node) {
                            queue.push_back(input.node);
                        }
                    }
                }
            }
        }
    }

    // Verify all nodes are reachable
    for &(node_id, _) in &model.graph.outputs {
        assert!(visited_nodes.contains_key(&node_id), "Output node {} is not reachable", node_id);
    }
}

#[test]
fn test_error_handling_integration() {
    // Test missing batch size
    let run_args = RunArgs {
        variables: HashMap::new(),
    };
    
    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let result = Model::new("models/resnet101-v1-7.onnx", &run_args, &visibility);
    assert!(matches!(result, Err(GraphError::InvalidInputShape)));

    // Test invalid model path
    let run_args = RunArgs {
        variables: HashMap::from([
            ("batch_size".to_string(), 1),
        ]),
    };

    let result = Model::new("nonexistent.onnx", &run_args, &visibility);
    assert!(matches!(result, Err(GraphError::UnableToReadModel)));

    // Test missing node error
    let nodes = BTreeMap::new();
    let invalid_node = 99999;
    let invalid_output = vec![(invalid_node, 0)];
    
    let parsed_nodes = ParsedNodes {
        nodes,
        inputs: vec![],
        outputs: invalid_output,
    };

    let scales_result = parsed_nodes.get_output_scales();
    assert!(matches!(scales_result, Err(GraphError::MissingNode(_))));
}
