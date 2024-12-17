use mina_zkml::graph::{errors::GraphError, model::*, scales::*};
use std::collections::{BTreeMap, HashMap};
use std::env;

#[test]
fn test_model_resnet() {
    let run_args = RunArgs {
        variables: HashMap::from([("N".to_string(), 1)]),
    };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let model_path = build_model_path("models/resnet101-v1-7.onnx");
    test_model_with_args(&model_path, run_args, visibility);
}

#[test]
fn test_model_mnist() {
    let run_args = RunArgs {
        variables: HashMap::default(),
    };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let model_path = build_model_path("models/mnist-v1-12.onnx");
    test_model_with_args(&model_path, run_args, visibility);
}

#[test]
fn test_model_lenet() {
    let run_args = RunArgs {
        variables: HashMap::default(),
    };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let model_path = build_model_path("models/lenet.onnx");
    test_model_with_args(&model_path, run_args, visibility);
}

#[test]
fn test_model_cardionet() {
    let run_args = RunArgs {
        variables: HashMap::from([("batch_size".to_string(), 1)]),
    };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let model_path = build_model_path("models/cardionetv2uno.onnx");
    test_model_with_args(&model_path, run_args, visibility);
}

#[test]
fn test_model_smol_llm() {
    let run_args = RunArgs {
        variables: HashMap::from([
            ("batch_size".to_string(), 1),
            ("sequence_length".to_string(), 1),
            ("past_sequence_length".to_string(), 1),
        ]),
    };

    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    let model_path = build_model_path("models/smol-llm.onnx");

    test_model_with_args(&model_path, run_args, visibility);
}

#[test]
fn test_model_graph_traversal() {
    // Setup model loading parameters
    let run_args = RunArgs {
        variables: HashMap::from([("N".to_string(), 1)]),
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
        assert!(
            visited_nodes.contains_key(&node_id),
            "Output node {} is not reachable",
            node_id
        );
    }
}

#[test]
fn test_error_handling_integration() {
    // Test missing batch size
    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    // Test invalid model path
    let run_args = RunArgs {
        variables: HashMap::from([("N".to_string(), 1)]),
    };

    let result = Model::new("nonexistent.onnx", &run_args, &visibility);
    assert!(matches!(result, Err(GraphError::UnableToReadModel(_))));

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

/// Build the full path for a model file based on the current directory
fn build_model_path(model_relative_path: &str) -> String {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    println!("Current directory: {:?}", current_dir.display());
    let model_path = current_dir.join(model_relative_path);
    println!("Model path: {:?}", model_path.display());
    let model_path_str = model_path
        .to_str()
        .expect("Failed to convert model path to string")
        .to_string();
    model_path_str
}

/// Helper function to test a model with specific arguments
fn test_model_with_args(model_path: &str, run_args: RunArgs, visibility: VarVisibility) {
    // Load the model
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
                let input_scale = Scale::new(1);
                let params_scale = Scale::new(2);
                let output_scale = Scale::new(node.out_scale);
                let rebase_multiplier = 2.0;

                let var_scales =
                    VarScales::new(input_scale, params_scale, output_scale, rebase_multiplier);

                assert_eq!(var_scales.input.value(), 1);
                assert_eq!(var_scales.params.value(), 2);
                assert_eq!(var_scales.output.value(), node.out_scale);

                let test_value = 10.0;
                let rebased = var_scales.rebase(test_value);
                let unrebased = var_scales.unrebase(rebased);
                assert!((test_value - unrebased).abs() < f64::EPSILON);
            }
            NodeType::SubGraph { out_scales, .. } => {
                assert!(!out_scales.is_empty());
                for &scale in out_scales {
                    let scale_obj = Scale::new(scale);
                    assert_eq!(scale_obj.value(), scale);
                }
            }
        }
    }
}
