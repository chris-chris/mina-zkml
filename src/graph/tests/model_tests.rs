#[cfg(test)]
mod tests {
    use super::super::errors::GraphError;
    use super::super::model::Node;
    use super::super::model::*;
    use std::collections::{BTreeMap, HashMap};
    use std::env;
    use tract_data::internal::tract_smallvec::SmallVec;
    use tract_onnx::tract_hir::ops::cnn::{PaddingSpec, PoolSpec};
    use tract_onnx::tract_hir::ops::nn::DataFormat;
    use tract_onnx::{prelude::*, tract_core};

    #[test]
    fn test_model_load_invalid_path() {
        let run_args = RunArgs {
            variables: std::collections::HashMap::from([("batch_size".to_string(), 1)]),
        };

        let visibility = VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        };

        let model_path = "nonexistent.onnx";
        let result = Model::new(model_path, &run_args, &visibility);
        assert!(matches!(result, Err(GraphError::UnableToReadModel)));
    }

    #[test]
    fn test_model_load_success() -> Result<(), Box<dyn std::error::Error>> {
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

        // Get current directory path
        let current_dir = env::current_dir()?;
        println!("current directory: {:?}", current_dir.display());

        let model_path = current_dir.join("models/resnet101-v1-7.onnx");
        println!("Model path: {:?}", model_path);

        let model_path_str = model_path.to_str().ok_or("Invalid model path")?;

        let result = Model::new(model_path_str, &run_args, &visibility);
        assert!(result.is_ok());
        println!("result: {:?}", result);

        let model = result.unwrap();
        assert!(model.graph.num_inputs() > 0);
        Ok(())
    }

    #[test]
    fn test_parsed_nodes_output_scales() {
        let mut nodes: BTreeMap<usize, NodeType> = BTreeMap::new();
        let inputs: Vec<(usize, usize)> = vec![];
        let pool_spec: PoolSpec = PoolSpec::new(
            DataFormat::NHWC,
            SmallVec::from_buf([2, 2, 2, 2]),
            PaddingSpec::Valid,
            None,
            None,
            1,
            2,
        );

        // Create a Node first
        let node = Node {
            op: Box::new(tract_core::ops::cnn::MaxPool::new(pool_spec, None)),
            inputs: inputs.clone(),
            out_dims: vec![],
            out_scale: 1,
            id: 0,
        };

        // Convert Node to SerializableNode
        let serializable_node = SerializableNode {
            inputs: node.inputs,
            out_dims: node.out_dims,
            out_scale: node.out_scale,
            id: node.id,
        };

        nodes.insert(0, NodeType::Node(serializable_node));

        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: vec![],
            outputs: vec![(0, 0)],
        };

        let scales = parsed_nodes.get_output_scales().unwrap();
        assert_eq!(scales, vec![1]);
    }

    #[test]
    fn test_model_serialization() -> Result<(), Box<dyn std::error::Error>> {
        // Create a model using the existing test setup
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

        let current_dir = env::current_dir()?;
        let model_path = current_dir.join("models/resnet101-v1-7.onnx");
        let model_path_str = model_path.to_str().ok_or("Invalid model path")?;

        let model = Model::new(model_path_str, &run_args, &visibility)?;

        // Test saving
        let save_path = "test_model.bin";
        assert!(model.save(save_path).is_ok());

        // Test loading
        let loaded_model = Model::load(save_path);
        assert!(loaded_model.is_ok());

        // Clean up
        std::fs::remove_file(save_path)?;

        // Compare original and loaded models
        let loaded_model = loaded_model.unwrap();
        assert_eq!(model.visibility, loaded_model.visibility);
        assert_eq!(model.graph.inputs.len(), loaded_model.graph.inputs.len());
        assert_eq!(model.graph.outputs.len(), loaded_model.graph.outputs.len());
        Ok(())
    }

    #[test]
    fn test_model_save_error() -> Result<(), Box<dyn std::error::Error>> {
        // Create a model using the existing test setup
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

        let current_dir = env::current_dir()?;
        let model_path = current_dir.join("models/resnet101-v1-7.onnx");
        let model_path_str = model_path.to_str().ok_or("Invalid model path")?;

        let model = Model::new(model_path_str, &run_args, &visibility)?;

        // Test saving to an invalid path
        let result = model.save("/invalid/path/model.bin");
        assert!(matches!(result, Err(GraphError::UnableToSaveModel)));
        Ok(())
    }

    #[test]
    fn test_model_load_binary_error() {
        // Test loading from a non-existent file
        let result = Model::load("non_existent_model.bin");
        assert!(matches!(result, Err(GraphError::UnableToReadModel)));

        // Test loading from an invalid binary file
        let invalid_bin_path = "invalid_model.bin";
        std::fs::write(invalid_bin_path, "invalid binary content").unwrap();
        let result = Model::load(invalid_bin_path);
        assert!(matches!(result, Err(GraphError::UnableToReadModel)));
        std::fs::remove_file(invalid_bin_path).unwrap();
    }
}
