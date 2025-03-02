use std::collections::{BTreeMap, HashMap};
use tract_onnx::tract_hir::ops::cnn::{PaddingSpec, PoolSpec};
use tract_onnx::tract_hir::ops::nn::DataFormat;

#[test]
fn test_deconv_basic() {
    use crate::graph::model::{
        Model, NodeType, OperationType, ParsedNodes, SerializableNode, VarVisibility, Visibility,
    };
    use crate::graph::utilities::insert_pool_spec_attributes;

    let input_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1, 1, 28, 28],
        out_scale: 1,
        id: 0,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    };

    let weight_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1, 1, 3, 3],
        out_scale: 1,
        id: 1,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    };

    let bias_node = SerializableNode {
        inputs: vec![],
        out_dims: vec![1],
        out_scale: 1,
        id: 2,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    };

    let mut attributes = HashMap::new();
    let pool_spec = PoolSpec {
        data_format: DataFormat::NCHW,
        kernel_shape: vec![3, 3].into(),
        padding: PaddingSpec::Explicit(vec![1, 1].into(), vec![1, 1].into()),
        dilations: None,
        strides: Some(vec![2, 2].into()),
        input_channels: 1,
        output_channels: 1,
    };
    insert_pool_spec_attributes(&mut attributes, &pool_spec);
    attributes.insert("kernel_format".to_string(), vec![0]);
    attributes.insert("adjustments".to_string(), vec![1, 1]);
    attributes.insert("group".to_string(), vec![1]);

    let deconv_node = SerializableNode {
        inputs: vec![(0, 0), (1, 0), (2, 0)],
        out_dims: vec![1, 1, 56, 56],
        out_scale: 1,
        id: 3,
        op_type: OperationType::Deconv,
        op_params: None,
        attributes,
    };

    let mut nodes = BTreeMap::new();
    nodes.insert(0, NodeType::Node(input_node));
    nodes.insert(1, NodeType::Node(weight_node));
    nodes.insert(2, NodeType::Node(bias_node));
    nodes.insert(3, NodeType::Node(deconv_node));

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0, 1, 2],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    let input_tensor = vec![1.0f32; 784];
    let weight_tensor = vec![1.0f32; 9];
    let bias_tensor = vec![0.0f32; 1];

    let input_data = [input_tensor, weight_tensor, bias_tensor];

    let result = model.graph.execute(&input_data).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1 * 1 * 56 * 56);
}
