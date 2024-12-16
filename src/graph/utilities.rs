use super::errors::GraphError;
use super::model::*;
use std::collections::HashMap;
use tract_onnx::prelude::{Node as OnnxNode, SymbolValues, TypedFact, TypedOp};

pub fn node_output_shapes(
    node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
    symbol_values: &SymbolValues,
) -> Result<Vec<Vec<usize>>, GraphError> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
    for output in outputs {
        let shape = output.fact.shape;
        let shape = shape
            .eval_to_usize(symbol_values)
            .map_err(|_| GraphError::InvalidInputShape)?;
        let mv = shape.to_vec();
        shapes.push(mv)
    }
    Ok(shapes)
}

// Utility function to create an input node
pub fn create_input_node(id: usize, shape: Vec<usize>) -> NodeType {
    NodeType::Node(SerializableNode {
        inputs: vec![],
        out_dims: shape,
        out_scale: 1,
        id,
        op_type: OperationType::Input,
        op_params: None,
        attributes: HashMap::new(),
    })
}

// Utility function to create a constant node (weight or bias)
pub fn create_const_node(id: usize, shape: Vec<usize>, values: Vec<f32>) -> NodeType {
    NodeType::Node(SerializableNode {
        inputs: vec![],
        out_dims: shape,
        out_scale: 1,
        id,
        op_type: OperationType::Const,
        op_params: Some(values),
        attributes: HashMap::new(),
    })
}

// Utility function to create a Conv node
pub fn create_conv_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    NodeType::Node(SerializableNode {
        inputs,
        out_dims,
        out_scale: 1,
        id,
        op_type: OperationType::Conv,
        op_params: None,
        attributes: attributes
            .into_iter()
            .map(|(key, value)| {
                // Map each value to usize explicitly
                (
                    key,
                    value
                        .into_iter()
                        .map(|v| v as usize)
                        .collect::<Vec<usize>>(),
                )
            })
            .collect::<HashMap<String, Vec<usize>>>(), // Collect into HashMap<String, Vec<usize>>
    })
}

// Utility function to create a MaxPool node
pub fn create_max_pool_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    NodeType::Node(SerializableNode {
        inputs,
        out_dims,
        out_scale: 1,
        id,
        op_type: OperationType::MaxPool,
        op_params: None,
        attributes: attributes
            .into_iter()
            .map(|(key, value)| {
                // Map each value to usize explicitly
                (
                    key,
                    value
                        .into_iter()
                        .map(|v| v as usize)
                        .collect::<Vec<usize>>(),
                )
            })
            .collect::<HashMap<String, Vec<usize>>>(), // Collect into HashMap<String, Vec<usize>>
    })
}
