use super::errors::GraphError;
use super::model::*;
use std::collections::HashMap;
use tract_onnx::prelude::{Node as OnnxNode, SymbolValues, TypedFact, TypedOp};
use tract_onnx::tract_core::ops::cnn::{KernelFormat, PaddingSpec};
use tract_onnx::tract_hir::ops::cnn::PoolSpec;

pub fn handle_pool_spec(
    attributes: &mut HashMap<String, Vec<usize>>,
    pool_spec: &PoolSpec,
    kernel_fmt: &Option<KernelFormat>,
) {
    println!("pool_spec: {:?} ", pool_spec);

    // Kernel shape
    attributes.insert(
        "kernel_shape".to_string(),
        pool_spec.kernel_shape.clone().into_iter().collect(),
    );

    // Strides
    if let Some(strides) = &pool_spec.strides {
        attributes.insert("strides".to_string(), strides.clone().into_iter().collect());
    }

    // Dilations
    if let Some(dilations) = &pool_spec.dilations {
        attributes.insert(
            "dilations".to_string(),
            dilations.clone().into_iter().collect(),
        );
    }

    // Padding
    match &pool_spec.padding {
        PaddingSpec::Explicit(before, after) => {
            let mut padding = before.clone();
            padding.extend(after.iter().cloned());
            attributes.insert("padding".to_string(), padding.into_vec());
        }
        PaddingSpec::ExplicitOnnxPool(before, after, count_include_pad) => {
            let mut padding = before.clone();
            padding.extend(after.iter().cloned());
            attributes.insert("padding".to_string(), padding.into_vec());
            attributes.insert(
                "count_include_pad".to_string(),
                vec![*count_include_pad as usize],
            );
        }
        PaddingSpec::Valid => {
            let kernel_rank = pool_spec.kernel_shape.len();
            attributes.insert("padding".to_string(), vec![0; kernel_rank * 2]);
        }
        _ => {
            let kernel_rank = pool_spec.kernel_shape.len();
            attributes.insert("padding".to_string(), vec![0; kernel_rank * 2]);
        }
    }

    // Kernel format
    attributes.insert(
        "kernel_format".to_string(),
        vec![match kernel_fmt {
            Some(KernelFormat::OIHW) => 0,
            Some(KernelFormat::HWIO) => 1,
            Some(KernelFormat::OHWI) => 2,
            _ => 3,
        }],
    );
}

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
            .map_err(|_| GraphError::InvalidInput("Utilities: node_output_shapes".to_string()))?;
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

// Utility function to create a Conv node
pub fn create_argmax_node(
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
        op_type: OperationType::ArgMax,
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
