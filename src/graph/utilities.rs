use super::errors::GraphError;
use super::model::*;
use anyhow::Error;
use std::collections::HashMap;
use tract_onnx::prelude::{Node as OnnxNode, SymbolValues, TypedFact, TypedOp};
use tract_onnx::tract_core::ops::cnn::PoolSpec;
use tract_onnx::tract_core::ops::cnn::{KernelFormat, PaddingSpec};

/// Handles the PoolSpec attributes and inserts them into the provided attributes HashMap.
///
/// This function processes the kernel shape, strides, dilations, padding, and kernel format
/// from the given PoolSpec and inserts them into the attributes HashMap.
///
/// # Arguments
///
/// * `attributes` - A mutable reference to a HashMap where the attributes will be inserted.
/// * `pool_spec` - A reference to the PoolSpec containing the pooling specifications.
/// * `kernel_fmt` - An optional reference to the KernelFormat.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn handle_pool_spec(
    attributes: &mut HashMap<String, Vec<usize>>,
    pool_spec: &PoolSpec,
    kernel_fmt: &Option<KernelFormat>,
) {
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

/// Retrieves the output shapes of a given ONNX node.
///
/// This function evaluates the shapes of the outputs of the provided ONNX node
/// using the given symbol values and returns them as a vector of vectors of usize.
///
/// # Arguments
///
/// * `node` - A reference to the ONNX node.
/// * `symbol_values` - A reference to the symbol values used for evaluation.
///
/// # Returns
///
/// A Result containing a vector of vectors of usize representing the output shapes,
/// or a GraphError if the evaluation fails.
///
/// # Errors
///
/// Returns a GraphError if the shape evaluation fails.
///
/// # Safety
///
/// This function is safe to use.
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

/// Creates a new node with the given parameters.
///
/// This function creates a new node with the specified id, inputs, output dimensions,
/// operation type, attributes, and operation parameters.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `op_type` - The type of operation for the node.
/// * `attributes` - An optional HashMap of attributes for the node.
/// * `op_params` - An optional vector of operation parameters.
///
/// # Returns
///
/// A NodeType representing the created node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    op_type: OperationType,
    attributes: Option<HashMap<String, Vec<i32>>>,
    op_params: Option<Vec<f32>>,
) -> NodeType {
    NodeType::Node(SerializableNode {
        inputs,
        out_dims,
        out_scale: 1,
        id,
        op_type,
        op_params,
        attributes: attributes
            .unwrap_or_default()
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

/// Creates an Input node with the given id and shape.
///
/// This function creates a new Input node with the specified id and shape.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `shape` - A vector of usize representing the shape of the input.
///
/// # Returns
///
/// A NodeType representing the created Input node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_input_node(id: usize, shape: Vec<usize>) -> NodeType {
    create_node(id, vec![], shape, OperationType::Input, None, None)
}

/// Creates a Const node with the given id, shape, and values.
///
/// This function creates a new Const node with the specified id, shape, and values.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `shape` - A vector of usize representing the shape of the constant.
/// * `values` - A vector of f32 representing the values of the constant.
///
/// # Returns
///
/// A NodeType representing the created Const node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_const_node(id: usize, shape: Vec<usize>, values: Vec<f32>) -> NodeType {
    create_node(id, vec![], shape, OperationType::Const, None, Some(values))
}

/// Creates a Conv node with the given parameters.
///
/// This function creates a new Conv node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created Conv node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_conv_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::Conv,
        Some(attributes),
        None,
    )
}

/// Creates an AddAxis node with the given parameters.
///
/// This function creates a new AddAxis node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created AddAxis node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_add_axis_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::AddAxis,
        Some(attributes),
        None,
    )
}

/// Creates a Softmax node with the given parameters.
///
/// This function creates a new Softmax node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created Softmax node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_softmax_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::Softmax,
        Some(attributes),
        None,
    )
}

/// Creates a Gather node with the given parameters.
///
/// This function creates a new Gather node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created Gather node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_gather_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::Gather,
        Some(attributes),
        None,
    )
}

/// Creates a Reduce node with the given parameters.
///
/// This function creates a new Reduce node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created Reduce node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_reduce_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::Reduce,
        Some(attributes),
        None,
    )
}

/// Creates a TypedBinOp node with the given parameters.
///
/// This function creates a new TypedBinOp node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created TypedBinOp node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_typedbin_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::TypedBinOp,
        Some(attributes),
        None,
    )
}

/// Creates an ElementWiseOp node with the given parameters.
///
/// This function creates a new ElementWiseOp node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created ElementWiseOp node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_elementwise_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::ElementWiseOp,
        Some(attributes),
        None,
    )
}

/// Creates a MaxPool node with the given parameters.
///
/// This function creates a new MaxPool node with the specified id, inputs, output dimensions,
/// and attributes.
///
/// # Arguments
///
/// * `id` - The unique identifier for the node.
/// * `inputs` - A vector of tuples representing the input connections.
/// * `out_dims` - A vector of usize representing the output dimensions.
/// * `attributes` - A HashMap of attributes for the node.
///
/// # Returns
///
/// A NodeType representing the created MaxPool node.
///
/// # Panics
///
/// This function does not panic.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Safety
///
/// This function is safe to use.
pub fn create_max_pool_node(
    id: usize,
    inputs: Vec<(usize, usize)>,
    out_dims: Vec<usize>,
    attributes: HashMap<String, Vec<i32>>,
) -> NodeType {
    create_node(
        id,
        inputs,
        out_dims,
        OperationType::MaxPool,
        Some(attributes),
        None,
    )
}

/// Retrieves a value from the attributes HashMap.
///
/// This function retrieves the value associated with the given key from the attributes HashMap
/// and returns it as a vector of usize.
///
/// # Arguments
///
/// * `key` - A string slice representing the key to look up in the attributes HashMap.
/// * `attributes` - A reference to the attributes HashMap.
///
/// # Returns
///
/// A Result containing a vector of usize representing the value associated with the key,
/// or an Error if the key is not found.
///
/// # Errors
///
/// Returns a GraphError if the key is not found in the attributes HashMap.
///
/// # Safety
///
/// This function is safe to use.
pub fn get_value_from_attributes(
    key: &str,
    attributes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<usize>, Error> {
    let value: &Vec<usize> = attributes
        .get(&key.to_string())
        .ok_or(GraphError::MissingAttributes(key.to_string()))?;

    Ok(value.clone())
}
