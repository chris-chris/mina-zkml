//! This module defines the core structures and functions for representing and manipulating
//! neural network graphs. It includes the main `Model` structure, which contains the parsed
//! graph and variable visibility settings, as well as various enums and structs for representing
//! nodes, operations, and connections within the graph.

use super::errors::GraphError;
use super::utilities::handle_pool_spec;
use crate::graph::tract_integration::*;
use crate::graph::utilities::get_value_from_attributes;
use crate::zk::operations::identify_tract_operation;
use chrono::Local;
use instant;
use log::debug;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};
use tract_data::internal::tract_smallvec::smallvec;
use tract_data::internal::tract_smallvec::ToSmallVec;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::AxisOp;
use tract_onnx::tract_core::internal::ElementWiseMiniOp;
use tract_onnx::tract_core::ops::array::Gather;
use tract_onnx::tract_core::ops::binary::TypedBinOp;
use tract_onnx::tract_core::ops::cast::Cast;
use tract_onnx::tract_core::ops::cnn::{Conv, MaxPool};
use tract_onnx::tract_core::ops::element_wise::ElementWiseOp;
use tract_onnx::tract_core::ops::nn::{Reduce, Softmax, SoftmaxExp};
use tract_onnx::tract_core::ops::{konst::Const, scan::Scan, EvalOp};
use types::{CustomBinOp, CustomDatumType, CustomElementWiseOp, CustomReducer};

/// Type alias for the graph loading result
pub type GraphLoadResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

/// Represents a node output connection as (node_index, output_slot)
pub type Outlet = (usize, usize);

/// Enum representing different types of operations that can be performed in the graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OperationType {
    Input,
    MatMul,
    Relu,
    Sigmoid,
    Add,
    EinSum,
    Const,
    RmAxis,
    Reshape,
    Conv,
    MaxPool,
    Gather,
    Softmax,
    Reduce,
    AddAxis,
    Cast,
    TypedBinOp,
    ElementWiseOp,
}

/// Serializable version of OutletId
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableOutletId {
    pub node: usize,
    pub slot: usize,
}

impl From<OutletId> for SerializableOutletId {
    fn from(outlet: OutletId) -> Self {
        SerializableOutletId {
            node: outlet.node,
            slot: outlet.slot,
        }
    }
}

impl From<&OutletId> for SerializableOutletId {
    fn from(outlet: &OutletId) -> Self {
        SerializableOutletId {
            node: outlet.node,
            slot: outlet.slot,
        }
    }
}

/// Main model structure containing the parsed graph and variable visibility settings
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Model {
    pub graph: ParsedNodes,
    pub visibility: VarVisibility,
}

/// Represents different types of nodes in the graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeType {
    /// A regular computation node
    Node(SerializableNode),
    /// A subgraph node (typically used for control flow operations like loops)
    SubGraph {
        model: Box<Model>,
        inputs: Vec<SerializableOutletId>,
        idx: usize,
        out_dims: Vec<Vec<usize>>,
        out_scales: Vec<i32>,
        output_mappings: Vec<Vec<OutputMapping>>,
        input_mappings: Vec<InputMapping>,
    },
}

/// Represents the parsed neural network graph structure
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ParsedNodes {
    /// Map of node indices to their corresponding node types
    pub nodes: BTreeMap<usize, NodeType>,
    /// Indices of input nodes
    pub inputs: Vec<usize>,
    /// List of output connections (node_index, slot)
    pub outputs: Vec<Outlet>,
}

impl ParsedNodes {
    /// Returns the number of input nodes in the graph
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Logs the parameters and biases of operations in the graph to a file.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnableToSaveModel` if the log file cannot be created or written to.
    pub fn log_op_params_and_biases(&self) -> Result<(), GraphError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("op_params_biases_log.txt")
            .map_err(|_| GraphError::UnableToSaveModel)?;

        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

        writeln!(file, "\n[{}] Operation Params Analysis", timestamp)
            .map_err(|_| GraphError::UnableToSaveModel)?;

        writeln!(file, "----------------------------------------")
            .map_err(|_| GraphError::UnableToSaveModel)?;

        // Build connection map
        let mut const_connections: HashMap<usize, Vec<(usize, OperationType)>> = HashMap::new();
        for (node_idx, node_type) in &self.nodes {
            if let NodeType::Node(node) = node_type {
                for (input_idx, _slot) in &node.inputs {
                    if let Some(NodeType::Node(input_node)) = self.nodes.get(input_idx) {
                        if matches!(input_node.op_type, OperationType::Const) {
                            const_connections
                                .entry(*input_idx)
                                .or_default()
                                .push((*node_idx, node.op_type.clone()));
                        }
                    }
                }
            }
        }

        // Create a sorted list of nodes for consistent output
        let mut node_indices: Vec<_> = self.nodes.keys().collect();
        node_indices.sort();

        for &node_idx in &node_indices {
            if let Some(NodeType::Node(node)) = self.nodes.get(node_idx) {
                if matches!(node.op_type, OperationType::Const) {
                    // Node header
                    writeln!(file, "\nConst Node {}", node_idx)
                        .map_err(|_| GraphError::UnableToSaveModel)?;

                    // Dimensions
                    writeln!(file, "Dimensions: {:?}", node.out_dims)
                        .map_err(|_| GraphError::UnableToSaveModel)?;

                    // Consumers
                    if let Some(consumers) = const_connections.get(node_idx) {
                        writeln!(file, "Used by:").map_err(|_| GraphError::UnableToSaveModel)?;
                        for (consumer_idx, op_type) in consumers {
                            writeln!(file, "  - Node {} ({:?})", consumer_idx, op_type)
                                .map_err(|_| GraphError::UnableToSaveModel)?;
                        }
                    }

                    // Values
                    if let Some(op_params) = &node.op_params {
                        writeln!(file, "\nAll Values:")
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "Total elements: {}", op_params.len())
                            .map_err(|_| GraphError::UnableToSaveModel)?;

                        // Write all values as a comma-separated list within brackets
                        write!(file, "[").map_err(|_| GraphError::UnableToSaveModel)?;
                        for (i, &value) in op_params.iter().enumerate() {
                            if i > 0 {
                                write!(file, ", ").map_err(|_| GraphError::UnableToSaveModel)?;
                            }
                            write!(file, "{:.6}", value)
                                .map_err(|_| GraphError::UnableToSaveModel)?;
                        }
                        writeln!(file, "]").map_err(|_| GraphError::UnableToSaveModel)?;

                        // Statistics
                        let min = op_params.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let sum: f32 = op_params.iter().sum();
                        let mean = sum / op_params.len() as f32;

                        // Count non-zero elements
                        let non_zero_count = op_params.iter().filter(|&&x| x != 0.0).count();

                        writeln!(file, "\nStatistics:")
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Total elements: {}", op_params.len())
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Non-zero elements: {}", non_zero_count)
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(
                            file,
                            "  Zero elements: {}",
                            op_params.len() - non_zero_count
                        )
                        .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(
                            file,
                            "  Sparsity: {:.2}%",
                            (op_params.len() - non_zero_count) as f32 / op_params.len() as f32
                                * 100.0
                        )
                        .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Min: {:.6}", min)
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Mean: {:.6}", mean)
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                    }

                    writeln!(file, "----------------------------------------")
                        .map_err(|_| GraphError::UnableToSaveModel)?;
                }
            }
        }

        Ok(())
    }

    /// Returns a vector of output scales for all output nodes
    ///
    /// # Errors
    ///
    /// Returns `GraphError::MissingNode` if an output node is not found in the graph.
    pub fn get_output_scales(&self) -> Result<Vec<i32>, GraphError> {
        self.outputs
            .iter()
            .map(|&(node, slot)| {
                self.nodes
                    .get(&node)
                    .ok_or(GraphError::MissingNode(node))
                    .map(|n| match n {
                        NodeType::Node(node) => node.out_scale,
                        NodeType::SubGraph { out_scales, .. } => out_scales[slot],
                    })
            })
            .collect()
    }

    /// Execute the graph with given inputs
    ///
    /// # Errors
    ///
    /// Returns `GraphError` if there is an issue with the execution, such as missing nodes,
    /// invalid input slots, or unsupported operations.
    pub fn execute(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, GraphError> {
        let mut node_outputs: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();

        // Store input values
        for (&node_idx, input) in self.inputs.iter().zip(inputs.iter()) {
            // Get the input node to check its dimensions
            if let Some(NodeType::Node(node)) = self.nodes.get(&node_idx) {
                if node.out_dims.len() > 1 {
                    // If input node expects a tensor, reshape the input
                    node_outputs.insert(node_idx, vec![input.clone()]);
                } else {
                    node_outputs.insert(node_idx, vec![input.clone()]);
                }
            } else {
                node_outputs.insert(node_idx, vec![input.clone()]);
            }
        }

        // Topologically sort nodes for execution
        let sorted_nodes = self.topological_sort()?;

        // Execute nodes in order
        for &node_idx in sorted_nodes.iter() {
            if let Some(node_type) = self.nodes.get(&node_idx) {
                match node_type {
                    NodeType::Node(node) => {
                        // Handle Const nodes
                        if matches!(node.op_type, OperationType::Const) {
                            if let Some(op_params) = &node.op_params {
                                node_outputs.insert(node_idx, vec![op_params.clone()]);
                            }
                            continue;
                        }

                        // Skip input nodes as they're already processed
                        if matches!(node.op_type, OperationType::Input) {
                            continue;
                        }

                        // Get input values
                        let mut input_values = Vec::new();
                        for &(input_node, slot) in &node.inputs {
                            if let Some(outputs) = node_outputs.get(&input_node) {
                                if slot < outputs.len() {
                                    input_values.push(outputs[slot].clone());
                                } else {
                                    return Err(GraphError::InvalidInputSlot(slot));
                                }
                            } else {
                                return Err(GraphError::MissingNode(input_node));
                            }
                        }

                        // Execute operation
                        let output = self.execute_operation(node, &input_values)?;
                        node_outputs.insert(node_idx, output);
                    }
                    NodeType::SubGraph { .. } => {
                        return Err(GraphError::UnsupportedOperation);
                    }
                }
            }
        }

        // Collect outputs
        let mut outputs = Vec::new();
        for &(node, slot) in &self.outputs {
            if let Some(node_output) = node_outputs.get(&node) {
                if slot < node_output.len() {
                    outputs.push(node_output[slot].clone());
                } else {
                    return Err(GraphError::InvalidOutputSlot(slot));
                }
            } else {
                return Err(GraphError::MissingNode(node));
            }
        }

        Ok(outputs)
    }

    /// Execute a single operation
    ///
    /// # Errors
    ///
    /// Returns `GraphError` if there is an issue with the execution, such as invalid input lengths,
    /// missing nodes, or unsupported operations.
    fn execute_operation(
        &self,
        node: &SerializableNode,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GraphError> {
        let result = match node.op_type {
            OperationType::Input => Ok(inputs.to_vec()),
            OperationType::Cast => {
                if inputs.len() != 1 {
                    return Err(GraphError::InvalidInput(format!(
                        "Cast: input len({}) is invalid",
                        inputs.len()
                    )));
                }
                let res: Vec<f32> = inputs[0].to_vec();

                Ok(vec![res])
            }
            OperationType::TypedBinOp => {
                if inputs.len() != 2 {
                    return Err(GraphError::InvalidInput(format!(
                        "TypedBinOp: input len({}) is invalid",
                        inputs.len()
                    )));
                }

                // parse first input, a
                let a = self
                    .nodes
                    .get(&node.inputs[0].0)
                    .ok_or(GraphError::NodeNotFound)?;
                let a_dims: Vec<usize> = match a {
                    NodeType::Node(input) => input.out_dims.clone(),
                    _ => return Err(GraphError::InvalidNodeType),
                };
                let a_f64: Vec<f64> = inputs[0].iter().map(|&x| x as f64).collect();
                let a_tract = vec_to_tract_vec(&a_dims, &a_f64)?;

                // parse second input, b
                let b: &NodeType = self
                    .nodes
                    .get(&node.inputs[1].0)
                    .ok_or(GraphError::NodeNotFound)?;
                let b_dims: Vec<usize> = match b {
                    NodeType::Node(input) => input.out_dims.clone(),
                    _ => return Err(GraphError::InvalidNodeType),
                };
                let b_f64: Vec<f64> = inputs[1].iter().map(|&x| x as f64).collect();
                let b_tract = vec_to_tract_vec(&b_dims, &b_f64)?;

                let bin_op_inputs: TVec<TValue> = smallvec![a_tract[0].clone(), b_tract[0].clone()];

                // get TypedBinOp and evaluate it
                let typed_bin_op = {
                    let idx = get_value_from_attributes("bin_op_idx", &node.attributes)?;
                    let op = CustomBinOp::get_op_from_index(&idx[0]).ok_or_else(|| {
                        TractError::msg("TypedBinOp: failed to parse CustomBinOp index")
                    })?;
                    TypedBinOp(op, None)
                };
                let eval: TValue = {
                    let eval = typed_bin_op.eval(bin_op_inputs)?;
                    eval[0].clone()
                };

                let res_f64 = tensor_to_vec::<f64>(&eval.into_tensor())?;
                let res: Vec<f32> = res_f64.iter().map(|&x| x as f32).collect();

                Ok(vec![res])
            }
            OperationType::ElementWiseOp => {
                if inputs.len() != 1 {
                    return Err(GraphError::InvalidInput(format!(
                        "ElementWiseOp: input len({}) is invalid",
                        inputs.len()
                    )));
                }

                // parse first input, a
                let a = self
                    .nodes
                    .get(&node.inputs[0].0)
                    .ok_or(GraphError::NodeNotFound)?;
                let a_dims: Vec<usize> = match a {
                    NodeType::Node(input) => input.out_dims.clone(),
                    _ => return Err(GraphError::InvalidNodeType),
                };
                let a_f64: Vec<f64> = inputs[0].iter().map(|&x| x as f64).collect();
                let a_tract = vec_to_tract_vec(&a_dims, &a_f64)?;

                // get ElementWiseOp and evaluate it
                let element_wise_op = {
                    let idx = get_value_from_attributes("element_wise_op_idx", &node.attributes)?;
                    let op: Box<dyn ElementWiseMiniOp> =
                        CustomElementWiseOp::get_op_from_index(&idx[0]).ok_or_else(|| {
                            TractError::msg("TypedBinOp: failed to parse CustomBinOp index")
                        })?;
                    ElementWiseOp(op, None)
                };
                let eval: TValue = {
                    let eval = element_wise_op.eval(a_tract)?;
                    eval[0].clone()
                };

                let res_f64 = tensor_to_vec::<f64>(&eval.into_tensor())?;
                let res: Vec<f32> = res_f64.iter().map(|&x| x as f32).collect();

                Ok(vec![res])
            }
            OperationType::Reduce => {
                if inputs.len() != 1 {
                    return Err(GraphError::InvalidInput(format!(
                        "Reduce: input len({}) is invalid",
                        inputs.len()
                    )));
                }

                // parse input_dims
                let input_node = self
                    .nodes
                    .get(&node.inputs[0].0)
                    .ok_or(GraphError::NodeNotFound)?;
                let input_dims: Vec<usize> = match input_node {
                    NodeType::Node(input) => input.out_dims.clone(),
                    _ => return Err(GraphError::InvalidNodeType),
                };

                // convert input into tract vec
                let inputs_i64: Vec<i64> = inputs[0].iter().map(|&x| x as i64).collect();
                let tract_input = vec_to_tract_vec(&input_dims, &inputs_i64)?;

                // get axes from attributes
                let axes_vec: Vec<usize> = get_value_from_attributes("axes", &node.attributes)?;
                let mut axes: [usize; 4] = [usize::MAX; 4];
                for (i, &dim) in axes_vec.iter().enumerate() {
                    axes[i] = dim;
                }

                // get reducer from attributes
                let reducer = {
                    let reducer_idx: usize =
                        *get_value_from_attributes("reducer", &node.attributes)?
                            .first()
                            .unwrap_or(&0);

                    CustomReducer::get_reducer_from_index(reducer_idx)
                        .ok_or_else(|| TractError::msg("Reduce: failed to parse reducer index"))?
                };

                // return res from eval
                let reduce = Reduce {
                    axes: axes.to_smallvec(),
                    reducer,
                };
                let eval: TValue = {
                    let eval = reduce.eval(tract_input)?;
                    eval[0].clone()
                };
                let res_i64 = tensor_to_vec::<i64>(&eval.into_tensor())?;
                let res: Vec<f32> = res_i64.iter().map(|&x| x as f32).collect();

                Ok(vec![res])
            }
            OperationType::Softmax => {
                if inputs.len() != 1 {
                    return Err(GraphError::InvalidInput(format!(
                        "Softmax: input len({}) is invalid",
                        inputs.len()
                    )));
                }

                // convert input into tract vec
                let tract_input = vec_to_tract_vec(&node.out_dims, &inputs[0])?;

                // get axes from attributes
                let axes_vec: Vec<usize> = get_value_from_attributes("axes", &node.attributes)?;
                let mut axes: [usize; 4] = [0; 4];
                for (i, &dim) in axes_vec.iter().enumerate() {
                    axes[i] = dim;
                }

                // return res from softmax eval
                let softmax = Softmax {
                    axes: axes.to_smallvec(),
                    quant_output_dt: None,
                    exp: SoftmaxExp::Libc,
                };
                let eval: TValue = {
                    let eval = softmax.eval(tract_input)?;
                    eval[0].clone()
                };
                let res = tensor_to_vec::<f32>(&eval.into_tensor())?;

                Ok(vec![res])
            }
            OperationType::Gather => {
                if inputs.len() != 2 {
                    return Err(GraphError::InvalidInput(format!(
                        "Gather: input len({}) is invalid",
                        inputs.len()
                    )));
                }
                let data = &inputs[0]; // Data tensor
                let indices = &inputs[1]; // Indices tensor
                let axis = node
                    .attributes
                    .get("axis")
                    .and_then(|v| v.first())
                    .copied()
                    .ok_or(GraphError::MissingAttributes("Gather: axis".to_string()))?;

                // Ensure axis is valid
                let data_node = self
                    .nodes
                    .get(&node.inputs[0].0)
                    .ok_or(GraphError::NodeNotFound)?;
                let data_shape = if let NodeType::Node(n) = data_node {
                    n.out_dims.clone()
                } else {
                    return Err(GraphError::InvalidInput(
                        "Gather: Node is not SerializableNode".to_string(),
                    ));
                };

                if axis >= data_shape.len() {
                    return Err(GraphError::InvalidInput(format!(
                        "Gather: axis({}) is invalid",
                        axis
                    )));
                }

                // Validate indices
                if indices
                    .iter()
                    .any(|&i| i < 0.0 || i >= data_shape[axis] as f32)
                {
                    return Err(GraphError::InvalidInput(
                        "Gather: indices do not match data shape".to_string(),
                    ));
                }

                // Perform Gather operation
                let mut gathered_values = vec![];
                let outer_size = data_shape[..axis].iter().product::<usize>(); // Product of dimensions before axis
                let stride = data_shape.iter().skip(axis + 1).product::<usize>(); // Product of dimensions after axis

                for outer_offset in 0..outer_size {
                    for &index in indices {
                        let idx = index as usize;
                        let start = outer_offset * data_shape[axis] * stride + idx * stride;
                        let end = start + stride;
                        gathered_values.extend_from_slice(&data[start..end]);
                    }
                }

                Ok(vec![gathered_values]) // Return gathered values
            }
            OperationType::AddAxis => {
                if inputs.len() != 1 {
                    return Err(GraphError::InvalidInput(format!(
                        "AddAxis: input len({}) is invalid",
                        inputs.len()
                    )));
                }

                // get tensor_data
                let mut tensor = vec_to_tensor(&node.out_dims, &inputs[0])?;

                // get axes from attributes
                let axis_vec: Vec<usize> = get_value_from_attributes("axis", &node.attributes)?;

                // change tensor with AxisOp
                let add_axis = AxisOp::Add(axis_vec[0]);
                add_axis.change_tensor(&mut tensor, false)?;
                let res = tensor_to_vec::<f32>(&tensor)?;

                Ok(vec![res])
            }
            OperationType::Const => {
                if let Some(op_params) = &node.op_params {
                    Ok(vec![op_params.clone()])
                } else {
                    Err(GraphError::InvalidInput(
                        "Const: op_parms does not exist".to_string(),
                    ))
                }
            }
            OperationType::Conv => {
                if inputs.len() != 3 {
                    return Err(GraphError::InvalidInput(format!(
                        "Conv: Input len({})",
                        inputs.len()
                    )));
                }

                // Parse dimensions from inputs[0] and weight
                let input_node = self
                    .nodes
                    .get(&node.inputs[0].0)
                    .ok_or(GraphError::NodeNotFound)?;

                // Extract weights and biases
                let weight = match self.nodes.get(&node.inputs[1].0) {
                    Some(NodeType::Node(SerializableNode {
                        op_type: OperationType::Const,
                        op_params: Some(weights),
                        ..
                    })) => weights.clone(),
                    _ => return Err(GraphError::InvalidInput("Conv: Weight parsing".to_string())),
                };
                let bias = match self.nodes.get(&node.inputs[2].0) {
                    Some(NodeType::Node(SerializableNode {
                        op_type: OperationType::Const,
                        op_params: Some(bias),
                        ..
                    })) => bias.clone(),
                    _ => return Err(GraphError::InvalidInput("Conv: Bias parsing".to_string())),
                };

                if let NodeType::Node(input) = input_node {
                    let input_dims = &input.out_dims;
                    let batch_size = input_dims[0] as i32; // N
                    let input_channels = input_dims[1] as i32; // C
                    let input_height = input_dims[2] as i32; // H
                    let input_width = input_dims[3] as i32; // W

                    // Parse Conv parameters
                    let kernel_shape = node
                        .attributes
                        .get("kernel_shape")
                        .ok_or(GraphError::MissingAttributes(
                            "Conv: kernel_shape".to_string(),
                        ))?
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();
                    let strides = node
                        .attributes
                        .get("strides")
                        .unwrap_or(&vec![1, 1])
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();
                    let padding = node
                        .attributes
                        .get("padding")
                        .unwrap_or(&vec![0, 0, 0, 0])
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();
                    let dilations = node
                        .attributes
                        .get("dilations")
                        .unwrap_or(&vec![1, 1])
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();

                    // Use Conv node output dimensions directly
                    let output_dims = &node.out_dims;
                    let output_channels = output_dims[1] as i32;
                    let output_height = output_dims[2] as i32;
                    let output_width = output_dims[3] as i32;

                    // Validate weight shape
                    assert_eq!(
                        weight.len() as i32,
                        output_channels * input_channels * kernel_shape[0] * kernel_shape[1],
                        "Weight dimensions do not match expected shape."
                    );

                    // Initialize output tensor
                    let mut output = vec![
                        0.0;
                        (batch_size * output_channels * output_height * output_width)
                            as usize
                    ];

                    // Perform convolution
                    for n in 0..batch_size {
                        for oc in 0..output_channels {
                            for oh in 0..output_height {
                                for ow in 0..output_width {
                                    let mut sum = 0.0;
                                    for ic in 0..input_channels {
                                        for kh in 0..kernel_shape[0] {
                                            for kw in 0..kernel_shape[1] {
                                                let ih = oh * strides[0] + kh * dilations[0]
                                                    - padding[0];
                                                let iw = ow * strides[1] + kw * dilations[1]
                                                    - padding[1];
                                                if ih >= 0
                                                    && ih < input_height
                                                    && iw >= 0
                                                    && iw < input_width
                                                {
                                                    // Proper indexing of the input tensor
                                                    let input_idx = (((n * input_channels + ic)
                                                        * input_height
                                                        + ih)
                                                        * input_width
                                                        + iw)
                                                        as usize;
                                                    let weight_idx = (((oc * input_channels + ic)
                                                        * kernel_shape[0]
                                                        + kh)
                                                        * kernel_shape[1]
                                                        + kw)
                                                        as usize;
                                                    sum +=
                                                        inputs[0][input_idx] * weight[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                    // Add bias
                                    sum += bias[oc as usize];
                                    let output_idx =
                                        (((n * output_channels + oc) * output_height + oh)
                                            * output_width
                                            + ow) as usize;
                                    output[output_idx] = sum;
                                }
                            }
                        }
                    }

                    Ok(vec![output])
                } else {
                    Err(GraphError::InvalidNodeType)
                }
            }

            OperationType::MatMul | OperationType::EinSum => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInput(
                        "MatMul | EinSum: Empty input".to_string(),
                    ));
                }

                let input = &inputs[0]; // Shape: [784]
                let op_params = if inputs.len() > 1 {
                    &inputs[1]
                } else if let Some(op_params) = &node.op_params {
                    op_params
                } else {
                    return Err(GraphError::InvalidInput(
                        "MatMul | EinSum: op_parms parsing".to_string(),
                    ));
                };

                let input_dim = input.len(); // 784
                let output_dim = node.out_dims.iter().product(); // 512
                let weight_rows = output_dim; // 512 (PyTorch convention)
                let weight_cols = input_dim; // 784 (PyTorch convention)

                // Verify dimensions match
                if op_params.len() != weight_rows * weight_cols {
                    return Err(GraphError::InvalidInput(format!(
                        "MatMul | EinSum: op_params.len(): {}, weight_rows * weight_cols: {}",
                        op_params.len(),
                        weight_rows * weight_cols
                    )));
                }

                let mut output = vec![0.0; output_dim];

                // Using iterators instead of range loops
                output.iter_mut().enumerate().for_each(|(i, out)| {
                    *out = input.iter().enumerate().fold(0.0, |sum, (j, &input_val)| {
                        let weight_idx = i * input_dim + j;
                        sum + input_val * op_params[weight_idx]
                    });
                });

                Ok(vec![output])
            }
            OperationType::Add => {
                let a = &inputs[0];
                let b = if inputs.len() > 1 {
                    &inputs[1]
                } else {
                    return Err(GraphError::InvalidInput(
                        "Add: second val is not found".to_string(),
                    ));
                };

                if a.len() != b.len() {
                    return Err(GraphError::InvalidInput(
                        "Add: dimension of a nd b are not same".to_string(),
                    ));
                }
                Ok(vec![a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()])
            }
            OperationType::Relu => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInput("Relu: Empty input".to_string()));
                }

                let result = inputs[0].iter().map(|&x| x.max(0.0)).collect();
                Ok(vec![result])
            }
            OperationType::Sigmoid => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInput(
                        "Sigmoid: Input is empty".to_string(),
                    ));
                }

                let expected_size: usize = node.out_dims.iter().product();
                if inputs[0].len() != expected_size {
                    return Err(GraphError::InvalidInput(format!(
                        "Sigmoid: Input length({})",
                        inputs[0].len()
                    )));
                }

                Ok(vec![inputs[0]
                    .iter()
                    .map(|&x| {
                        if x > 20.0 {
                            1.0
                        } else if x < -20.0 {
                            0.0
                        } else {
                            1.0 / (1.0 + (-x).exp())
                        }
                    })
                    .collect()])
            }
            OperationType::RmAxis => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInput(
                        "RmAxis: Input is empty".to_string(),
                    ));
                }

                let expected_size: usize = node.out_dims.iter().product();
                let input = &inputs[0];

                if input.len() != expected_size {
                    return Err(GraphError::InvalidInput(format!(
                        "RmAxis: Input size({})",
                        input.len()
                    )));
                }

                Ok(vec![input.clone()])
            }
            OperationType::Reshape => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInput(
                        "Reshape: Input is empty".to_string(),
                    ));
                }
                Ok(vec![inputs[0].clone()])
            }
            OperationType::MaxPool => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInput(
                        "MaxPool: Input is empty".to_string(),
                    ));
                }

                let input_node = self
                    .nodes
                    .get(&node.inputs[0].0)
                    .ok_or(GraphError::NodeNotFound)?;

                if let NodeType::Node(input) = input_node {
                    let input_dims = &input.out_dims;
                    let batch_size = input_dims[0] as i32; // N
                    let input_channels = input_dims[1] as i32; // C
                    let input_height = input_dims[2] as i32; // H
                    let input_width = input_dims[3] as i32; // W
                                                            // Parse MaxPool parameters
                    let kernel_shape = node
                        .attributes
                        .get("kernel_shape")
                        .ok_or(GraphError::MissingAttributes(
                            "MaxPool: kernel_shape".to_string(),
                        ))?
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();
                    let strides = node
                        .attributes
                        .get("strides")
                        .unwrap_or(&vec![1, 1])
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();
                    let padding = node
                        .attributes
                        .get("padding")
                        .unwrap_or(&vec![0, 0, 0, 0])
                        .iter()
                        .map(|&x| x as i32)
                        .collect::<Vec<i32>>();

                    // Calculate output dimensions
                    let output_height =
                        (input_height + padding[0] + padding[2] - kernel_shape[0]) / strides[0] + 1;
                    let output_width =
                        (input_width + padding[1] + padding[3] - kernel_shape[1]) / strides[1] + 1;

                    // Initialize output tensor
                    let mut output = vec![
                        0.0;
                        (batch_size * input_channels * output_height * output_width)
                            as usize
                    ];

                    // Perform MaxPool
                    for n in 0..batch_size {
                        for c in 0..input_channels {
                            for oh in 0..output_height {
                                for ow in 0..output_width {
                                    let mut max_value = f32::NEG_INFINITY;

                                    for kh in 0..kernel_shape[0] {
                                        for kw in 0..kernel_shape[1] {
                                            let ih = oh * strides[0] + kh - padding[0];
                                            let iw = ow * strides[1] + kw - padding[1];

                                            // Ensure index is within bounds
                                            if ih >= 0
                                                && ih < input_height
                                                && iw >= 0
                                                && iw < input_width
                                            {
                                                let input_idx =
                                                    (((n * input_channels + c) * input_height + ih)
                                                        * input_width
                                                        + iw)
                                                        as usize;

                                                max_value = max_value.max(inputs[0][input_idx]);
                                            }
                                        }
                                    }

                                    let output_idx =
                                        (((n * input_channels + c) * output_height + oh)
                                            * output_width
                                            + ow) as usize;
                                    output[output_idx] = max_value;
                                }
                            }
                        }
                    }

                    Ok(vec![output])
                } else {
                    Err(GraphError::InvalidNodeType)
                }
            }
        };

        result
    }

    /// Perform topological sort of nodes
    ///
    /// # Errors
    ///
    /// Returns `GraphError::CyclicDependency` if a cyclic dependency is detected in the graph.
    fn topological_sort(&self) -> Result<Vec<usize>, GraphError> {
        let mut visited = HashMap::new();
        let mut sorted = Vec::new();

        fn visit(
            node: usize,
            visited: &mut HashMap<usize, bool>,
            sorted: &mut Vec<usize>,
            nodes: &BTreeMap<usize, NodeType>,
        ) -> Result<(), GraphError> {
            if let Some(&in_progress) = visited.get(&node) {
                if in_progress {
                    return Err(GraphError::CyclicDependency);
                }
                return Ok(());
            }

            visited.insert(node, true);

            if let Some(node_type) = nodes.get(&node) {
                match node_type {
                    NodeType::Node(node) => {
                        if !matches!(node.op_type, OperationType::Const) {
                            for &(input_node, _) in &node.inputs {
                                visit(input_node, visited, sorted, nodes)?;
                            }
                        }
                    }
                    NodeType::SubGraph { inputs, .. } => {
                        for input in inputs {
                            visit(input.node, visited, sorted, nodes)?;
                        }
                    }
                }
            }

            visited.insert(node, false);
            sorted.push(node);
            Ok(())
        }

        for &node in self.nodes.keys() {
            visit(node, &mut visited, &mut sorted, &self.nodes)?;
        }

        Ok(sorted)
    }
}

/// Serializable version of Node that excludes TypedOp
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableNode {
    /// Input connections to this node
    pub inputs: Vec<Outlet>,
    /// Output dimensions
    pub out_dims: Vec<usize>,
    /// Output scale factor
    pub out_scale: i32,
    /// Unique identifier for the node
    pub id: usize,
    /// Operation type
    pub op_type: OperationType,
    /// Parameters (op_params or bias)
    pub op_params: Option<Vec<f32>>,
    /// Attributes for the operations
    pub attributes: HashMap<String, Vec<usize>>,
}

impl From<&Node<TypedFact, Box<dyn TypedOp>>> for SerializableNode {
    fn from(node: &Node<TypedFact, Box<dyn TypedOp>>) -> Self {
        let op_name = node.op.name();
        let op_type: OperationType = if op_name == "Const" {
            println!("Found Const operation");
            OperationType::Const
        } else if node.inputs.is_empty() {
            println!("Found Input operation");
            OperationType::Input
        } else if let Some(op_type) = identify_tract_operation(node) {
            op_type
        } else {
            // TODO: Need an error handling. Default Operator should not be RmAxis
            // panic!("Unsupported operation: {}", op_name);
            println!("Unknown operation: {}", op_name);
            OperationType::RmAxis // Default to RmAxis for unknown operations
        };

        // Extract op_params Or attributes
        let op_params = match op_name.as_ref() {
            "Const" => {
                if let Some(const_op) = node.op.downcast_ref::<Const>() {
                    // TODO: should handle ALL supported types
                    if let Ok(tensor_data) = const_op.0.as_slice::<f32>() {
                        Some(tensor_data.to_vec())
                    } else if let Ok(tensor_data) = const_op.0.as_slice::<i32>() {
                        Some(tensor_data.to_vec().iter().map(|&x| x as f32).collect())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        // Extract convolution attributes
        let mut attributes = HashMap::new();
        if let Some(op) = node.op.as_any().downcast_ref::<Conv>() {
            handle_pool_spec(&mut attributes, &op.pool_spec, &Some(op.kernel_fmt));
        } else if let Some(op) = node.op.as_any().downcast_ref::<MaxPool>() {
            handle_pool_spec(&mut attributes, &op.pool_spec, &None);
        } else if let Some(op) = node.op.as_any().downcast_ref::<Reduce>() {
            attributes.insert("axes".to_string(), op.axes.to_vec());
            attributes.insert(
                "reducer".to_string(),
                vec![CustomReducer::get_index_from_reducer(op.reducer)],
            );
        } else if let Some(op) = node.op.as_any().downcast_ref::<Gather>() {
            attributes.insert("axis".to_string(), vec![op.axis]);
        } else if let Some(op) = node.op.as_any().downcast_ref::<Softmax>() {
            attributes.insert("axes".to_string(), op.axes.to_vec());
            // TODO: Cover exe and quant_output_dt
        } else if let Some(op) = node.op.as_any().downcast_ref::<AxisOp>() {
            match op {
                AxisOp::Add(i) => attributes.insert("axis".to_string(), vec![*i]),
                // TODO: Consider Rm, Move, Reshape
                _ => attributes.insert("axis".to_string(), vec![0]),
            };
        } else if let Some(op) = node.op.as_any().downcast_ref::<Cast>() {
            attributes.insert(
                "to".to_string(),
                vec![CustomDatumType::get_index_from_datum_type(op.to)],
            );
        } else if let Some(op) = node.op.as_any().downcast_ref::<TypedBinOp>() {
            // TODO: Consider all TypedBinOp ops
            let idx = match CustomBinOp::get_index_from_op(&*op.0) {
                Some(idx) => idx,
                None => {
                    println!("TypedBinOp: should be parsed between 0 to 6");
                    0
                }
            };
            attributes.insert("bin_op_idx".to_string(), vec![idx]);
        } else if let Some(op) = node.op.as_any().downcast_ref::<ElementWiseOp>() {
            // TODO: Consider all ElementWise ops
            let idx = match CustomElementWiseOp::get_index_from_op(&*op.0) {
                Some(idx) => idx,
                None => {
                    println!("ElementWiseOp: should be parsed between 0 to 24");
                    0
                }
            };
            attributes.insert("element_wise_op_idx".to_string(), vec![idx]);
        }

        SerializableNode {
            inputs: node.inputs.iter().map(|o| (o.node, o.slot)).collect(),
            out_dims: node.outputs[0]
                .fact
                .shape
                .iter()
                .map(|d| d.to_i64().unwrap() as usize)
                .collect(),
            out_scale: node.outputs[0]
                .fact
                .konst
                .as_ref()
                .map_or(1, |k| *k.to_scalar::<i32>().unwrap_or(&1)),
            id: node.id,
            op_type,
            op_params,
            attributes,
        }
    }
}

/// Arguments for running the model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunArgs {
    /// Map of variable names to their values
    pub variables: HashMap<String, usize>,
}

/// Controls visibility of variables in the model
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VarVisibility {
    pub input: Visibility,
    pub output: Visibility,
}

impl Default for VarVisibility {
    fn default() -> Self {
        VarVisibility {
            input: Visibility::Private,
            output: Visibility::Private,
        }
    }
}

/// Defines how inputs are mapped in subgraphs
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputMapping {
    /// Input is passed through completely
    Full,
    /// Input represents a state
    State,
    /// Input is stacked along an axis
    Stacked { axis: usize, chunk: usize },
}

/// Defines how outputs are mapped in subgraphs
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OutputMapping {
    /// Single output connection
    Single { outlet: usize, is_state: bool },
    /// Output stacked along an axis
    Stacked {
        outlet: usize,
        axis: usize,
        is_state: bool,
    },
}

impl OutputMapping {
    /// Returns whether this output represents a state
    pub fn is_state(&self) -> bool {
        match self {
            OutputMapping::Single { is_state, .. } => *is_state,
            OutputMapping::Stacked { is_state, .. } => *is_state,
        }
    }

    /// Returns the output slot index
    pub fn outlet(&self) -> usize {
        match self {
            OutputMapping::Single { outlet, .. } => *outlet,
            OutputMapping::Stacked { outlet, .. } => *outlet,
        }
    }
}

/// Variable visibility levels
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Visibility {
    Public,  // Visible externally
    Private, // Internal only
    Fixed,   // Cannot be modified
}

impl Model {
    /// Creates a new `Model` by loading an ONNX model from the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the ONNX model file.
    /// * `run_args` - The arguments for running the model.
    /// * `visibility` - The visibility settings for variables in the model.
    ///
    /// # Errors
    ///
    /// Returns `GraphError` if there is an issue loading the ONNX model.
    pub fn new(
        path: &str,
        run_args: &RunArgs,
        visibility: &VarVisibility,
    ) -> Result<Self, GraphError> {
        let parsed_nodes = Self::load_onnx_model(path, run_args, visibility)?;
        Ok(Model {
            graph: parsed_nodes,
            visibility: visibility.clone(),
        })
    }

    /// Loads an ONNX model and converts it into the internal `ParsedNodes` representation.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the ONNX model file.
    /// * `run_args` - The arguments for running the model.
    /// * `visibility` - The visibility settings for variables in the model.
    ///
    /// # Errors
    ///
    /// Returns `GraphError` if there is an issue loading or parsing the ONNX model.
    pub fn load_onnx_model(
        path: &str,
        run_args: &RunArgs,
        visibility: &VarVisibility,
    ) -> Result<ParsedNodes, GraphError> {
        let start = instant::Instant::now();
        let (model, symbol_values) = Self::load_onnx_using_tract(path, run_args)?;
        let nodes = Self::nodes_from_graph(&model, visibility.clone(), symbol_values)?;
        println!("Model loaded in {:?}", start.elapsed());

        // Collect all input nodes (nodes with OperationType::Input)
        let inputs: Vec<usize> = nodes
            .iter()
            .filter_map(|(idx, node)| match node {
                NodeType::Node(n) => {
                    if matches!(n.op_type, OperationType::Input) {
                        Some(*idx)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();

        let parsed_nodes = ParsedNodes {
            nodes,
            inputs,
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };
        Ok(parsed_nodes)
    }

    /// Loads an ONNX model using the `tract` library.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the ONNX model file.
    /// * `run_args` - The arguments for running the model.
    ///
    /// # Errors
    ///
    /// Returns `GraphError` if there is an issue loading or parsing the ONNX model.
    pub fn load_onnx_using_tract<P: AsRef<Path>>(
        path: P,
        run_args: &RunArgs,
    ) -> Result<GraphLoadResult, GraphError> {
        debug!("Starting load_onnx_using_tract");
        use tract_onnx::tract_hir::internal::GenericFactoid;
        let mut reader = std::fs::File::open(path).map_err(|err| {
            GraphError::UnableToReadModel(format!("Failed to read a file: {:?}", err))
        })?;

        let mut model = match tract_onnx::onnx().model_for_read(&mut reader) {
            Ok(model) => model,
            Err(err) => {
                return Err(GraphError::UnableToReadModel(format!(
                    "Failed to load a model: {:?}",
                    err
                )))
            }
        };

        let variables: std::collections::HashMap<String, usize> =
            std::collections::HashMap::from_iter(run_args.variables.clone());

        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node_mut(id.node);
            let mut fact: InferenceFact = input.outputs[0].fact.clone();

            for (i, x) in fact.clone().shape.dims().enumerate() {
                if matches!(x, GenericFactoid::Any) {
                    let batch_size = match variables.get("batch_size") {
                        Some(x) => x,
                        None => return Err(GraphError::MissingBatchSize),
                    };
                    fact.shape.set_dim(i, TDim::Val(*batch_size as i64));
                }
            }

            model.set_input_fact(i, fact).map_err(|err| {
                GraphError::UnableToReadModel(format!("Failed to set input fact: {:?}", err))
            })?;
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            match model.set_output_fact(i, InferenceFact::default()) {
                Ok(_) => (),
                Err(err) => {
                    return Err(GraphError::UnableToReadModel(format!(
                        "Failed to set output fact: {}",
                        err
                    )))
                }
            }
        }

        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbol_table.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
            debug!("set {} to {}", symbol, value);
        }

        let typed_model = model
            .into_typed()
            .map_err(|err| {
                GraphError::UnableToReadModel(format!("Failed to analyze and convert: {}", err))
            })?
            .concretize_dims(&symbol_values)
            .map_err(|err| {
                GraphError::UnableToReadModel(format!("Failed to concretize dims: {}", err))
            })?
            .into_decluttered()
            .map_err(|err| {
                GraphError::UnableToReadModel(format!("Failed to declutter: {}", err))
            })?;

        debug!("Completed load_onnx_using_tract successfully");
        Ok((typed_model, symbol_values))
    }

    /// Saves the model to a binary file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file where the model will be saved.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnableToSaveModel` if the model cannot be serialized or written to the file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let encoded: Vec<u8> =
            bincode::serialize(self).map_err(|_| GraphError::UnableToSaveModel)?;
        std::fs::write(path, encoded).map_err(|_| GraphError::UnableToSaveModel)
    }

    /// Loads a model from a binary file
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the binary file containing the model.
    ///
    /// # Errors
    ///
    /// Returns `GraphError::UnableToReadModel` if the model cannot be read or deserialized from the file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GraphError> {
        let bytes = std::fs::read(path).map_err(|err| {
            GraphError::UnableToReadModel(format!(
                "Failed to read a model from a binary file: {}",
                err
            ))
        })?;
        bincode::deserialize(&bytes)
            .map_err(|err| GraphError::UnableToReadModel(format!("Failed to deserialize: {}", err)))
    }

    /// Converts a tract graph into the internal node representation
    ///
    /// # Arguments
    ///
    /// * `graph` - The tract graph to be converted.
    /// * `visibility` - The visibility settings for variables in the model.
    /// * `symbol_values` - The symbol values for the graph.
    ///
    /// # Errors
    ///
    /// Returns `GraphError` if there is an issue converting the graph.
    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        visibility: VarVisibility,
        symbol_values: SymbolValues,
    ) -> Result<BTreeMap<usize, NodeType>, GraphError> {
        use super::utilities::node_output_shapes;
        let mut nodes = BTreeMap::new();

        // Process all nodes
        for (idx, node) in graph.nodes.iter().enumerate() {
            println!("Node: {:?}", node);
            match node.op().downcast_ref::<Scan>() {
                Some(scan_op) => {
                    debug!("Processing scan node {}", idx);

                    // Process input mappings
                    let mut input_mappings = vec![];
                    for mapping in &scan_op.input_mapping {
                        match mapping {
                            tract_onnx::tract_core::ops::scan::InputMapping::Scan(info) => {
                                input_mappings.push(InputMapping::Stacked {
                                    axis: info.axis,
                                    chunk: info.chunk as usize,
                                });
                            }
                            tract_onnx::tract_core::ops::scan::InputMapping::State => {
                                input_mappings.push(InputMapping::State);
                            }
                            tract_onnx::tract_core::ops::scan::InputMapping::Full => {
                                input_mappings.push(InputMapping::Full);
                            }
                        }
                    }

                    // Process output mappings
                    let mut output_mappings = vec![];
                    for (i, mapping) in scan_op.output_mapping.iter().enumerate() {
                        let mut mappings = vec![];
                        if let Some(outlet) = mapping.last_value_slot {
                            mappings.push(OutputMapping::Single {
                                outlet,
                                is_state: mapping.state,
                            });
                        } else if mapping.state {
                            mappings.push(OutputMapping::Single {
                                outlet: i,
                                is_state: mapping.state,
                            });
                        }
                        if let Some(last) = mapping.scan {
                            mappings.push(OutputMapping::Stacked {
                                outlet: last.0,
                                axis: last.1.axis,
                                is_state: false,
                            });
                        }
                        output_mappings.push(mappings);
                    }

                    // Process subgraph
                    let subgraph_nodes = Self::nodes_from_graph(
                        &scan_op.body,
                        visibility.clone(),
                        symbol_values.clone(),
                    )?;
                    let out_dims = node_output_shapes(node, &symbol_values)?;

                    nodes.insert(
                        idx,
                        NodeType::SubGraph {
                            model: Box::new(Model {
                                graph: ParsedNodes {
                                    nodes: subgraph_nodes,
                                    inputs: scan_op.body.inputs.iter().map(|o| o.node).collect(),
                                    outputs: scan_op
                                        .body
                                        .outputs
                                        .iter()
                                        .map(|o| (o.node, o.slot))
                                        .collect(),
                                },
                                visibility: visibility.clone(),
                            }),
                            inputs: node.inputs.iter().map(SerializableOutletId::from).collect(),
                            idx,
                            out_dims,
                            out_scales: vec![1; scan_op.output_mapping.len()],
                            output_mappings,
                            input_mappings,
                        },
                    );
                }
                None => {
                    debug!("Processing regular node {}", idx);

                    // Create the node with proper operation type and op_params
                    let serializable_node = SerializableNode::from(node);
                    nodes.insert(idx, NodeType::Node(serializable_node));
                }
            }
        }

        // Verify all required nodes exist
        let mut missing_nodes = Vec::new();

        // Check inputs for non-Const nodes
        for node in nodes.values() {
            match node {
                NodeType::Node(n) => {
                    if matches!(n.op_type, OperationType::Const) {
                        continue;
                    }
                    for &(input_node, _) in &n.inputs {
                        if !nodes.contains_key(&input_node) {
                            missing_nodes.push(input_node);
                        }
                    }
                }
                NodeType::SubGraph { inputs, .. } => {
                    for input in inputs {
                        if !nodes.contains_key(&input.node) {
                            missing_nodes.push(input.node);
                        }
                    }
                }
            }
        }

        // Check outputs
        for output in &graph.outputs {
            if !nodes.contains_key(&output.node) {
                missing_nodes.push(output.node);
            }
        }

        if !missing_nodes.is_empty() {
            debug!("Missing nodes: {:?}", missing_nodes);
            return Err(GraphError::MissingNode(missing_nodes[0]));
        }

        Ok(nodes)
    }

    /// Returns a string representation of the graph
    pub fn to_str(&self) -> Option<String> {
        let mut result = String::new();
        for (idx, node) in &self.graph.nodes {
            result.push_str(&format!("Node {}: {:?}\n", idx, node));
        }
        Some(result)
    }
}
