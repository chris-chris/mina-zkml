use super::errors::GraphError;
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
use tract_onnx::tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec};
use tract_onnx::{prelude::*, tract_hir::ops::konst::Const, tract_hir::ops::scan::Scan};

use crate::zk::operations::identify_tract_operation;

/// Type alias for the graph loading result
pub type GraphLoadResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

/// Represents a node output connection as (node_index, output_slot)
pub type Outlet = (usize, usize);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OperationType {
    Input,
    MatMul,
    Relu,
    Sigmoid,
    Add,
    EinSum,
    Max,
    Const,
    RmAxis,
    Reshape,
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

    pub fn log_weights_and_biases(&self) -> Result<(), GraphError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("weights_biases_log.txt")
            .map_err(|_| GraphError::UnableToSaveModel)?;

        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

        writeln!(file, "\n[{}] Weights and Biases Analysis", timestamp)
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
                    if let Some(weights) = &node.weights {
                        writeln!(file, "\nAll Values:")
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "Total elements: {}", weights.len())
                            .map_err(|_| GraphError::UnableToSaveModel)?;

                        // Write all values as a comma-separated list within brackets
                        write!(file, "[").map_err(|_| GraphError::UnableToSaveModel)?;
                        for (i, &value) in weights.iter().enumerate() {
                            if i > 0 {
                                write!(file, ", ").map_err(|_| GraphError::UnableToSaveModel)?;
                            }
                            write!(file, "{:.6}", value)
                                .map_err(|_| GraphError::UnableToSaveModel)?;
                        }
                        writeln!(file, "]").map_err(|_| GraphError::UnableToSaveModel)?;

                        // Statistics
                        let min = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let sum: f32 = weights.iter().sum();
                        let mean = sum / weights.len() as f32;

                        // Count non-zero elements
                        let non_zero_count = weights.iter().filter(|&&x| x != 0.0).count();

                        writeln!(file, "\nStatistics:")
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Total elements: {}", weights.len())
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Non-zero elements: {}", non_zero_count)
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Zero elements: {}", weights.len() - non_zero_count)
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(
                            file,
                            "  Sparsity: {:.2}%",
                            (weights.len() - non_zero_count) as f32 / weights.len() as f32 * 100.0
                        )
                        .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Min: {:.6}", min)
                            .map_err(|_| GraphError::UnableToSaveModel)?;
                        writeln!(file, "  Max: {:.6}", max)
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
                            if let Some(weights) = &node.weights {
                                node_outputs.insert(node_idx, vec![weights.clone()]);
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
    fn execute_operation(
        &self,
        node: &SerializableNode,
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GraphError> {
        let result = match node.op_type {
            OperationType::Input => Ok(inputs.to_vec()),
            OperationType::Const => {
                if let Some(weights) = &node.weights {
                    Ok(vec![weights.clone()])
                } else {
                    Err(GraphError::InvalidInputShape)
                }
            }
            OperationType::MatMul | OperationType::EinSum => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInputShape);
                }

                let input = &inputs[0]; // Shape: [784]
                let weights = if inputs.len() > 1 {
                    &inputs[1]
                } else if let Some(weights) = &node.weights {
                    weights
                } else {
                    return Err(GraphError::InvalidInputShape);
                };

                let input_dim = input.len(); // 784
                let output_dim = node.out_dims.iter().product(); // 512
                let weight_rows = output_dim; // 512 (PyTorch convention)
                let weight_cols = input_dim; // 784 (PyTorch convention)

                // Verify dimensions match
                if weights.len() != weight_rows * weight_cols {
                    return Err(GraphError::InvalidInputShape);
                }

                let mut output = vec![0.0; output_dim];

                // Using iterators instead of range loops
                output.iter_mut().enumerate().for_each(|(i, out)| {
                    *out = input.iter().enumerate().fold(0.0, |sum, (j, &input_val)| {
                        let weight_idx = i * input_dim + j;
                        sum + input_val * weights[weight_idx]
                    });
                });

                Ok(vec![output])
            }
            OperationType::Add => {
                let a = &inputs[0];
                let b = if inputs.len() > 1 {
                    &inputs[1]
                } else if let Some(bias) = &node.bias {
                    bias
                } else {
                    return Err(GraphError::InvalidInputShape);
                };

                if a.len() != b.len() {
                    return Err(GraphError::InvalidInputShape);
                }
                Ok(vec![a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()])
            }
            OperationType::Relu | OperationType::Max => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInputShape);
                }

                let result = inputs[0].iter().map(|&x| x.max(0.0)).collect();
                Ok(vec![result])
            }
            OperationType::Sigmoid => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInputShape);
                }

                let expected_size: usize = node.out_dims.iter().product();
                if inputs[0].len() != expected_size {
                    return Err(GraphError::InvalidInputShape);
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
                    return Err(GraphError::InvalidInputShape);
                }

                let expected_size: usize = node.out_dims.iter().product();
                let input = &inputs[0];

                if input.len() != expected_size {
                    return Err(GraphError::InvalidInputShape);
                }

                Ok(vec![input.clone()])
            }
            OperationType::Reshape => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidInputShape);
                }
                Ok(vec![inputs[0].clone()])
            }
        };

        // Log the tensor values after each operation
        // if let Ok(outputs) = &result {
        //     if let Err(e) = self.log_tensor_values(node.id, &node.op_type, outputs) {
        //         println!("Warning: Failed to log tensor values: {:?}", e);
        //     }
        // }

        result
    }

    /// Perform topological sort of nodes
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
    /// Weights
    pub weights: Option<Vec<f32>>,
    /// Bias
    pub bias: Option<Vec<f32>>,
    /// Attributes for the operations
    pub attributes: HashMap<String, Vec<usize>>,
}

impl From<&Node<TypedFact, Box<dyn TypedOp>>> for SerializableNode {
    fn from(node: &Node<TypedFact, Box<dyn TypedOp>>) -> Self {
        let op_name = node.op.name();
        let op_type = if op_name == "Const" {
            println!("Found Const operation");
            OperationType::Const
        } else if node.inputs.is_empty() {
            println!("Found Input operation");
            OperationType::Input
        } else if op_name.starts_with("Rm(") {
            println!("Found RmAxis operation");
            OperationType::RmAxis
        } else if let Some(op_type) = identify_tract_operation(node) {
            op_type
        } else {
            println!("Unknown operation: {}", op_name);
            OperationType::RmAxis // Default to RmAxis for unknown operations
        };

        println!("Node From : {:?}", node);
        println!("Node op_type: {:?}", op_name.as_ref());

        // Extract weights and biases based on node type
        let (weights, bias) = match op_name.as_ref() {
            "Const" => {
                if let Some(const_op) = node.op.downcast_ref::<Const>() {
                    if let Ok(tensor_data) = const_op.0.as_slice::<f32>() {
                        (Some(tensor_data.to_vec()), None)
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            _ => (None, None),
        };

        // Extract convolution attributes
        let mut attributes = HashMap::new();
        if let Some(conv_op) = node.op.downcast_ref::<Conv>() {
            let pool_spec = &conv_op.pool_spec;

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
            if let PaddingSpec::Explicit(before, after) = &pool_spec.padding {
                let mut pads = before.clone();
                pads.extend(after.iter().cloned());
                println!("pool_spec: {:?} ", pads);
                attributes.insert("pads".to_string(), pads.into_vec());
            }

            // Kernel format
            attributes.insert(
                "kernel_format".to_string(),
                vec![match conv_op.kernel_fmt {
                    KernelFormat::OIHW => 0,
                    KernelFormat::HWIO => 1,
                    KernelFormat::OHWI => 2,
                }],
            );
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
            weights,
            bias,
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
                    fact.shape
                        .set_dim(i, tract_onnx::prelude::TDim::Val(*batch_size as i64));
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
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let encoded: Vec<u8> =
            bincode::serialize(self).map_err(|_| GraphError::UnableToSaveModel)?;
        std::fs::write(path, encoded).map_err(|_| GraphError::UnableToSaveModel)
    }

    /// Loads a model from a binary file
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
                            tract_onnx::tract_hir::ops::scan::InputMapping::Scan(info) => {
                                input_mappings.push(InputMapping::Stacked {
                                    axis: info.axis,
                                    chunk: info.chunk as usize,
                                });
                            }
                            tract_onnx::tract_hir::ops::scan::InputMapping::State => {
                                input_mappings.push(InputMapping::State);
                            }
                            tract_onnx::tract_hir::ops::scan::InputMapping::Full => {
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

                    // Create the node with proper operation type and weights/biases
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
