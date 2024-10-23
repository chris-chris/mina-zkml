use super::errors::GraphError;
use instant;
use log::debug;
use serde::{Deserialize, Serialize};
use tract_data::internal::tract_smallvec::SmallVec;
use tract_itertools::Itertools;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::ops::scan::Scan;

pub type Outlet = (usize, usize);
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);

#[derive(Clone, Debug)]
pub struct Model {
    pub graph: ParsedNodes,
    pub visibility: VarVisibility,
}

#[derive(Clone, Debug)]
pub struct ParsedNodes {
    nodes: BTreeMap<usize, NodeType>,
    inputs: Vec<usize>,
    outputs: Vec<Outlet>,
}

impl ParsedNodes {
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn get_output_scales(&self) -> Result<Vec<i32>, GraphError> {
        self.outputs
            .iter()
            .map(|&(node, slot)| {
                self.nodes
                    .get(&node)
                    .ok_or(GraphError::MissingNode(node))
                    .map(|n| n.out_scales()[slot])
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub enum NodeType {
    Node(Node),
    SubGraph {
        model: Box<Model>,
        inputs: Vec<tract_onnx::prelude::OutletId>,
        idx: usize,
        out_dims: Vec<Vec<usize>>,
        out_scales: Vec<i32>,
        output_mappings: Vec<Vec<OutputMapping>>,
        input_mappings: Vec<InputMapping>,
    },
}

impl NodeType {
    pub fn out_scales(&self) -> &[i32] {
        match self {
            NodeType::Node(node) => std::slice::from_ref(&node.out_scale),
            NodeType::SubGraph { out_scales, .. } => out_scales,
        }
    }

    pub fn inputs(&self) -> Vec<Outlet> {
        match self {
            NodeType::Node(node) => node.inputs.clone(),
            NodeType::SubGraph { inputs, .. } => inputs.iter().map(|i| (i.node, i.slot)).collect(),
        }
    }

    pub fn out_dims(&self) -> Vec<Vec<usize>> {
        match self {
            NodeType::Node(node) => vec![node.out_dims.clone()],
            NodeType::SubGraph { out_dims, .. } => out_dims.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    op: Box<dyn TypedOp>,
    inputs: Vec<Outlet>,
    out_dims: Vec<usize>,
    out_scale: i32,
    id: usize,
}

#[derive(Clone, Debug)]
pub struct RunArgs {
    pub variables: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
pub struct VarVisibility {
    pub input: Visibility,
    pub output: Visibility,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputMapping {
    Full,
    State,
    Stacked { axis: usize, chunk: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OutputMapping {
    Single {
        outlet: usize,
        is_state: bool,
    },
    Stacked {
        outlet: usize,
        axis: usize,
        is_state: bool,
    },
}

impl OutputMapping {
    pub fn is_state(&self) -> bool {
        match self {
            OutputMapping::Single { is_state, .. } => *is_state,
            OutputMapping::Stacked { is_state, .. } => *is_state,
        }
    }

    pub fn outlet(&self) -> usize {
        match self {
            OutputMapping::Single { outlet, .. } => *outlet,
            OutputMapping::Stacked { outlet, .. } => *outlet,
        }
    }
}

fn input_state_idx(input_mappings: &[InputMapping]) -> Vec<usize> {
    input_mappings
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r, InputMapping::State))
        .map(|(index, _)| index)
        .collect()
}

fn output_state_idx(output_mappings: &[Vec<OutputMapping>]) -> Vec<usize> {
    output_mappings
        .iter()
        .flatten()
        .filter_map(|x| if x.is_state() { Some(x.outlet()) } else { None })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Visibility {
    Public,
    Private,
    Fixed,
}

impl Model {
    pub fn load_onnx_using_tract<P: AsRef<Path>>(
        path: P,
        run_args: &RunArgs,
    ) -> Result<TractResult, GraphError> {
        debug!("Starting load_onnx_using_tract");
        use tract_onnx::tract_hir::internal::GenericFactoid;

        let mut reader = std::fs::File::open(path).map_err(|_| GraphError::UnableToReadModel)?;

        let mut model = match tract_onnx::onnx().model_for_read(&mut reader) {
            Ok(model) => model,
            Err(_) => return Err(GraphError::UnableToReadModel),
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

            model
                .set_input_fact(i, fact)
                .map_err(|_| GraphError::UnableToReadModel)?;
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            match model.set_output_fact(i, InferenceFact::default()) {
                Ok(_) => (),
                Err(_) => return Err(GraphError::UnableToReadModel),
            }
        }

        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbol_table.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
            debug!("set {} to {}", symbol, value);
        }

        // Note: do not optimize the model, as the layout will depend on underlying hardware
        let typed_model = model
            .into_typed()
            .map_err(|_| GraphError::UnableToReadModel)?
            .concretize_dims(&symbol_values)
            .map_err(|_| GraphError::UnableToReadModel)?
            .into_decluttered()
            .map_err(|_| GraphError::UnableToReadModel)?;

        debug!("Completed load_onnx_using_tract successfully");
        Ok((typed_model, symbol_values))
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
        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };
        Ok(parsed_nodes)
    }

    pub fn new(path: &str, run_args: &RunArgs, visibility: &VarVisibility) -> Result<Self, GraphError> {
        let parsed_nodes = Self::load_onnx_model(path, &run_args, &visibility)?;
        Ok(Model {
            graph: parsed_nodes,
            visibility: visibility.clone(),
        })
    }

    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        visibility: VarVisibility,
        symbol_values: SymbolValues,
    ) -> Result<BTreeMap<usize, NodeType>, GraphError> {
        println!("Loading nodes from graph...");
        use super::utilities::node_output_shapes;
        debug!("Starting nodes_from_graph");

        let mut nodes = BTreeMap::<usize, NodeType>::new();
        println!("Loading nodes from graph now...");
        let mut input_idx = 0;
        for (i, n) in graph.nodes.iter().enumerate() {
            match n.op().downcast_ref::<Scan>() {
                Some(b) => {
                    let model = b.body.clone();
                    println!("Subgraph: {:?}", model);
                    let input_scales = n
                        .inputs
                        .iter()
                        .map(|i| {
                            Ok(nodes
                                .get(&i.node)
                                .ok_or(GraphError::MissingNode(i.node))?
                                .out_scales()[0])
                        })
                        .collect::<Result<Vec<_>, GraphError>>()?;
                    println!("Input scales: {:?}", input_scales);
                    let mut input_mappings = vec![];
                    for mapping in &b.input_mapping {
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
                    let input_state_idx = input_state_idx(&input_mappings);
                    println!("Input mappings: {:?}", input_mappings);
                    let mut output_mappings = vec![];
                    for (i, mapping) in b.output_mapping.iter().enumerate() {
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
                    let output_state_idx = output_state_idx(&output_mappings);

                    let mut output_scale_override: HashMap<usize, i32> = HashMap::new();

                    // if input_state_idx and output_state_idx have mismatched scales we need to rebase the scale of the output node
                    for (input_idx, output_idx) in input_state_idx.iter().zip(output_state_idx) {
                        let input_scale = input_scales[*input_idx];
                        // output mappings is a vec of vec. we need to find the outer index of the output node we want to rebase.
                        let mut traversed_len = 0;
                        for (outer_idx, mappings) in output_mappings.iter().enumerate() {
                            let mapping_len = mappings.len();
                            if traversed_len + mapping_len > output_idx {
                                let output_node_idx = b.body.outputs[outer_idx].node;
                                output_scale_override.insert(output_node_idx, input_scale);
                            }
                            traversed_len += mapping_len;
                        }
                    }

                    let subgraph_nodes =
                        Self::nodes_from_graph(&model, visibility.clone(), symbol_values.clone())?;

                    let subgraph = ParsedNodes {
                        nodes: subgraph_nodes,
                        inputs: model.inputs.iter().map(|o| o.node).collect(),
                        outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
                    };

                    let om = Model {
                        graph: subgraph,
                        visibility: visibility.clone(),
                    };
                    let out_dims = node_output_shapes(n, &symbol_values)?;

                    let mut output_scales = BTreeMap::new();

                    for (i, _mapping) in b.output_mapping.iter().enumerate() {
                        for mapping in b.output_mapping.iter() {
                            if let Some(outlet) = mapping.last_value_slot {
                                output_scales.insert(outlet, om.graph.get_output_scales()?[i]);
                            }
                            if let Some(last) = mapping.scan {
                                output_scales.insert(last.0, om.graph.get_output_scales()?[i]);
                            }
                        }
                    }

                    let out_scales = output_scales.into_values().collect_vec();

                    nodes.insert(
                        i,
                        NodeType::SubGraph {
                            model: Box::new(om),
                            inputs: n.inputs.iter().map(|i| OutletId::new(i.node, i.slot)).collect_vec(),
                            idx: i,
                            output_mappings,
                            input_mappings,
                            out_dims,
                            out_scales,
                        },
                    );
                }
                None => {
                    let node = Node {
                        op: n.op.clone(),
                        inputs: n.inputs.iter().map(|i| (i.node, i.slot)).collect(),
                        out_dims: node_output_shapes(n, &symbol_values)?
                            .pop()
                            .unwrap_or_default(),
                        out_scale: 1, // Default scale
                        id: i,
                    };
                    nodes.insert(i, NodeType::Node(node));
                }
            }
        }
        Ok(nodes)
    }

    pub fn run_prediction(&self, inputs: SmallVec<[TValue; 4]>) -> Result<Vec<Tensor>, GraphError> {
        if inputs.len() != self.graph.inputs.len() {
            return Err(GraphError::InvalidInputShape);
        }

        let mut intermediate_results: BTreeMap<usize, Vec<Tensor>> = BTreeMap::new();
        println!("Input shapes {:?}", inputs.to_vec());
        // Initialize with input tensors
        for (idx, input) in self.graph.inputs.iter().zip(inputs.into_iter()) {
            intermediate_results.insert(*idx, vec![input.into_tensor()]);
        }
        println!("Length of intermediate results {:?}", intermediate_results.len());
        // Traverse nodes in topological order
        for (idx, node) in self.graph.nodes.iter() {
            match node {
                NodeType::Node(n) => {
                    let input_tensors: Vec<Tensor> = n
                        .inputs
                        .iter()
                        .map(|&(node, slot)| {
                            intermediate_results
                                .get(&node)
                                .and_then(|tensors| tensors.get(slot))
                                .cloned()
                                .ok_or(GraphError::MissingNode(node))
                        })
                        .collect::<Result<Vec<Tensor>, GraphError>>()?;
                    println!("Executing operation {:?}", n.op);
                    println!("Input tensors {:?}", input_tensors);
                    println!("Input shapes {:?}", input_tensors.iter().map(|t| t.shape()).collect::<Vec<_>>());
                    let outputs = self.execute_operation(&n.op, input_tensors)?;
                    intermediate_results.insert(*idx, outputs);
                }
                NodeType::SubGraph {
                    model,
                    inputs,
                    input_mappings,
                    output_mappings,
                    ..
                } => {
                    let input_tensors: Vec<Tensor> = inputs
                        .iter()
                        .map(|input| {
                            intermediate_results
                                .get(&input.node)
                                .and_then(|tensors| tensors.get(input.slot))
                                .cloned()
                                .ok_or(GraphError::MissingNode(input.node))
                        })
                        .collect::<Result<Vec<Tensor>, GraphError>>()?;

                    let mut subgraph_inputs = Vec::new();
                    for (tensor, mapping) in input_tensors.iter().zip(input_mappings.iter()) {
                        match mapping {
                            InputMapping::Full => subgraph_inputs.push(tensor.clone()),
                            InputMapping::State => subgraph_inputs.push(tensor.clone()),
                            InputMapping::Stacked { axis, chunk } => {
                                let mut sliced_tensors = Vec::new();
                                let dim_size = tensor.shape()[*axis];
                                for start in (0..dim_size).step_by(*chunk) {
                                    let end = std::cmp::min(start + chunk, dim_size);
                                    let slice = tensor.slice(*axis, start, end).unwrap();
                                    sliced_tensors.push(slice);
                                }
                                subgraph_inputs.extend(sliced_tensors);
                            }
                        }
                    }

                    let subgraph_inputs: SmallVec<[TValue; 4]> = subgraph_inputs.into_iter().map(|t| t.into_tvalue()).collect();
                    let subgraph_outputs = model.run_prediction(subgraph_inputs)?;

                    let mut outputs = Vec::new();
                    for mapping in output_mappings.iter() {
                        for m in mapping {
                            match m {
                                OutputMapping::Single { outlet, .. } => {
                                    outputs.push(subgraph_outputs[*outlet].clone());
                                }
                                OutputMapping::Stacked { outlet, axis, .. } => {
                                    let stacked =
                                        Tensor::stack_tensors(*axis, &subgraph_outputs[*outlet..])
                                            .unwrap();
                                    outputs.push(stacked);
                                }
                            }
                        }
                    }

                    intermediate_results.insert(*idx, outputs);
                }
            }
        }

        // Collect output tensors
        let outputs: Result<Vec<Tensor>, GraphError> = self
            .graph
            .outputs
            .iter()
            .map(|&(node, slot)| {
                intermediate_results
                    .get(&node)
                    .and_then(|tensors| tensors.get(slot))
                    .cloned()
                    .ok_or(GraphError::MissingNode(node))
            })
            .collect();

        outputs
    }

    fn execute_operation(
        &self,
        op: &Box<dyn TypedOp>,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, GraphError> {
        let mut model = TypedModel::default();
        let mut node_inputs = tvec!();
        for (idx, input) in inputs.iter().enumerate() {
            let fact = tensor_to_fact(input);
            let input = model.add_source(format!("input_{}", idx), fact).unwrap();
            node_inputs.push(input);
        }
        let node = model
            .wire_node("operation", op.clone(), &node_inputs)
            .unwrap();

        for (idx, &output) in node.iter().enumerate() {
            model.set_output_fact(idx, model.outlet_fact(output).unwrap().clone());
        }

        let plan = SimplePlan::new(model).unwrap();

        let input_values: TVec<TValue> = inputs.into_iter().map(|t| t.into_tvalue()).collect();

        let outputs = plan.run(input_values).unwrap();

        Ok(outputs.iter().map(|v| v.clone().into_tensor()).collect())
    }
}

fn tensor_to_fact(tensor: &Tensor) -> TypedFact {
    let shape: Vec<usize> = tensor.shape().into();
    let datum_type = tensor.datum_type();
    TypedFact::dt_shape(datum_type, shape)
}
