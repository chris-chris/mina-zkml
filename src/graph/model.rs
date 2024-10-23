use super::errors::GraphError;
use instant;
use log::debug;
use serde::{Deserialize, Serialize};
use tract_onnx::tract_core::plan::SessionState;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;
use tract_data::internal::tract_smallvec::SmallVec;
use tract_itertools::Itertools;
use tract_onnx::{prelude::*, tract_core};
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

    pub fn new(
        path: &str,
        run_args: &RunArgs,
        visibility: &VarVisibility,
    ) -> Result<Self, GraphError> {
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
        use super::utilities::node_output_shapes;
        let mut nodes = BTreeMap::new();

        // First pass: Create all nodes
        for (idx, node) in graph.nodes.iter().enumerate() {
            match node.op().downcast_ref::<Scan>() {
                Some(scan_op) => {
                    println!("Processing scan node {}", idx);

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
                            inputs: node
                                .inputs
                                .iter()
                                .map(|i| OutletId::new(i.node, i.slot))
                                .collect(),
                            idx,
                            out_dims,
                            out_scales: vec![1; scan_op.output_mapping.len()],
                            output_mappings,
                            input_mappings,
                        },
                    );
                }
                None => {
                    println!("Processing regular node {}", idx);
                    // Create regular node
                    let out_dims = node_output_shapes(node, &symbol_values)?
                        .pop()
                        .unwrap_or_default();

                    nodes.insert(
                        idx,
                        NodeType::Node(Node {
                            op: node.op.clone(),
                            inputs: node.inputs.iter().map(|i| (i.node, i.slot)).collect(),
                            out_dims,
                            out_scale: 1,
                            id: idx,
                        }),
                    );
                }
            }
        }

        // Verify all required nodes exist
        let mut missing_nodes = Vec::new();

        // Check inputs
        for node in nodes.values() {
            match node {
                NodeType::Node(n) => {
                    for &(input_node, _) in &n.inputs {
                        if !nodes.contains_key(&input_node)
                            && !graph.inputs.iter().any(|x| x.node == input_node)
                        {
                            missing_nodes.push(input_node);
                        }
                    }
                }
                NodeType::SubGraph { inputs, .. } => {
                    for input in inputs {
                        if !nodes.contains_key(&input.node)
                            && !graph.inputs.iter().any(|x| x.node == input.node)
                        {
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
            println!("Missing nodes: {:?}", missing_nodes);
            return Err(GraphError::MissingNode(missing_nodes[0]));
        }

        Ok(nodes)
    }
}