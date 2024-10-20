use super::errors::GraphError;
use instant;
use log::debug;
use serde::{Deserialize, Serialize};
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
    /// Returns the number of the computational graph's inputs
    pub fn num_inputs(&self) -> usize {
        let input_nodes = self.inputs.iter();
        input_nodes.len()
    }

    /// Input types
    pub fn get_input_types(&self) -> Result<Vec<InputType>, GraphError> {
        self.inputs
            .iter()
            .map(|o| {
                match self
                    .nodes
                    .get(o)
                    .ok_or(GraphError::MissingNode(*o))?
                    .opkind()
                {
                    SupportedOp::Input(Input { datum_type, .. }) => Ok(datum_type.clone()),
                    _ => Err(GraphError::InvalidInputTypes),
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[derive(Clone, Debug)]
pub enum NodeType {
    Node(Node),
    SubGraph {
        model: Box<Model>,
        inputs: Vec<OutletId>,
        idx: usize,
        out_dims: Vec<usize>,
        out_scales: Vec<i32>,
    },
}

impl NodeType {
    pub fn out_scales(&self) -> &[i32] {
        match self {
            NodeType::Node(node) => std::slice::from_ref(&node.out_scale),
            NodeType::SubGraph { out_scales, .. } => out_scales,
        }
    }

    /// Returns the operation kind of the node (if any).
    pub fn opkind(&self) -> SupportedOp {
        match self {
            NodeType::Node(n) => n.opkind.clone(),
            NodeType::SubGraph { .. } => SupportedOp::Unknown(Unknown),
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
    ///
    Full,
    ///
    State,
    ///
    Stacked {
        ///
        axis: usize,
        ///
        chunk: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OutputMapping {
    ///
    Single {
        ///
        outlet: usize,
        ///
        is_state: bool,
    },
    ///
    Stacked {
        ///
        outlet: usize,
        ///
        axis: usize,
        ///
        is_state: bool,
    },
}

impl OutputMapping {
    ///
    pub fn is_state(&self) -> bool {
        match self {
            OutputMapping::Single { is_state, .. } => *is_state,
            OutputMapping::Stacked { is_state, .. } => *is_state,
        }
    }

    ///
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
        .collect::<Vec<_>>()
}

fn output_state_idx(output_mappings: &[Vec<OutputMapping>]) -> Vec<usize> {
    output_mappings
        .iter()
        .flatten()
        .filter_map(|x| if x.is_state() { Some(x.outlet()) } else { None })
        .collect::<Vec<_>>()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Visibility {
    Public,
    Private,
    Fixed,
}

impl Model {
    pub fn load_onnx_using_tract<P: AsRef<Path>>(path: P) -> Result<TractResult, GraphError> {
        use tract_onnx::tract_hir::internal::GenericFactoid;

        let mut reader = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(_) => return Err(GraphError::UnableToReadModel),
        };
        let mut model = match tract_onnx::onnx().model_for_read(&mut reader) {
            Ok(model) => model,
            Err(_) => return Err(GraphError::UnableToReadModel),
        };

        //TOOD: Variable HashMap from Run Args
        let variables: HashMap<String, usize> = HashMap::new();

        //TODO: Impment Run args right now its dummy
        let run_args = RunArgs {
            variables: HashMap::new(),
        };

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
            model.set_input_fact(i, fact);
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default());
        }

        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbols.sym(symbol);
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

        Ok((typed_model, symbol_values))
    }

    // pub fn load_onnx_model<P: AsRef<Path>>(
    //     path: P,
    //     visibility: VarVisibility,
    // ) -> Result<ParsedNodes, GraphError> {
    //     let start_time = instant::Instant::now();

    //     let (graph, symbol_values) = Model::load_onnx_using_tract(path)?;

    //     //TODO: Implement RunArgs
    //     let run_args = RunArgs {
    //         variables: HashMap::from([
    //             ("batch_size".to_string(), 1),
    //             ("sequence_length".to_string(), 128),
    //             ("feature_size".to_string(), 64),
    //             ("input_scale".to_string(), 1),
    //             ("param_scale".to_string(), 1),
    //             ("scale_rebase_multipler".to_string(), 1),
    //         ]),
    //     };

    //     //TODO: Implment Scale
    //     let scales: VarScale = VarScale {
    //         input: 1,
    //         output: 1,
    //         param: 1,
    //     };
    // }

    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        visibility: VarVisibility,
        symbol_values: SymbolValues,
    ) -> Result<BTreeMap<usize, NodeType>, GraphError> {
        use super::utilities::node_output_shapes;

        let mut nodes = BTreeMap::<usize, NodeType>::new();
        let mut input_idx = 0;

        for (i, n) in graph.nodes.iter().enumerate() {
            // Extract the slope layer hyperparams
            match n.op().downcast_ref::<Scan>() {
                Some(b) => {
                    let model = b.body.clone();
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

                    let mut output_scale_override = HashMap::new();
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

                    let subgraph_nodes = Self::nodes_from_graph(
                        &model,
                        visibility,
                        symbol_values,
                        // Some(input_scales.clone()),
                        // Some(output_scale_override),
                    )?;

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
                            model: om,
                            inputs: n.inputs.iter().map(|i| (i.node, i.slot)).collect_vec(),
                            idx: i,
                            output_mappings,
                            input_mappings,
                            out_dims,
                            out_scales,
                        },
                    );
                }
                None => {
                    let mut n =
                        Node::new(n.clone(), &mut nodes, scales, i, symbol_values, run_args)?;
                    if let Some(ref scales) = override_input_scales {
                        if let Some(inp) = n.opkind.get_input() {
                            let scale = scales[input_idx];
                            n.opkind = SupportedOp::Input(Input {
                                scale,
                                datum_type: inp.datum_type,
                            });
                            input_idx += 1;
                            n.out_scale = scale;
                        }
                    }
                    if let Some(ref scales) = override_output_scales {
                        if scales.contains_key(&i) {
                            let scale_diff = n.out_scale - scales[&i];
                            n.opkind = if scale_diff > 0 {
                                RebaseScale::rebase(
                                    n.opkind,
                                    scales[&i],
                                    n.out_scale,
                                    1,
                                    run_args.div_rebasing,
                                )
                            } else {
                                RebaseScale::rebase_up(
                                    n.opkind,
                                    scales[&i],
                                    n.out_scale,
                                    run_args.div_rebasing,
                                )
                            };
                            n.out_scale = scales[&i];
                        }
                    }
                    nodes.insert(i, NodeType::Node(n));
                }
            }
        }
        //Self::remove_unused_nodes(&mut nodes);
        Ok(nodes)
    }
}
