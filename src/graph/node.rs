use std::fmt;
use tract_onnx::prelude::*;
use super::{model::{NodeType, RunArgs}, scales::{Scale, VarScales}};

pub type Outlet = (usize, usize);

#[derive(Clone, Debug)]
pub struct Node {
    pub opkind: SupportedOp,
    pub inputs: Vec<Outlet>,
    pub out_dims: Vec<usize>,
    pub out_scale: Scale,
    pub num_uses: usize,
    pub idx: usize,
}

impl Node {
    pub fn new(
        node: &tract_onnx::prelude::Node,
        nodes: &mut std::collections::BTreeMap<usize, NodeType>,
        scales: &VarScales,
        idx: usize,
        symbol_values: &SymbolValues,
        run_args: &RunArgs,
    ) -> Result<Self, ModelError> {
        let op = SupportedOp::from_tract_op(&node.op, run_args)?;
        let inputs = node.inputs.iter().map(|i| (i.node, i.slot)).collect();
        let out_dims = node_output_shapes(node, symbol_values)?;
        let out_scale = scales.output;
        let num_uses = nodes.values().filter(|n| n.inputs().contains(&(idx, 0))).count();

        Ok(Node {
            opkind: op,
            inputs,
            out_dims,
            out_scale,
            num_uses,
            idx,
        })
    }
}

#[derive(Clone, Debug)]
pub enum SupportedOp {
    Input(Input),
    Constant(Constant),
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Gemm { alpha: f32, beta: f32, transpose_a: bool, transpose_b: bool },
    Conv { kernel_shape: Vec<usize>, strides: Vec<usize>, pads: Vec<usize> },
    MaxPool { kernel_shape: Vec<usize>, strides: Vec<usize>, pads: Vec<usize> },
    AveragePool { kernel_shape: Vec<usize>, strides: Vec<usize>, pads: Vec<usize> },
    GlobalAveragePool,
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu { alpha: f32 },
    Softmax { axis: i64 },
    Reshape { shape: Vec<i64> },
    Transpose { perm: Vec<usize> },
    Concat { axis: i64 },
    Slice { starts: Vec<i64>, ends: Vec<i64>, axes: Vec<i64> },
    Squeeze { axes: Vec<i64> },
    Unsqueeze { axes: Vec<i64> },
    Expand,
    Flatten { axis: i64 },
    Gather { axis: i64 },
    Clip { min: Option<f32>, max: Option<f32> },
    Pad { mode: PadMode, pads: Vec<i64>, value: f32 },
    ReduceMean { axes: Vec<i64>, keepdims: bool },
    ReduceSum { axes: Vec<i64>, keepdims: bool },
    ReduceMax { axes: Vec<i64>, keepdims: bool },
    ReduceMin { axes: Vec<i64>, keepdims: bool },
    Unknown,
}

#[derive(Clone, Debug)]
pub enum PadMode {
    Constant,
    Reflect,
    Edge,
}

#[derive(Clone, Debug)]
pub struct Input {
    pub scale: Scale,
    pub datum_type: DatumType,
}

#[derive(Clone, Debug)]
pub struct Constant {
    pub quantized_values: Tensor,
    pub raw_values: Option<Tensor>,
}

impl SupportedOp {
    pub fn from_tract_op(op: &dyn TypedOp, run_args: &RunArgs) -> Result<Self, ModelError> {
        match op.downcast_ref::<tract_onnx::prelude::ops::konst::Const>() {
            Some(c) => Ok(SupportedOp::Constant(Constant {
                quantized_values: c.0.clone(),
                raw_values: None,
            })),
            None => {
                match op.name() {
                    "Add" => Ok(SupportedOp::Add),
                    "Sub" => Ok(SupportedOp::Sub),
                    "Mul" => Ok(SupportedOp::Mul),
                    "Div" => Ok(SupportedOp::Div),
                    "MatMul" => Ok(SupportedOp::MatMul),
                    "Gemm" => {
                        if let Some(gemm) = op.downcast_ref::<tract_onnx::ops::matmul::Gemm>() {
                            Ok(SupportedOp::Gemm {
                                alpha: gemm.alpha,
                                beta: gemm.beta,
                                transpose_a: gemm.trans_a,
                                transpose_b: gemm.trans_b,
                            })
                        } else {
                            Err(ModelError::UnsupportedOperation("Gemm".to_string()))
                        }
                    },
                    "Conv" => {
                        if let Some(conv) = op.downcast_ref::<tract_onnx::ops::cnn::Conv>() {
                            Ok(SupportedOp::Conv {
                                kernel_shape: conv.kernel_shape.clone(),
                                strides: conv.strides.clone(),
                                pads: conv.padding.iter().cloned().flatten().collect(),
                            })
                        } else {
                            Err(ModelError::UnsupportedOperation("Conv".to_string()))
                        }
                    },
                    "MaxPool" => {
                        if let Some(pool) = op.downcast_ref::<tract_onnx::ops::cnn::MaxPool>() {
                            Ok(SupportedOp::MaxPool {
                                kernel_shape: pool.kernel_shape.clone(),
                                strides: pool.strides.clone(),
                                pads: pool.padding.iter().cloned().flatten().collect(),
                            })
                        } else {
                            Err(ModelError::UnsupportedOperation("MaxPool".to_string()))
                        }
                    },
                    "AveragePool" => {
                        if let Some(pool) = op.downcast_ref::<tract_onnx::ops::cnn::AvgPool>() {
                            Ok(SupportedOp::AveragePool {
                                kernel_shape: pool.kernel_shape.clone(),
                                strides: pool.strides.clone(),
                                pads: pool.padding.iter().cloned().flatten().collect(),
                            })
                        } else {
                            Err(ModelError::UnsupportedOperation("AveragePool".to_string()))
                        }
                    },
                    "GlobalAveragePool" => Ok(SupportedOp::GlobalAveragePool),
                    "Relu" => Ok(SupportedOp::Relu),
                    "Sigmoid" => Ok(SupportedOp::Sigmoid),
                    "Tanh" => Ok(SupportedOp::Tanh),
                    "LeakyRelu" => {
                        if let Some(leaky) = op.downcast_ref::<tract_onnx::ops::activations::LeakyRelu>() {
                            Ok(SupportedOp::LeakyRelu { alpha: leaky.alpha })
                        } else {
                            Err(ModelError::UnsupportedOperation("LeakyRelu".to_string()))
                        }
                    },
                    "Softmax" => {
                        if let Some(softmax) = op.downcast_ref::<tract_onnx::ops::nn::Softmax>() {
                            Ok(SupportedOp::Softmax { axis: softmax.axis })
                        } else {
                            Err(ModelError::UnsupportedOperation("Softmax".to_string()))
                        }
                    },
                    "Reshape" => {
                        if let Some(reshape) = op.downcast_ref::<tract_onnx::ops::array::Reshape>() {
                            Ok(SupportedOp::Reshape { shape: reshape.shape.clone() })
                        } else {
                            Err(ModelError::UnsupportedOperation("Reshape".to_string()))
                        }
                    },
                    "Transpose" => {
                        if let Some(transpose) = op.downcast_ref::<tract_onnx::ops::array::Transpose>() {
                            Ok(SupportedOp::Transpose { perm: transpose.perm.clone() })
                        } else {
                            Err(ModelError::UnsupportedOperation("Transpose".to_string()))
                        }
                    },
                    "Concat" => {
                        if let Some(concat) = op.downcast_ref::<tract_onnx::ops::array::Concat>() {
                            Ok(SupportedOp::Concat { axis: concat.axis })
                        } else {
                            Err(ModelError::UnsupportedOperation("Concat".to_string()))
                        }
                    },
                    // Add more operations as needed...
                    _ => Ok(SupportedOp::Unknown),
                }
            }
        }
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, SupportedOp::Constant(_))
    }

    pub fn is_input(&self) -> bool {
        matches!(self, SupportedOp::Input(_))
    }

    pub fn as_string(&self) -> String {
        match self {
            SupportedOp::Input(_) => "Input".to_string(),
            SupportedOp::Constant(_) => "Constant".to_string(),
            SupportedOp::Add => "Add".to_string(),
            SupportedOp::Sub => "Sub".to_string(),
            SupportedOp::Mul => "Mul".to_string(),
            SupportedOp::Div => "Div".to_string(),
            SupportedOp::MatMul => "MatMul".to_string(),
            SupportedOp::Gemm { .. } => "Gemm".to_string(),
            SupportedOp::Conv { .. } => "Conv".to_string(),
            SupportedOp::MaxPool { .. } => "MaxPool".to_string(),
            SupportedOp::AveragePool { .. } => "AveragePool".to_string(),
            SupportedOp::GlobalAveragePool => "GlobalAveragePool".to_string(),
            SupportedOp::Relu => "Relu".to_string(),
            SupportedOp::Sigmoid => "Sigmoid".to_string(),
            SupportedOp::Tanh => "Tanh".to_string(),
            SupportedOp::LeakyRelu { .. } => "LeakyRelu".to_string(),
            SupportedOp::Softmax { .. } => "Softmax".to_string(),
            SupportedOp::Reshape { .. } => "Reshape".to_string(),
            SupportedOp::Transpose { .. } => "Transpose".to_string(),
            SupportedOp::Concat { .. } => "Concat".to_string(),
            SupportedOp::Slice { .. } => "Slice".to_string(),
            SupportedOp::Squeeze { .. } => "Squeeze".to_string(),
            SupportedOp::Unsqueeze { .. } => "Unsqueeze".to_string(),
            SupportedOp::Expand => "Expand".to_string(),
            SupportedOp::Flatten { .. } => "Flatten".to_string(),
            SupportedOp::Gather { .. } => "Gather".to_string(),
            SupportedOp::Clip { .. } => "Clip".to_string(),
            SupportedOp::Pad { .. } => "Pad".to_string(),
            SupportedOp::ReduceMean { .. } => "ReduceMean".to_string(),
            SupportedOp::ReduceSum { .. } => "ReduceSum".to_string(),
            SupportedOp::ReduceMax { .. } => "ReduceMax".to_string(),
            SupportedOp::ReduceMin { .. } => "ReduceMin".to_string(),
            SupportedOp::Unknown => "Unknown".to_string(),
        }
    }
}

impl fmt::Display for SupportedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

// Helper function to get output shapes
fn node_output_shapes(node: &tract_onnx::prelude::Node, symbol_values: &SymbolValues) -> Result<Vec<usize>, ModelError> {
    node.outputs
        .iter()
        .map(|o| {
            o.fact
                .shape
                .eval_to_usize(symbol_values)
                .map_err(|e| ModelError::ShapeError(e.to_string()))
        })
        .collect()
}

#[derive(Debug)]
pub enum ModelError {
    UnsupportedOperation(String),
    ShapeError(String),
    // Add other error types as needed
}