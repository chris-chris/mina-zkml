use super::errors::GraphError;
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
