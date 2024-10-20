use super::errors::GraphError;
use tract_onnx::prelude::{DatumType, Node as OnnxNode, SymbolValues, TypedFact, TypedOp};



pub fn node_output_shapes(
    node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
    symbol_values: &SymbolValues,
) -> Result<Vec<Vec<usize>>, GraphError> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
    for output in outputs {
        let shape = output.fact.shape;
        let shape = shape.eval_to_usize(symbol_values);
        match shape {
            Ok(s) => shapes.push(s.into_owned().into_iter().collect()),
            Err(_) => return Err(GraphError::UnableToReadModel),
        }
    }
    Ok(shapes)
}