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
        println!("Symbol Values: {:?}", symbol_values);
        println!("Shape: {:?}", shape.to_vec());
        println!("Shape: {:?}", shape.eval_to_usize(symbol_values));
        let shape = shape.eval_to_usize(symbol_values).unwrap();
        let mv = shape.to_vec();
        shapes.push(mv)
    }
    Ok(shapes)
}