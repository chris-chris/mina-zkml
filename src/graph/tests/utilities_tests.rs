#[cfg(test)]
mod tests {
    use super::super::utilities::*;
    use tract_onnx::prelude::*;

    #[test]
    fn test_node_output_shapes() {
        let mut model = TypedModel::default();
        
        // Create a simple input node
        let input_fact = TypedFact::dt_shape(DatumType::F32, tvec!(1, 3, 224, 224));
        let input = model.add_source("input", input_fact).unwrap();
        
        // Create symbol values
        let symbol_values = SymbolValues::default();
        
        // Get the node
        let node = model.nodes()[input.node].clone();
        
        // Test shape calculation
        let shapes = node_output_shapes(&node, &symbol_values).unwrap();
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0], vec![1, 3, 224, 224]);
    }
}
