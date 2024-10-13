use tract_onnx::prelude::*;

fn main() {
    let model_path = "/Users/sshivaditya/PROJECTS/onnx-parser/models/resnet101-v1-7.onnx";
    let model = load_model(model_path.to_string()).unwrap();
    print_model_structure(model);
}


fn load_model(
    path: String,
) -> Result<
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    Box<dyn std::error::Error>,
> {
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, f32::fact([1, 3, 224, 224]).into())?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn print_model_structure(
    graph: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
) {
    println!("Model Structure:");
    println!("================");

    let nodes = graph.model().nodes();
    let total_nodes = nodes.len();

    for (node_id, node) in nodes.iter().enumerate() {
        let op_name = node.op().name();
        let inputs = node.inputs.len();
        let outputs = node.outputs.len();

        println!("Node {}/{}: {} (inputs: {}, outputs: {})", node_id + 1, total_nodes, op_name, inputs, outputs);

        for (i, input) in node.inputs.iter().enumerate() {
            if let Ok(fact) = graph.model().outlet_fact(*input) {
                println!("  Input  {}: Shape: {:?}, Type: {:?}", i, fact.shape, fact.datum_type);
            } else {
                println!("  Input  {}: Unknown", i);
            }
        }
        println!();
    }

    println!("Total nodes: {}", total_nodes);
}

