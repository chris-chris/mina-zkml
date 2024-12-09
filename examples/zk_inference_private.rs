use mina_zkml::{
    graph::{model::{Model, RunArgs, VarVisibility, Visibility}, errors::GraphError},
};
use std::collections::HashMap;

fn main() -> Result<(), GraphError> {
    // Create run arguments with batch size
    let mut variables = HashMap::new();
    variables.insert("batch_size".to_string(), 1);

    let run_args = RunArgs {
        variables,
    };

    // Set all variables as private for ZK inference
    let visibility = VarVisibility {
        input: Visibility::Private,
        output: Visibility::Private,
    };

    // Load the model with private visibility settings
    let model = Model::new("models/simple_perceptron.onnx", &run_args, &visibility)?;

    // Example input data (all private)
    let inputs = vec![vec![0.5, 0.3, 0.2]];

    // Execute model with private inputs and outputs
    let result = model.graph.execute(&inputs)?;
    println!("Private inference result: {:?}", result);

    Ok(())
}
