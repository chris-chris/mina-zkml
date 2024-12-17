use mina_zkml::graph::model::{Model, RunArgs, VarVisibility, Visibility};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create run args with batch size
    let mut variables = HashMap::new();
    variables.insert("batch_size".to_string(), 1);
    let run_args = RunArgs { variables };

    // Create visibility settings
    let visibility = VarVisibility {
        input: Visibility::Public,
        output: Visibility::Public,
    };

    // Load the perceptron model
    println!("Loading perceptron model...");
    let model = Model::new("models/simple_perceptron.onnx", &run_args, &visibility)?;

    // Print model structure
    println!("\nModel structure:");
    println!("Number of nodes: {}", model.graph.nodes.len());
    println!("Input nodes: {:?}", model.graph.inputs);
    println!("Output nodes: {:?}", model.graph.outputs);

    // Print node connections
    println!("\nNode connections:");
    for (id, node) in &model.graph.nodes {
        match node {
            mina_zkml::graph::model::NodeType::Node(n) => {
                println!("Node {}: {:?} inputs: {:?}", id, n.op_type, n.inputs);
                println!("Output dimensions: {:?}", n.out_dims);
                println!("Weight Tensor: {:?}", n.op_params);
                // println!("Bias Tensor: {:?}", n.bias);
            }
            mina_zkml::graph::model::NodeType::SubGraph { .. } => {
                println!("Node {}: SubGraph", id);
            }
        }
    }

    // Create a sample input vector of size 10
    let input = vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.7, 0.1, -0.4, 0.9, 0.6];
    println!("\nInput vector (size 10):");
    println!("{:?}", input);

    // Execute the model
    let result = model.graph.execute(&[input])?;

    // Print the output
    println!("\nOutput vector (size 3, after ReLU):");
    println!("{:?}", result[0]);

    Ok(())
}
