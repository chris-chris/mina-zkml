use mina_zkml::graph::model::{Model, RunArgs, VarVisibility, Visibility};
use std::collections::HashMap;

fn preprocess_image(img_path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Load and convert image to grayscale
    let img = image::open(img_path)?.into_luma8();

    // Ensure image is 28x28
    let resized = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);

    // Convert to f32 and normalize to [0, 1]
    let pixels: Vec<f32> = resized.into_raw().into_iter().map(|x| x as f32).collect();

    //Apply normalization
    let pixels: Vec<f32> = pixels
        .into_iter()
        .map(|x| (x / 255.0 - 0.1307) / 0.3081)
        .collect();

    // Create a batch dimension by wrapping the flattened pixels
    let mut input = Vec::with_capacity(1 * 28 * 28);
    input.extend_from_slice(&pixels);
    Ok(input)
}

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

    // Load the MNIST model
    println!("Loading MNIST model...");
    let model = Model::new("models/mnist_mlp.onnx", &run_args, &visibility).map_err(|e| {
        println!("Error loading model: {:?}", e);
        e
    })?;

    // Print model structure
    println!("\nModel structure:");
    println!("Number of nodes: {}", model.graph.nodes.len());
    println!("Input nodes: {:?}", model.graph.inputs);
    println!("Output nodes: {:?}", model.graph.outputs);

    // Load and preprocess the image
    println!("\nLoading and preprocessing image...");
    let input = preprocess_image("models/data/1052.png")?;

    // Execute the model
    println!("\nRunning inference...");
    let result = model.graph.execute(&[input])?;

    //Result
    println!("Result: {:?}", result);

    // Print the output probabilities
    println!("\nOutput probabilities for digits 0-9:");
    if let Some(probabilities) = result.get(0) {
        // The model outputs logits, so we need to apply softmax
        let max_logit = probabilities
            .iter()
            .take(10)
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = probabilities
            .iter()
            .take(10)
            .map(|&x| (x - max_logit).exp())
            .sum();

        let softmax: Vec<f32> = probabilities
            .iter()
            .take(10)
            .map(|&x| ((x - max_logit).exp()) / exp_sum)
            .collect();

        for (digit, &prob) in softmax.iter().enumerate() {
            println!("Digit {}: {:.4}", digit, prob);
        }

        // Find the predicted digit
        let predicted_digit = softmax
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(digit, _)| digit)
            .unwrap();

        println!("\nPredicted digit: {}", predicted_digit);
    }

    Ok(())
}
