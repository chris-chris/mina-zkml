#![allow(warnings)]

use std::collections::HashMap;
use kimchi::graph::model::{Model, RunArgs, VarVisibility, Visibility};
use log::debug;
use tract_onnx::prelude::*;
use image::{self, ImageBuffer, Rgb};
use ndarray::{Array4, CowArray};


fn main() {
    let model_path = "/Users/sshivaditya/PROJECTS/onnx-parser/models/resnet101-v1-7.onnx";
    //let model_native = load_model(model_path.to_string()).unwrap();
    // print_model_structure(model);
    let run_args = RunArgs {
        variables: HashMap::from([
            ("N".to_string(), 1),
            ("C".to_string(), 3),
            ("H".to_string(), 224),
            ("W".to_string(), 224),
            ("batch_size".to_string(), 1),
            ("sequence_length".to_string(), 128),
        ]),
    };

    let visibility = VarVisibility {
        input: Visibility::Private,
        output: Visibility::Public,
    };

    // let (model, symbol_values) = Model::load_onnx_using_tract(model_path, &run_args).unwrap();
    // let nodes = Model::nodes_from_graph(&model, visibility, symbol_values).unwrap();
    let model = Model::new(model_path, &run_args, &visibility).unwrap();

    // Load and preprocess the image
    let image_path = "/Users/sshivaditya/PROJECTS/mina-zkml/test data/dog.jpeg";
    let input_tensor = load_and_preprocess_image(image_path).unwrap();

    // Run inference on native tract model
    // println!("Running inference on native tract model...");
    // let native_output = model_native.run(tvec!(input_tensor.clone().into_tvalue())).unwrap();
    // let native_output_array = native_output[0].to_array_view::<f32>().unwrap().into_dimensionality::<ndarray::Ix2>().unwrap();
    // print_top_5(&native_output_array);

    // Run inference on custom Model
    // println!("\nRunning inference on custom Model...");
    // let custom_output = model.run_prediction(tvec!(input_tensor.clone().into_tvalue())).unwrap();
    // let custom_output_array = custom_output[0].to_array_view::<f32>().unwrap().into_dimensionality::<ndarray::Ix2>().unwrap();
    // print_top_5(&custom_output_array);
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

fn load_and_preprocess_image(path: &str) -> Result<Tensor, Box<dyn std::error::Error>> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(path)?.resize_exact(224, 224, image::imageops::FilterType::Triangle).to_rgb8();
    let mut array = Array4::zeros((1, 3, 224, 224));
    
    for (x, y, pixel) in img.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - 0.485) / 0.229;
        array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - 0.456) / 0.224;
        array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - 0.406) / 0.225;
    }

    Ok(Tensor::from(array.into_owned()))
}

fn print_top_5(output: &ndarray::ArrayView2<f32>) {
    let mut indices: Vec<usize> = (0..output.len()).collect();
    indices.sort_unstable_by(|&i, &j| output[[0, j]].partial_cmp(&output[[0, i]]).unwrap());

    println!("Top 5 predictions:");
    for &idx in indices.iter().take(5) {
        println!("Class {}: {:.2}%", idx, output[[0, idx]] * 100.0);
    }
}
