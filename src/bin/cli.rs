use anyhow::Result;
use clap::{Parser, Subcommand};
use mina_zkml::graph::model::Model;
use prettytable::{row, Table};
use serde_json;
use std::fs;
use std::path::PathBuf;
use mina_zkml::graph::model::RunArgs;
use std::collections::HashMap;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a table from ONNX model operators
    Table {
        /// Path to the ONNX model file
        #[arg(short, long)]
        model: PathBuf,
    },
    /// Convert ONNX model to graph JSON
    Convert {
        /// Path to the ONNX model file
        #[arg(short, long)]
        model: PathBuf,
        /// Output JSON file path
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Generate proof from model
    Proof {
        /// Path to the ONNX model file
        #[arg(short, long)]
        model: PathBuf,
        /// Output proof file path
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Verify a proof
    Verify {
        /// Path to the proof file
        #[arg(short, long)]
        proof: PathBuf,
        /// Path to the ONNX model file for verification
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Table { model } => {
            let mut table = Table::new();
            table.add_row(row!["Operator", "Input Shape", "Output Shape"]);
            
            // Create default RunArgs
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };

            // Load the model
            let model = Model::new(
                model.to_str().unwrap(),
                &run_args,
                &mina_zkml::graph::model::VarVisibility {
                    input: mina_zkml::graph::model::Visibility::Public,
                    output: mina_zkml::graph::model::Visibility::Public,
                },
            )?;

            // Add rows for each node
            for (_, node_type) in &model.graph.nodes {
                if let mina_zkml::graph::model::NodeType::Node(node) = node_type {
                    let op_type = format!("{:?}", node.op_type);
                    
                    // Get input shapes from connected nodes
                    let input_shapes: Vec<String> = node.inputs
                        .iter()
                        .filter_map(|(node_id, _)| {
                            if let Some(mina_zkml::graph::model::NodeType::Node(input_node)) = model.graph.nodes.get(node_id) {
                                Some(format!("{:?}", input_node.out_dims))
                            } else {
                                None
                            }
                        })
                        .collect();

                    let input_shape_str = if input_shapes.is_empty() {
                        "[]".to_string()
                    } else {
                        input_shapes.join(", ")
                    };

                    table.add_row(row![
                        op_type,
                        input_shape_str,
                        format!("{:?}", node.out_dims)
                    ]);
                }
            }
            
            table.printstd();
        }
        Commands::Convert { model, output } => {
            // Create default RunArgs
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };

            // Load and convert the model
            let model = Model::new(
                model.to_str().unwrap(),
                &run_args,
                &mina_zkml::graph::model::VarVisibility {
                    input: mina_zkml::graph::model::Visibility::Public,
                    output: mina_zkml::graph::model::Visibility::Public,
                },
            )?;

            // Serialize to JSON
            let json = serde_json::to_string_pretty(&model)?;
            fs::write(output, json)?;
            println!("Model converted and saved to {:?}", output);
        }
        Commands::Proof { model, output } => {
            // Create default RunArgs
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };

            // Load the model
            let model = Model::new(
                model.to_str().unwrap(),
                &run_args,
                &mina_zkml::graph::model::VarVisibility {
                    input: mina_zkml::graph::model::Visibility::Public,
                    output: mina_zkml::graph::model::Visibility::Public,
                },
            )?;

            // Generate proof (this is a placeholder - actual proof generation needs to be implemented)
            let proof = serde_json::json!({
                "model_hash": format!("{:x}", md5::compute(serde_json::to_string(&model)?)),
                "timestamp": chrono::Utc::now().to_rfc3339(),
                // Add more proof-specific fields here
            });

            // Save the proof
            fs::write(output, serde_json::to_string_pretty(&proof)?)?;
            println!("Proof generated and saved to {:?}", output);
        }
        Commands::Verify { proof, model } => {
            // Read the proof
            let proof_content = fs::read_to_string(proof)?;
            let proof_data: serde_json::Value = serde_json::from_str(&proof_content)?;

            // Create default RunArgs
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };

            // Load the model
            let model = Model::new(
                model.to_str().unwrap(),
                &run_args,
                &mina_zkml::graph::model::VarVisibility {
                    input: mina_zkml::graph::model::Visibility::Public,
                    output: mina_zkml::graph::model::Visibility::Public,
                },
            )?;

            // Verify the proof (this is a placeholder - actual verification needs to be implemented)
            let model_hash = format!("{:x}", md5::compute(serde_json::to_string(&model)?));
            if model_hash == proof_data["model_hash"].as_str().unwrap() {
                println!("Proof verification successful!");
            } else {
                println!("Proof verification failed!");
            }
        }
    }

    Ok(())
}
