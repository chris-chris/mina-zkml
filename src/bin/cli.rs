use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use mina_zkml::{
    graph::model::{Model, RunArgs, VarVisibility, Visibility},
    zk::proof::{ProverOutput, ProverSystem},
};
use prettytable::{row, Table};
use serde_json::{self, json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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
        /// Path to input data file (JSON array of arrays)
        #[arg(short, long)]
        input: PathBuf,
        /// Output proof file path
        #[arg(short, long)]
        output: PathBuf,
        /// Input visibility (public/private)
        #[arg(long, default_value = "public")]
        input_visibility: String,
        /// Output visibility (public/private)
        #[arg(long, default_value = "public")]
        output_visibility: String,
    },
    /// Verify a proof
    Verify {
        /// Path to the proof file
        #[arg(short, long)]
        proof: PathBuf,
        /// Path to the ONNX model file for verification
        #[arg(short, long)]
        model: PathBuf,
        /// Path to input data file for verification (required if input is public)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Path to expected output file for verification (required if output is public)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Input visibility (public/private)
        #[arg(long, default_value = "public")]
        input_visibility: String,
        /// Output visibility (public/private)
        #[arg(long, default_value = "public")]
        output_visibility: String,
    },
    /// Display proof details
    ShowProof {
        /// Path to the proof file
        #[arg(short, long)]
        proof: PathBuf,
    },
}

fn parse_visibility(s: &str) -> Result<Visibility> {
    match s.to_lowercase().as_str() {
        "public" => Ok(Visibility::Public),
        "private" => Ok(Visibility::Private),
        _ => Err(anyhow!(
            "Invalid visibility: {}. Must be 'public' or 'private'",
            s
        )),
    }
}

// Custom serialization for ProverOutput
fn serialize_prover_output(output: &ProverOutput) -> Result<Vec<u8>> {
    bincode::serialize(output).with_context(|| "Failed to serialize prover output")
}

// Custom deserialization for ProverOutput
fn deserialize_prover_output(bytes: &[u8]) -> Result<ProverOutput> {
    bincode::deserialize(bytes).with_context(|| "Failed to deserialize prover output")
}

// Validate input JSON format
fn validate_input_json(input_data: &Value) -> Result<Vec<Vec<f32>>> {
    if !input_data.is_array() {
        return Err(anyhow!("Input JSON must be an array of arrays"));
    }

    let input_array = input_data.as_array().unwrap();
    if input_array.is_empty() {
        return Err(anyhow!("Input array cannot be empty"));
    }

    let mut result = Vec::new();
    for (i, arr) in input_array.iter().enumerate() {
        if !arr.is_array() {
            return Err(anyhow!("Element at index {} must be an array", i));
        }

        let inner_array = arr.as_array().unwrap();
        if inner_array.is_empty() {
            return Err(anyhow!("Inner array at index {} cannot be empty", i));
        }

        let mut inner_result = Vec::new();
        for (j, val) in inner_array.iter().enumerate() {
            if !val.is_number() {
                return Err(anyhow!("Element at index [{}, {}] must be a number", i, j));
            }
            inner_result.push(val.as_f64().unwrap() as f32);
        }
        result.push(inner_result);
    }

    Ok(result)
}

#[allow(deprecated)]
fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Table { model } => {
            let mut table = Table::new();
            table.add_row(row!["Operator", "Input Shape", "Output Shape"]);

            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };

            let model = Model::new(
                model.to_str().context("Invalid model path")?,
                &run_args,
                &VarVisibility {
                    input: Visibility::Public,
                    output: Visibility::Public,
                },
            )
            .with_context(|| format!("Failed to load model from {:?}", model))?;

            for node_type in model.graph.nodes.values() {
                if let mina_zkml::graph::model::NodeType::Node(node) = node_type {
                    let op_type = format!("{:?}", node.op_type);

                    let input_shapes: Vec<String> = node
                        .inputs
                        .iter()
                        .filter_map(|(node_id, _)| {
                            if let Some(mina_zkml::graph::model::NodeType::Node(input_node)) =
                                model.graph.nodes.get(node_id)
                            {
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
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };

            let model = Model::new(
                model.to_str().context("Invalid model path")?,
                &run_args,
                &VarVisibility {
                    input: Visibility::Public,
                    output: Visibility::Public,
                },
            )
            .with_context(|| format!("Failed to load model from {:?}", model))?;

            let json = serde_json::to_string_pretty(&model)
                .context("Failed to serialize model to JSON")?;
            fs::write(output, json)
                .with_context(|| format!("Failed to write JSON to {:?}", output))?;
            println!("Model converted and saved to {:?}", output);
        }
        Commands::Proof {
            model,
            input,
            output,
            input_visibility,
            output_visibility,
        } => {
            // Read and validate input data
            let input_json: Value = serde_json::from_str(
                &fs::read_to_string(input)
                    .with_context(|| format!("Failed to read input file {:?}", input))?,
            )
            .with_context(|| "Failed to parse input file as JSON")?;

            let input_data = validate_input_json(&input_json)?;

            // Parse visibility settings
            let visibility = VarVisibility {
                input: parse_visibility(input_visibility)?,
                output: parse_visibility(output_visibility)?,
            };

            // Load model with visibility settings
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };
            let model = Model::new(
                model.to_str().context("Invalid model path")?,
                &run_args,
                &visibility,
            )
            .with_context(|| format!("Failed to load model from {:?}", model))?;

            // Create prover system and generate proof
            let prover = ProverSystem::new(&model);
            let prover_output = prover
                .prove(&input_data)
                .map_err(|e| anyhow!("Failed to generate proof: {}", e))?;

            // Serialize the prover output
            let serialized_output = serialize_prover_output(&prover_output)?;

            // Save proof data with metadata
            let proof_data = json!({
                "metadata": {
                    "model": model.to_str().context("Invalid model path")?,
                    "timestamp": "chrono::Utc::now().to_rfc3339()",
                    "visibility": {
                        "input": input_visibility,
                        "output": output_visibility,
                    }
                },
                "input_shape": input_data.iter().map(|v| v.len()).collect::<Vec<_>>(),
                "output": prover_output.output,
                "proof": base64::encode(serialized_output),
            });

            fs::write(output, serde_json::to_string_pretty(&proof_data)?)
                .with_context(|| format!("Failed to write proof to {:?}", output))?;
            println!("Proof generated and saved to {:?}", output);
        }
        Commands::Verify {
            proof,
            model,
            input,
            output,
            input_visibility,
            output_visibility,
        } => {
            // Load proof data
            let proof_data: Value = serde_json::from_str(
                &fs::read_to_string(proof)
                    .with_context(|| format!("Failed to read proof file {:?}", proof))?,
            )
            .context("Failed to parse proof data as JSON")?;

            // Parse visibility settings
            let visibility = VarVisibility {
                input: parse_visibility(input_visibility)?,
                output: parse_visibility(output_visibility)?,
            };

            // Load model with visibility settings
            let run_args = RunArgs {
                variables: HashMap::from([("batch_size".to_string(), 1)]),
            };
            let model = Model::new(
                model.to_str().context("Invalid model path")?,
                &run_args,
                &visibility,
            )
            .with_context(|| format!("Failed to load model from {:?}", model))?;

            // Create prover system to get verifier
            let prover = ProverSystem::new(&model);
            let verifier = prover.verifier();

            // Load and validate input data if needed
            let input_data = if visibility.input == Visibility::Public {
                if let Some(input_path) = input {
                    let input_json: Value =
                        serde_json::from_str(&fs::read_to_string(input_path).with_context(
                            || format!("Failed to read input file {:?}", input_path),
                        )?)
                        .context("Failed to parse input file as JSON")?;
                    Some(validate_input_json(&input_json)?)
                } else {
                    return Err(anyhow!("Input data required for public input verification"));
                }
            } else {
                None
            };

            // Load and validate output data if needed
            let output_data = if visibility.output == Visibility::Public {
                if let Some(output_path) = output {
                    let output_json: Value =
                        serde_json::from_str(&fs::read_to_string(output_path).with_context(
                            || format!("Failed to read output file {:?}", output_path),
                        )?)
                        .context("Failed to parse output file as JSON")?;
                    Some(validate_input_json(&output_json)?)
                } else {
                    return Err(anyhow!(
                        "Output data required for public output verification"
                    ));
                }
            } else {
                None
            };

            // Deserialize and verify the proof
            let proof_bytes = base64::decode(
                proof_data["proof"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Missing 'proof' field in proof file"))?,
            )
            .context("Failed to decode base64 proof data")?;
            let prover_output = deserialize_prover_output(&proof_bytes)?;

            let is_valid = verifier
                .verify(
                    &prover_output.proof,
                    input_data.as_deref(),
                    output_data.as_deref(),
                )
                .map_err(|e| anyhow!("Failed to verify proof: {}", e))?;

            if is_valid {
                println!("✅ Proof verification successful!");
                if let Some(output) = prover_output.output {
                    println!("\nModel output:");
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
            } else {
                println!("❌ Proof verification failed!");
            }
        }
        Commands::ShowProof { proof } => {
            // Load and parse proof file
            let proof_data: Value = serde_json::from_str(
                &fs::read_to_string(proof)
                    .with_context(|| format!("Failed to read proof file {:?}", proof))?,
            )
            .context("Failed to parse proof data as JSON")?;

            // Create a table to display proof details
            let mut table = Table::new();
            table.add_row(row!["Property", "Value"]);

            // Add metadata
            if let Some(metadata) = proof_data["metadata"].as_object() {
                for (key, value) in metadata {
                    table.add_row(row![key, format!("{}", value)]);
                }
            }

            // Add input shape
            if let Some(input_shape) = proof_data["input_shape"].as_array() {
                table.add_row(row!["Input Shape", format!("{:?}", input_shape)]);
            }

            // Add output if present
            if let Some(output) = proof_data["output"].as_array() {
                table.add_row(row!["Output", format!("{:?}", output)]);
            }

            table.printstd();
        }
    }

    Ok(())
}
