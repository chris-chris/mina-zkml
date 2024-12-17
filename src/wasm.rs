#![cfg(feature = "wasm")]

extern crate wasm_bindgen;
extern crate js_sys;
extern crate web_sys;
extern crate console_error_panic_hook;
extern crate serde_wasm_bindgen;
extern crate wasm_bindgen_futures;

use wasm_bindgen::prelude::*;
use js_sys::Promise;
use serde::{Serialize, Deserialize};
use crate::graph::model::{Model, RunArgs, VarVisibility, Visibility, GraphLoadResult, ParsedNodes};
use crate::zk::proof::{ProofSystem, ProverOutput};
use wasm_bindgen_futures::future_to_promise;
use web_sys::console;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::GenericFactoid;

// Custom error type for WASM
#[derive(Debug)]
pub struct WasmError(String);

impl From<std::io::Error> for WasmError {
    fn from(error: std::io::Error) -> Self {
        WasmError(error.to_string())
    }
}

#[derive(Serialize, Deserialize)]
struct JsRunArgs {
    variables: std::collections::HashMap<String, usize>,
}

#[derive(Serialize, Deserialize)]
struct JsVisibility {
    input: String,
    output: String,
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export interface RunArgs {
    variables: { [key: string]: number };
}

export interface VarVisibility {
    input: "Public" | "Private";
    output: "Public" | "Private";
}

export interface ProverOutput {
    output?: number[][];
    proof: string;
    verifier_index: string;
}
"#;

#[derive(Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmProverOutput {
    output: Option<Vec<Vec<f32>>>,
    proof: String,
    verifier_index: String,
}

#[wasm_bindgen]
impl WasmProverOutput {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.output)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize output: {}", e)))
    }

    #[wasm_bindgen(getter)]
    pub fn proof(&self) -> String {
        self.proof.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn verifier_index(&self) -> String {
        self.verifier_index.clone()
    }
}

impl From<ProverOutput> for WasmProverOutput {
    fn from(output: ProverOutput) -> Self {
        Self {
            output: output.output,
            proof: serde_json::to_string(&output.proof)
                .expect("Failed to serialize proof"),
            verifier_index: serde_json::to_string(&output.verifier_index)
                .expect("Failed to serialize verifier index"),
        }
    }
}

// Custom Model implementation for WASM
#[wasm_bindgen]
pub struct WasmModel {
    inner: Model
}

impl WasmModel {
    // WASM-compatible ONNX loading function that takes bytes instead of file path
    fn load_onnx_from_bytes(
        bytes: &[u8],
        run_args: &RunArgs,
    ) -> Result<GraphLoadResult, JsValue> {
        // Create model from bytes instead of file
        let mut model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .map_err(|e| JsValue::from_str(&format!("Failed to load model from bytes: {}", e)))?;

        let variables: std::collections::HashMap<String, usize> =
            std::collections::HashMap::from_iter(run_args.variables.clone());

        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node_mut(id.node);
            let mut fact: InferenceFact = input.outputs[0].fact.clone();

            for (i, x) in fact.clone().shape.dims().enumerate() {
                if matches!(x, GenericFactoid::Any) {
                    let batch_size = variables
                        .get("batch_size")
                        .ok_or_else(|| JsValue::from_str("Missing batch_size"))?;
                    fact.shape
                        .set_dim(i, tract_onnx::prelude::TDim::Val(*batch_size as i64));
                }
            }

            model
                .set_input_fact(i, fact)
                .map_err(|e| JsValue::from_str(&format!("Failed to set input fact: {}", e)))?;
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            model
                .set_output_fact(i, InferenceFact::default())
                .map_err(|e| JsValue::from_str(&format!("Failed to set output fact: {}", e)))?;
        }

        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbol_table.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
            console::log_1(&format!("Set {} to {}", symbol, value).into());
        }

        let typed_model = model
            .into_typed()
            .map_err(|e| JsValue::from_str(&format!("Failed to analyze and convert: {}", e)))?
            .concretize_dims(&symbol_values)
            .map_err(|e| JsValue::from_str(&format!("Failed to concretize dims: {}", e)))?
            .into_decluttered()
            .map_err(|e| JsValue::from_str(&format!("Failed to declutter: {}", e)))?;

        Ok((typed_model, symbol_values))
    }
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], run_args_js: JsValue, visibility_js: JsValue) -> Result<WasmModel, JsValue> {
        console_error_panic_hook::set_once();

        // Parse run args
        let run_args: JsRunArgs = serde_wasm_bindgen::from_value(run_args_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid run args: {}", e)))?;
        
        let run_args = RunArgs {
            variables: run_args.variables,
        };

        // Parse visibility
        let visibility: JsVisibility = serde_wasm_bindgen::from_value(visibility_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid visibility: {}", e)))?;
        
        let visibility = VarVisibility {
            input: match visibility.input.as_str() {
                "Public" => Visibility::Public,
                "Private" => Visibility::Private,
                _ => return Err(JsValue::from_str("Invalid input visibility")),
            },
            output: match visibility.output.as_str() {
                "Public" => Visibility::Public,
                "Private" => Visibility::Private,
                _ => return Err(JsValue::from_str("Invalid output visibility")),
            },
        };

        // Load ONNX model from bytes
        let (model, symbol_values) = Self::load_onnx_from_bytes(model_bytes, &run_args)?;
        
        // Convert to parsed nodes
        let nodes = Model::nodes_from_graph(&model, visibility.clone(), symbol_values)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse nodes: {}", e)))?;

        // Create parsed nodes
        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };

        // Create model
        let inner = Model {
            graph: parsed_nodes,
            visibility,
        };

        Ok(WasmModel { inner })
    }

    // Override file system operations with in-memory alternatives
    pub fn save(&self) -> Result<Vec<u8>, JsValue> {
        bincode::serialize(&self.inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize model: {}", e)))
    }

    // Replace file logging with console logging
    pub fn log_weights_and_biases(&self) -> Result<(), JsValue> {
        for (node_idx, node_type) in &self.inner.graph.nodes {
            if let crate::graph::model::NodeType::Node(node) = node_type {
                if matches!(node.op_type, crate::graph::model::OperationType::Const) {
                    console::log_1(&format!("Const Node {}", node_idx).into());
                    console::log_1(&format!("Dimensions: {:?}", node.out_dims).into());
                    
                    if let Some(weights) = &node.weights {
                        console::log_1(&format!("Total elements: {}", weights.len()).into());
                        
                        // Calculate statistics
                        let min = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let sum: f32 = weights.iter().sum();
                        let mean = sum / weights.len() as f32;
                        let non_zero_count = weights.iter().filter(|&&x| x != 0.0).count();
                        
                        console::log_1(&format!("\nStatistics:").into());
                        console::log_1(&format!("  Total elements: {}", weights.len()).into());
                        console::log_1(&format!("  Non-zero elements: {}", non_zero_count).into());
                        console::log_1(&format!("  Zero elements: {}", weights.len() - non_zero_count).into());
                        console::log_1(&format!("  Sparsity: {:.2}%", 
                            (weights.len() - non_zero_count) as f32 / weights.len() as f32 * 100.0).into());
                        console::log_1(&format!("  Min: {:.6}", min).into());
                        console::log_1(&format!("  Max: {:.6}", max).into());
                        console::log_1(&format!("  Mean: {:.6}", mean).into());
                    }
                }
            }
        }
        Ok(())
    }
}

#[wasm_bindgen]
pub struct WasmProofSystem {
    inner: ProofSystem,
}

#[wasm_bindgen]
impl WasmProofSystem {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], run_args_js: JsValue, visibility_js: JsValue) -> Result<WasmProofSystem, JsValue> {
        // Create WasmModel first
        let wasm_model = WasmModel::new(model_bytes, run_args_js, visibility_js)?;
        
        // Create proof system from model
        let inner = ProofSystem::new(&wasm_model.inner);

        Ok(WasmProofSystem { inner })
    }

    #[wasm_bindgen]
    pub fn prove(&self, inputs_js: JsValue) -> Result<Promise, JsValue> {
        // Parse inputs
        let inputs: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(inputs_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid inputs: {}", e)))?;

        // Clone proof system for async operation
        let proof_system = self.inner.clone();

        // Create async operation
        let future = async move {
            let result = proof_system.prove(&inputs)
                .map(WasmProverOutput::from)
                .map_err(|e| JsValue::from_str(&format!("Failed to generate proof: {}", e)))?;
            
            serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
        };

        // Convert future to Promise
        Ok(future_to_promise(future))
    }

    #[wasm_bindgen]
    pub fn verify(
        &self,
        proof: String,
        public_inputs_js: &JsValue,
        public_outputs_js: &JsValue,
    ) -> Result<Promise, JsValue> {
        // Parse proof
        let proof = serde_json::from_str(&proof)
            .map_err(|e| JsValue::from_str(&format!("Invalid proof: {}", e)))?;

        // Parse public inputs if provided
        let public_inputs: Option<Vec<Vec<f32>>> = if !public_inputs_js.is_null() && !public_inputs_js.is_undefined() {
            Some(serde_wasm_bindgen::from_value(public_inputs_js.clone())
                .map_err(|e| JsValue::from_str(&format!("Invalid public inputs: {}", e)))?)
        } else {
            None
        };

        // Parse public outputs if provided
        let public_outputs: Option<Vec<Vec<f32>>> = if !public_outputs_js.is_null() && !public_outputs_js.is_undefined() {
            Some(serde_wasm_bindgen::from_value(public_outputs_js.clone())
                .map_err(|e| JsValue::from_str(&format!("Invalid public outputs: {}", e)))?)
        } else {
            None
        };

        // Clone proof system for async operation
        let proof_system = self.inner.clone();

        // Create async operation
        let future = async move {
            let result = proof_system.verify(
                &proof,
                public_inputs.as_deref(),
                public_outputs.as_deref()
            ).map_err(|e| JsValue::from_str(&format!("Failed to verify proof: {}", e)))?;
            
            serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
        };

        // Convert future to Promise
        Ok(future_to_promise(future))
    }
}
