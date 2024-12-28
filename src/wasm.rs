#![cfg(feature = "wasm")]

extern crate console_error_panic_hook;
extern crate js_sys;
extern crate serde_wasm_bindgen;
extern crate wasm_bindgen;
extern crate wasm_bindgen_futures;
extern crate web_sys;

use crate::graph::model::{
    GraphLoadResult, Model, ParsedNodes, RunArgs, VarVisibility, Visibility,
};
use crate::zk::proof::{ProverOutput, ProverSystem, VerifierSystem};
use crate::zk::ZkOpeningProof;
use base64::prelude::*;
use groupmap::GroupMap;
use js_sys::Promise;
use kimchi::{
    circuits::polynomials::permutation::{vanishes_on_last_n_rows, zk_w},
    proof::ProverProof,
    verifier_index::VerifierIndex,
};
use mina_curves::pasta::{Fp, Vesta, VestaParameters};
use mina_poseidon::{
    constants::PlonkSpongeConstantsKimchi,
    sponge::{DefaultFqSponge, DefaultFrSponge},
};
use once_cell::sync::OnceCell;
use poly_commitment::{commitment::CommitmentCurve, ipa::SRS, SRS as _};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::GenericFactoid;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use web_sys::console;

type SpongeParams = PlonkSpongeConstantsKimchi;
type BaseSponge = DefaultFqSponge<VestaParameters, SpongeParams>;
type ScalarSponge = DefaultFrSponge<Fp, SpongeParams>;

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
const _TS_APPEND_CONTENT: &str = r#"
export type RunArgs = {
    variables: { [key: string]: number };
};

export type VarVisibility = {
    input: "Public" | "Private";
    output: "Public" | "Private";
};

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

export class WasmVerifierSystem {
    constructor(verifier_index_js: string);
    serialize(): string;
    static deserialize(data: string): WasmVerifierSystem;
    readonly verifier_index: string;
    verify(proof_js: string, public_inputs_js: number[][] | null, public_outputs_js: number[][] | null): Promise<boolean>;
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
            proof: serde_json::to_string(&output.proof).expect("Failed to serialize proof"),
            verifier_index: serde_json::to_string(&output.verifier_index)
                .expect("Failed to serialize verifier index"),
        }
    }
}

#[wasm_bindgen]
pub struct WasmModel {
    inner: Model,
}

impl WasmModel {
    fn load_onnx_from_bytes(bytes: &[u8], run_args: &RunArgs) -> Result<GraphLoadResult, JsValue> {
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
    pub fn new(
        model_bytes: &[u8],
        run_args_js: JsValue,
        visibility_js: JsValue,
    ) -> Result<WasmModel, JsValue> {
        console_error_panic_hook::set_once();

        let run_args: JsRunArgs = serde_wasm_bindgen::from_value(run_args_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid run args: {}", e)))?;

        let run_args = RunArgs {
            variables: run_args.variables,
        };

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

        let (model, symbol_values) = Self::load_onnx_from_bytes(model_bytes, &run_args)?;

        let nodes = Model::nodes_from_graph(&model, visibility.clone(), symbol_values)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse nodes: {}", e)))?;

        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };

        let inner = Model {
            graph: parsed_nodes,
            visibility,
        };

        Ok(WasmModel { inner })
    }

    pub fn save(&self) -> Result<Vec<u8>, JsValue> {
        bincode::serialize(&self.inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize model: {}", e)))
    }
}

#[wasm_bindgen]
pub struct WasmProverSystem {
    inner: ProverSystem,
}

#[wasm_bindgen]
pub struct WasmVerifierSystem {
    inner: VerifierSystem,
}

#[wasm_bindgen]
impl WasmProverSystem {
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_bytes: &[u8],
        run_args_js: JsValue,
        visibility_js: JsValue,
    ) -> Result<WasmProverSystem, JsValue> {
        let wasm_model = WasmModel::new(model_bytes, run_args_js, visibility_js)?;
        let inner = ProverSystem::new(&wasm_model.inner);
        Ok(WasmProverSystem { inner })
    }

    #[wasm_bindgen]
    pub fn prove(&self, inputs_js: JsValue) -> Result<Promise, JsValue> {
        let inputs: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(inputs_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid inputs: {}", e)))?;

        let prover_system = self.inner.clone();

        let future = async move {
            let result = prover_system
                .prove(&inputs)
                .map(WasmProverOutput::from)
                .map_err(|e| JsValue::from_str(&format!("Failed to generate proof: {}", e)))?;

            serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
        };

        Ok(future_to_promise(future))
    }

    #[wasm_bindgen]
    pub fn verifier(&self) -> WasmVerifierSystem {
        WasmVerifierSystem {
            inner: self.inner.verifier(),
        }
    }
}

#[wasm_bindgen]
impl WasmVerifierSystem {
    #[wasm_bindgen]
    pub fn serialize(&self) -> Result<String, JsValue> {
        // Serialize verifier index using MessagePack
        let mut buf = Vec::new();
        rmp_serde::encode::write_named(&mut buf, &self.inner.verifier_index).map_err(|e| {
            JsValue::from_str(&format!("Failed to serialize verifier index: {}", e))
        })?;

        Ok(BASE64_STANDARD.encode(&buf))
    }

    #[wasm_bindgen]
    pub fn deserialize(data: &str) -> Result<WasmVerifierSystem, JsValue> {
        let bytes = BASE64_STANDARD
            .decode(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode base64: {}", e)))?;

        // Create SRS for the loaded verifier
        let srs = Arc::new(SRS::<Vesta>::create(4096));

        // Deserialize verifier index
        let mut loaded_verifier_index: VerifierIndex<Vesta, ZkOpeningProof> =
            rmp_serde::decode::from_slice(&bytes).map_err(|e| {
                JsValue::from_str(&format!("Failed to deserialize verifier index: {}", e))
            })?;

        // Set the SRS and other skipped fields in the loaded verifier index
        loaded_verifier_index.srs = Arc::clone(&srs);

        Ok(WasmVerifierSystem {
            inner: VerifierSystem::new(loaded_verifier_index),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn verifier_index(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.verifier_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize verifier index: {}", e)))
    }

    #[wasm_bindgen]
    pub fn verify(
        &self,
        proof_js: JsValue,
        public_inputs_js: &JsValue,
        public_outputs_js: &JsValue,
    ) -> Result<bool, JsValue> {
        console::log_1(&"Starting verification...".into());

        let proof_str = proof_js
            .as_string()
            .ok_or_else(|| JsValue::from_str("Invalid proof: not a string"))?;

        console::log_1(&format!("Parsing proof of length: {}", proof_str.len()).into());

        let proof: ProverProof<Vesta, ZkOpeningProof> =
            serde_json::from_str(&proof_str).map_err(|e| {
                console::error_1(&format!("Failed to parse proof JSON: {}", e).into());
                JsValue::from_str(&format!("Failed to parse proof: {}", e))
            })?;

        console::log_1(&"Successfully parsed proof".into());

        let public_inputs: Option<Vec<Vec<f32>>> = if !public_inputs_js.is_null()
            && !public_inputs_js.is_undefined()
        {
            console::log_1(&"Parsing public inputs...".into());
            let inputs = serde_wasm_bindgen::from_value(public_inputs_js.clone()).map_err(|e| {
                console::error_1(&format!("Failed to parse public inputs: {}", e).into());
                JsValue::from_str(&format!("Invalid public inputs: {}", e))
            })?;
            console::log_1(&"Successfully parsed public inputs".into());
            Some(inputs)
        } else {
            None
        };

        let public_outputs: Option<Vec<Vec<f32>>> =
            if !public_outputs_js.is_null() && !public_outputs_js.is_undefined() {
                console::log_1(&"Parsing public outputs...".into());
                let outputs =
                    serde_wasm_bindgen::from_value(public_outputs_js.clone()).map_err(|e| {
                        console::error_1(&format!("Failed to parse public outputs: {}", e).into());
                        JsValue::from_str(&format!("Invalid public outputs: {}", e))
                    })?;
                console::log_1(&"Successfully parsed public outputs".into());
                Some(outputs)
            } else {
                None
            };

        // Setup group map for verification
        let group_map = <Vesta as CommitmentCurve>::Map::setup();

        // Convert inputs and outputs to field elements for verification
        let mut public_values = Vec::new();

        // Add public inputs
        if let Some(ref inputs) = public_inputs {
            for input_vec in inputs {
                for x in input_vec {
                    const SCALE: f32 = 1_000_000.0;
                    let scaled = (x * SCALE) as i64;
                    let field_element = if scaled < 0 {
                        -Fp::from((-scaled) as u64)
                    } else {
                        Fp::from(scaled as u64)
                    };
                    public_values.push(field_element);
                }
            }
        }

        // Add public outputs
        if let Some(ref outputs) = public_outputs {
            for output_vec in outputs {
                for x in output_vec {
                    const SCALE: f32 = 1_000_000.0;
                    let scaled = (x * SCALE) as i64;
                    let field_element = if scaled < 0 {
                        -Fp::from((-scaled) as u64)
                    } else {
                        Fp::from(scaled as u64)
                    };
                    public_values.push(field_element);
                }
            }
        }

        console::log_1(&"Starting proof verification...".into());
        // let result = kimchi::verifier::verify::<Vesta, BaseSponge, ScalarSponge, ZkOpeningProof>(
        //     &group_map,
        //     &self.inner.verifier_index,
        //     &proof,
        //     &public_values,
        // );
        let result = self
            .inner
            .verify(&proof, public_inputs.as_deref(), public_outputs.as_deref());

        match result {
            Ok(_) => {
                console::log_1(&"Verification completed successfully".into());
                Ok(true)
            }
            Err(e) => {
                console::error_1(&format!("Verification failed: {}", e).into());
                Err(JsValue::from_str(&format!("Failed to verify proof: {}", e)))
            }
        }
    }
}
