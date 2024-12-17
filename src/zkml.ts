import init, { ProverOutput, RunArgs, VarVisibility, wasm_init, wasm_main, WasmProofSystem } from '../pkg/mina_zkml.js';
import { readFile } from 'fs/promises';
import { isAbsolute, resolve } from 'path';

// WASM environment setup
const wasmEnv = {
  memory: new WebAssembly.Memory({ initial: 256 }),
  now: () => BigInt(Date.now()),
  seed: () => BigInt(Date.now()),
  abort: () => { throw new Error('abort called'); },
  process_exit: (code: number) => {
    if (code !== 0) {
      throw new Error(`process.exit(${code})`);
    }
  }
};

// Make wasmEnv globally available
(globalThis as any).env = wasmEnv;

export class ZKML {
  private proofSystem: WasmProofSystem;
  private static initialized = false;
  private static initPromise: Promise<void> | null = null;
  
  private constructor(proofSystem: any) {
    this.proofSystem = proofSystem;
  }

  private static async initialize() {
    // if (!ZKML.initialized) {
    //   if (!ZKML.initPromise) {
    //     ZKML.initPromise = init().then(() => {
    //       wasm_init();
    //       wasm_main();
    //       ZKML.initialized = true;
    //     });
    //   }
    //   await ZKML.initPromise;
    // }
  }

  /**
   * Load model bytes from a path or URL
   * @param modelPath Path or URL to the ONNX model file
   * @returns Uint8Array of model bytes
   */
  private static async loadModelBytes(modelPath: string): Promise<Uint8Array> {
    try {
      // Check if modelPath is a URL
      new URL(modelPath);
      // If no error was thrown, it's a valid URL
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }
      return new Uint8Array(await response.arrayBuffer());
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('Invalid URL')) {
        // Not a URL, treat as local file path
        const path = isAbsolute(modelPath) ? modelPath : resolve(process.cwd(), modelPath);
        const buffer = await readFile(path);
        return new Uint8Array(buffer);
      }
      throw err;
    }
  }

  /**
   * Initialize ZKML with a model
   * @param modelPath Path to the ONNX model file
   * @param runArgs Runtime arguments for model execution
   * @param visibility Model input/output visibility configuration
   */
  static async create(
    modelPath: string,
    runArgs: RunArgs = { variables: { batch_size: 1 } },
    visibility: VarVisibility = {
      input: "Private",
      output: "Private"
    }
  ): Promise<ZKML> {
    await ZKML.initialize();

    console.log(`Loading model from: ${modelPath}`);
    const modelBytes = await ZKML.loadModelBytes(modelPath);
    console.log(`Loaded ${modelBytes.length} bytes`);

    // Create proof system with model bytes
    const proofSystem = new WasmProofSystem(
      modelBytes,
      runArgs,
      {
        input: visibility.input,
        output: visibility.output
      }
    );
    return new ZKML(proofSystem);
  }

  /**
   * Generate a proof for model execution
   * @param inputs Model input values
   * @returns Proof and optional public outputs
   */
  async prove(inputs: number[][]): Promise<ProverOutput> {
    const result = await this.proofSystem.prove(inputs);
    
    return {
      output: result.output as number[][] | undefined,
      proof: result.proof,
      verifier_index: result.verifier_index
    };
  }

  /**
   * Verify a proof with optional public inputs/outputs
   * @param proof The proof to verify
   * @param publicInputs Optional public input values
   * @param publicOutputs Optional public output values
   * @returns Whether the proof is valid
   */
  async verify(
    proof: string,
    publicInputs?: number[][],
    publicOutputs?: number[][]
  ): Promise<boolean> {
    const result = await this.proofSystem.verify(
      proof,
      publicInputs || null,
      publicOutputs || null
    );
    return result as boolean;
  }
}
