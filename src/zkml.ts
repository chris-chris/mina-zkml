import { wasm_init, wasm_main, WasmProverSystem, WasmProverOutput, WasmVerifierSystem } from '../pkg/mina_zkml.js';
import { readFile } from 'fs/promises';
import { isAbsolute, resolve } from 'path';

// Types from wasm bindings
export type RunArgs = {
    variables: { [key: string]: number };
};

export type VarVisibility = {
    input: "Public" | "Private";
    output: "Public" | "Private";
};

// WASM environment setup
const wasmEnv = {
  memory: new WebAssembly.Memory({ initial: 512, maximum: 1024 }),
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
  private proofSystem: WasmProverSystem | null;
  private static initialized = false;
  private static initPromise: Promise<void> | null = null;
  
  private constructor(proofSystem: WasmProverSystem) {
    this.proofSystem = proofSystem;
  }

  async cleanup() {
    if (this.proofSystem) {
      try {
        // Force garbage collection of the proof system
        this.proofSystem = null;
        // Force a small delay to allow cleanup
        await new Promise(resolve => setTimeout(resolve, 100));
      } catch (error) {
        console.error("Cleanup error:", error);
      }
    }
  }

  private ensureProofSystem(): WasmProverSystem {
    if (!this.proofSystem) {
      throw new Error("ProofSystem is not initialized or has been cleaned up");
    }
    return this.proofSystem;
  }

  private static async initialize() {
    if (!ZKML.initialized) {
      if (!ZKML.initPromise) {
        ZKML.initPromise = Promise.all([
          Promise.resolve().then(() => {
            wasm_init();
            wasm_main();
          })
        ]).then(() => {
          ZKML.initialized = true;
        });
      }
      await ZKML.initPromise;
    }
  }

  private static async loadModelBytes(modelPath: string): Promise<Uint8Array> {
    const path = isAbsolute(modelPath) ? modelPath : resolve(process.cwd(), modelPath);
    const buffer = await readFile(path);
    return new Uint8Array(buffer);
  }

  static async create(
    modelPath: string,
    runArgs: RunArgs = { variables: { batch_size: 1 } },
    visibility: VarVisibility = {
      input: "Private",
      output: "Private"
    }
  ): Promise<ZKML> {
    await ZKML.initialize();
    const modelBytes = await ZKML.loadModelBytes(modelPath);
    const proofSystem = new WasmProverSystem(
      modelBytes,
      runArgs,
      {
        input: visibility.input,
        output: visibility.output
      }
    );
    return new ZKML(proofSystem);
  }

  async prove(inputs: number[][]): Promise<WasmProverOutput> {
    const proofSystem = this.ensureProofSystem();
    const result = await proofSystem.prove(inputs);
    return result;
  }

  async exportVerifier(): Promise<WasmVerifierSystem> {
    const proofSystem = this.ensureProofSystem();
    const verifier = proofSystem.verifier();
    return verifier;
  }

  async serializeVerifier(): Promise<string> {
    try {
      const verifier = await this.exportVerifier();
      const serialized = verifier.serialize();
      return serialized;
    } catch (error) {
      throw error;
    }
  }

  static async deserializeVerifier(serialized: string): Promise<WasmVerifierSystem> {
    try {
      await ZKML.initialize();
      const verifier = WasmVerifierSystem.deserialize(serialized);
      return verifier;
    } catch (error) {
      throw error;
    }
  }

  static async verify(
    proof: string,
    verifier: WasmVerifierSystem,
    publicInputs?: number[][],
    publicOutputs?: number[][]
  ): Promise<boolean> {
    await ZKML.initialize();
    try {
      // Validate inputs
      if (!proof) throw new Error("Proof string is empty");
      if (!verifier) throw new Error("Verifier is not provided");
      // Verify the proof
      const result = await verifier.verify(
        proof,
        publicInputs || null,
        publicOutputs || null
      );
      return result;
    } catch (error) {
      console.error("Verification error:", error);
      return false;
    }
  }
}
