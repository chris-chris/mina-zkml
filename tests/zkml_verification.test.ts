import { ZKML } from '../src/zkml';
import { resolve } from 'path';

describe('zkml verification', () => {
  let modelPath: string;

  beforeAll(async () => {
    // Create ZKML instance with simple perceptron model
    modelPath = resolve(__dirname, '../models/simple_perceptron.onnx');
  });

  describe('public inputs with public outputs', () => {
    it('should verify successfully with valid inputs', async () => {
      jest.setTimeout(120000);
      let zkml = await ZKML.create(modelPath, {
        variables: { batch_size: 1 }
      }, {
        input: "Public",
        output: "Public"
      });

      // Export verifier for testing
      let verifier = await zkml.exportVerifier();
      const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
      const proofOutput = await zkml.prove(publicInputs);
      const parsedProof = JSON.parse(proofOutput.proof);
      const proof = JSON.stringify(parsedProof, null, 2)
      const output = proofOutput.output as number[][];

      //Format the input and output
      const formattedInput = publicInputs?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public input value");
          return num;
        })
      );
      const formattedOutput = output?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public output value");
          return num;
        })
      );

      const result = await ZKML.verify(
        proof,
        verifier,
        JSON.parse(JSON.stringify(formattedInput)),
        JSON.parse(JSON.stringify(formattedOutput))
      );
      expect(result).toBe(true);
    });

    it('should fail verification with invalid outputs', async () => {
      let zkml = await ZKML.create(modelPath, {
        variables: { batch_size: 1 }
      }, {
        input: "Public",
        output: "Public"
      });

      // Export verifier for testing
      let verifier = await zkml.exportVerifier();
      const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
      const proofOutput = await zkml.prove(publicInputs);
      const parsedProof = JSON.parse(proofOutput.proof);
      const proof = JSON.stringify(parsedProof, null, 2)
      const output = [[0.1, 0.2, 0.3]]

      //Format the input and output
      const formattedInput = publicInputs?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public input value");
          return num;
        })
      );
      const formattedOutput = output?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public output value");
          return num;
        })
      );

      const result = await ZKML.verify(
        proof,
        verifier,
        JSON.parse(JSON.stringify(formattedInput)),
        JSON.parse(JSON.stringify(formattedOutput))
      );
      expect(result).toBe(false);
    });
  });

  describe('public inputs with private outputs', () => {
    it('should verify successfully with valid inputs', async () => {
      let zkml = await ZKML.create(modelPath, {
        variables: { batch_size: 1 }
      }, {
        input: "Public",
        output: "Private"
      });

      // Export verifier for testing
      let verifier = await zkml.exportVerifier();
      const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
      const proofOutput = await zkml.prove(publicInputs);
      const parsedProof = JSON.parse(proofOutput.proof);
      const proof = JSON.stringify(parsedProof, null, 2)
      const output = proofOutput.output as number[][];

      //Format the input and output
      const formattedInput = publicInputs?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public input value");
          return num;
        })
      );

      const result = await ZKML.verify(
        proof,
        verifier,
        JSON.parse(JSON.stringify(formattedInput))
      );
      expect(result).toBe(true);
    });
  });

  describe('private inputs with private outputs', () => {
    it('should verify successfully with valid inputs', async () => {
      let zkml = await ZKML.create(modelPath, {
        variables: { batch_size: 1 }
      }, {
        input: "Private",
        output: "Private"
      });

      // Export verifier for testing
      let verifier = await zkml.exportVerifier();
      const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
      const proofOutput = await zkml.prove(publicInputs);
      const parsedProof = JSON.parse(proofOutput.proof);
      const proof = JSON.stringify(parsedProof, null, 2)
      const result = await ZKML.verify(
        proof,
        verifier
      );
      expect(result).toBe(true);
    });
  });

  describe('private input and public output', () => {
    it('should verify with private input and public output', async () => {
      let zkml = await ZKML.create(modelPath, {
        variables: { batch_size: 1 }
      }, {
        input: "Private",
        output: "Public"
      });

      // Export verifier for testing
      let verifier = await zkml.exportVerifier();
      const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
      const proofOutput = await zkml.prove(publicInputs);
      const parsedProof = JSON.parse(proofOutput.proof);
      const proof = JSON.stringify(parsedProof, null, 2)
      const output = proofOutput.output as number[][];

      //Format the input and output
      const formattedInput = publicInputs?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public input value");
          return num;
        })
      );
      const formattedOutput = output?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public output value");
          return num;
        })
      );

      const result = await ZKML.verify(
        proof,
        verifier,
        undefined,
        JSON.parse(JSON.stringify(formattedOutput))
      );
      expect(result).toBe(true);
    });

    it('should fail verification with invalid output', async () => {
      let zkml = await ZKML.create(modelPath, {
        variables: { batch_size: 1 }
      }, {
        input: "Private",
        output: "Public"
      });

      // Export verifier for testing
      let verifier = await zkml.exportVerifier();
      const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
      const proofOutput = await zkml.prove(publicInputs);
      const parsedProof = JSON.parse(proofOutput.proof);
      const proof = JSON.stringify(parsedProof, null, 2)
      const output = [[1, 2, 3]]

      //Format the input and output
      const formattedOutput = output?.map(row =>
        row.map(val => {
          const num = Number(val);
          if (isNaN(num)) throw new Error("Invalid public output value");
          return num;
        })
      );

      const result = await ZKML.verify(
        proof,
        verifier,
        undefined,
        JSON.parse(JSON.stringify(formattedOutput))
      );
      expect(result).toBe(false);
    });
  });

  describe('Create a proof, then create a new zkml instance and verify the proof', () => {
    it('should verify successfully with valid inputs', async () => {
      jest.setTimeout(300000); // 5 minutes
      let zkml: ZKML | null = null;
      let zkml2: ZKML | null = null;
      
      try {
        // Create first ZKML instance and generate proof
        zkml = await ZKML.create(modelPath, {
          variables: { batch_size: 1 }
        }, {
          input: "Private",
          output: "Private"
        });

        const publicInputs = [[1.0, 0.5, -0.3, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
        const proofOutput = await zkml.prove(publicInputs);
        const parsedProof = JSON.parse(proofOutput.proof);
        const proof = JSON.stringify(parsedProof, null, 2);

        // Clean up first instance
        await zkml.cleanup();
        zkml = null;

        // Force a small delay between instances
        await new Promise(resolve => setTimeout(resolve, 500));

        // Create second ZKML instance
        zkml2 = await ZKML.create(modelPath, {
          variables: { batch_size: 1 }
        }, {
          input: "Private",
          output: "Private"
        });

        // Export verifier and verify proof
        const verifier = await zkml2.exportVerifier();
        const result = await ZKML.verify(
          proof,
          verifier
        );
        expect(result).toBe(true);
      } catch (error) {
        console.error("Test error:", error);
        throw error;
      } finally {
        // Clean up both instances
        if (zkml) await zkml.cleanup();
        if (zkml2) await zkml2.cleanup();
      }
    }, 300000) // Match timeout in test
  })
});
