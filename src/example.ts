import { VarVisibility } from '../pkg/mina_zkml.js';
import { ZKML } from './zkml.js';
import { join } from 'path';

async function main() {
    try {
        // Initialize ZKML with a simple perceptron model
        const modelPath = join(process.cwd(), 'models', 'simple_perceptron.onnx');
        console.log('Loading model from:', modelPath);
        
        const zkml = await ZKML.create(modelPath, {
            variables: { batch_size: 1 }
        }, {
            input: "Public",
            output: "Public"
        });

        // Input with correct shape: 5 original values + 5 padding zeros
        const inputs = [[
            1.0, 0.5, -0.3, 0.8, -0.2,  // Original values
            0.0, 0.0, 0.0, 0.0, 0.0     // Padding to reach size 10
        ]];

        console.log('Generating proof...');
        const result = await zkml.prove(inputs);
        console.log('Proof generated:', {
            output: result.output,
            proofLength: result.proof.length,
            verifierIndexLength: result.verifier_index.length
        });

        console.log('Verifying proof...');
        const isValid = await zkml.verify(
            result.proof,
            inputs,
            result.output
        );
        console.log('Proof verified:', isValid);
    } catch (error) {
        console.error('Error:', error);
        if (error instanceof Error) {
            console.error('Stack:', error.stack);
        }
        process.exit(1);
    }
}

main().catch(error => {
    console.error('Unhandled error:', error);
    if (error instanceof Error) {
        console.error('Stack:', error.stack);
    }
    process.exit(1);
});
