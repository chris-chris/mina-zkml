# mina-zkml

**mina-zkml** is an open-source library designed to integrate AI technologies seamlessly with the MINA blockchain. By leveraging zero-knowledge machine learning (zkML), this library empowers developers to build and deploy AI models securely and efficiently on the MINA blockchain.

## 🚀 Features

- **zkML Enabled Library**: Facilitates the implementation of zero-knowledge machine learning within the MINA ecosystem.
- **ONNX Conversion Support**: Easily convert models to the Open Neural Network Exchange (ONNX) format for broader compatibility.
- **Verifier Script Generation**: Automatically generate verifier scripts tailored for the MINA blockchain.
- **Verifier Deployment**: Deploy verifier scripts directly to the MINA blockchain with ease.

## 🛠️ Installation

### Prerequisites
Ensure you have [Rust](https://www.rust-lang.org/tools/install) installed on your system.

### CLI Installation (Linux)
```bash
curl -L -o mina-zkml-cli https://github.com/chris-chris/mina-zkml/releases/latest/download/mina-zkml-cli
chmod +x mina-zkml-cli
./mina-zkml-cli --help
```

### From Source
```bash
# Clone the repository
git clone https://github.com/chris-chris/mina-zkml.git

# Navigate to the project directory
cd mina-zkml

# Build the project
cargo build --release

# Run the CLI
./target/release/mina-zkml-cli --help
```

## 📚 Usage

### Running Examples
mina-zkml comes with example projects to demonstrate its capabilities. Below are two primary examples:

#### Perceptron ZKML Model Prediction
Execute the perceptron model prediction using the following command:
```bash
cargo run --example perceptron
```

#### MNIST Lenet ZKML Model End-to-End Example
Run the MNIST model end-to-end example through the provided notebook:
`./examples/notebook/lenet.ipynb`

### Creating a Verifier Script & Deploying to MINA Devnet
For detailed instructions, visit the [mina-zkml-verifier repository](https://github.com/chris-chris/mina-zkml-verifier).

## 🤝 Contributing
We welcome contributions from the community! If you're interested in contributing, please follow these steps:

1. Check out the [open issues](https://github.com/chris-chris/mina-zkml/issues) for ideas on where to contribute.
2. Fork the repository and create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

If you have any questions, feel free to reach out to the maintainers:
- [Chris](https://github.com/chris-chris)
- [sshivaditya](https://github.com/sshivaditya)
- [Hugo](https://github.com/energyGiver)