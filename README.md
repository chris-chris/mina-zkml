# mina-zkml (Working Title)

**mina-zkml** is an open-source library designed to integrate AI technologies seamlessly with the MINA blockchain. By leveraging zero-knowledge machine learning (zkML), this library empowers developers to build and deploy AI models securely and efficiently on the MINA blockchain.

## 🚀 Features

- **zkML Enabled Library**: Facilitates the implementation of zero-knowledge machine learning within the MINA ecosystem.
- **ONNX Conversion Support**: Easily convert models to the Open Neural Network Exchange (ONNX) format for broader compatibility.
- **Verifier Script Generation**: Automatically generate verifier scripts tailored for the MINA blockchain.
- **Verifier Deployment**: Deploy verifier scripts directly to the MINA blockchain with ease.

## 🛠️ Installation

To get started with **mina-zkml**, ensure you have [Rust](https://www.rust-lang.org/tools/install) installed on your system.

```bash
# Clone the repository
git clone https://github.com/chris-chris/mina-zkml.git

# Navigate to the project directory
cd mina-zkml

# Build the project
cargo build --release

./target/release/mina-zkml-cli --help
```

## 📚 Usage

Running Examples

mina-zkml comes with example projects to demonstrate its capabilities. Below are two primary examples:

### Perceptron ZKML Model Prediction

Execute the perceptron model prediction using the following command:

```bash
cargo run --example perceptron
```

### MNIST Lenet ZKML Model End 2 End Example

Execute the MNIST model end-2-end example through notebook.

`./examples/notebook/lenet.ipynb`

### Creating a Verifier Script & Deploy it to the MINA devnet

https://github.com/chris-chris/mina-zkml-verifier
