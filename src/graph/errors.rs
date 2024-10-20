use thiserror::Error;

/// circuit related errors.
#[derive(Debug, Error)]
pub enum GraphError {
    /// Missing Batch Size
    #[error("unknown dimension batch_size in model inputs, set batch_size in variables")]
    MissingBatchSize,
    // Unable to Read ONNX Model
    #[error("unable to read onnx model")]
    UnableToReadModel,
    //Missing Node
    #[error("missing node in model")]
    MissingNode(usize),
}