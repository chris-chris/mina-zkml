
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq)]
pub enum GraphError {
    #[error("Unable to read model")]
    UnableToReadModel,
    #[error("Unable to save model")]
    UnableToSaveModel,
    #[error("Missing batch size")]
    MissingBatchSize,
    #[error("Missing node {0}")]
    MissingNode(usize),
    #[error("Invalid input shape")]
    InvalidInputShape,
    #[error("Invalid input slot {0}")]
    InvalidInputSlot(usize),
    #[error("Invalid output slot {0}")]
    InvalidOutputSlot(usize),
    #[error("Cyclic dependency detected")]
    CyclicDependency,
    #[error("Unsupported operation")]
    UnsupportedOperation,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
}
