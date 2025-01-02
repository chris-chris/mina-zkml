use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum GraphError {
    #[error("Unable to read model: More Info: {0}")]
    UnableToReadModel(String),
    #[error("Unable to save model")]
    UnableToSaveModel,
    #[error("Missing batch size")]
    MissingBatchSize,
    #[error("Missing node {0}")]
    MissingNode(usize),
    #[error("Invalid input shape {0}")]
    InvalidInputShape(usize),
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
    #[error("Invalid parameter")]
    InvalidParams,
    #[error("Invalid Node Type")]
    InvalidNodeType,
    #[error("Node Not Found")]
    NodeNotFound,
}
