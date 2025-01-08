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
    #[error("Invalid input found at [{0}]")]
    InvalidInput(String),
    #[error("Invalid input slot {0}")]
    InvalidInputSlot(usize),
    #[error("Invalid output slot {0}")]
    InvalidOutputSlot(usize),
    #[error("Cyclic dependency detected")]
    CyclicDependency,
    #[error("Unsupported operation")]
    UnsupportedOperation,
    #[error("Invalid Output shape")]
    InvalidOutputShape,
    #[error("Invalid parameter")]
    InvalidParams,
    #[error("Invalid node type")]
    InvalidNodeType,
    #[error("Node not found")]
    NodeNotFound,
    #[error("Missing attributes: {0}")]
    MissingAttributes(String),
    #[error("Tract parsing failure: {0}")]
    TractParseFailure(String),
}

impl From<anyhow::Error> for GraphError {
    fn from(err: anyhow::Error) -> Self {
        GraphError::TractParseFailure(err.to_string())
    }
}
