use thiserror::Error;

/// Represents various errors that can occur within the graph processing module.
///
/// The `GraphError` enum is used to categorize and provide detailed error messages
/// for different failure scenarios encountered during graph operations. This helps
/// in debugging and handling errors gracefully in applications that utilize this module.
///
/// This enum is used to represent errors that can occur during graph operations. Each variant
/// provides a specific error message that can be used to understand the nature of the failure.
///
/// # Safety
///
/// This enum does not involve any unsafe code and is safe to use in any context.
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

/// Converts an `anyhow::Error` into a `GraphError`.
///
/// This implementation allows for seamless conversion of errors from the `anyhow`
/// library into `GraphError`, specifically mapping them to the `TractParseFailure`
/// variant. This is useful for integrating with code that uses `anyhow` for error
/// handling.
///
/// # Arguments
///
/// * `err` - An `anyhow::Error` that needs to be converted.
///
/// # Returns
///
/// A `GraphError` variant representing the `anyhow` error.
impl From<anyhow::Error> for GraphError {
    fn from(err: anyhow::Error) -> Self {
        GraphError::TractParseFailure(err.to_string())
    }
}
