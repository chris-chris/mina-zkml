//! Functions for integrating with Tract for tensor manipulation and conversion.
//!
//! These utilities facilitate conversions between Rust standard data structures (e.g., vectors)
//! and Tract's tensor representation, as well as performing common tensor operations.

use anyhow::Error;
use std::fmt::Debug;
use tract_data::internal::tract_smallvec::smallvec;
use tract_onnx::prelude::*;

/// Converts a vector and its dimensions into a Tract-compatible tensor vector.
///
/// This function takes a slice representing the dimensions and another slice representing the data,
/// and returns a `TVec<TValue>` containing the tensor.
///
/// # Errors
/// Returns an error if:
/// - The number of dimensions exceeds 4.
/// - The data cannot be shaped into the specified dimensions.
///
/// # Examples
/// ```
/// use your_crate::vec_to_tract_vec;
/// use tract_onnx::prelude::*;
///
/// let dims = &[2, 2];
/// let data = &[1.0, 2.0, 3.0, 4.0];
/// let result = vec_to_tract_vec(dims, data);
/// assert!(result.is_ok());
/// ```
pub fn vec_to_tract_vec<T: Debug + Datum + Copy>(
    dims: &[usize],
    data: &[T],
) -> TractResult<TVec<TValue>> {
    if dims.len() > 4 {
        return Err(Error::msg(format!(
            "Invalid dimensions: dims has more than 4 elements (dims.len() = {})",
            dims.len()
        )));
    }

    let mut shape: [usize; 4] = [1; 4];
    for (i, &dim) in dims.iter().enumerate() {
        shape[i] = dim;
    }

    let tensor_data = Tensor::from_shape(&shape, data)?;
    let tvec = smallvec![TValue::from_const(Arc::new(tensor_data))];

    Ok(tvec)
}

/// Converts a vector and its dimensions into a Tract-compatible tensor.
///
/// This function is similar to `vec_to_tract_vec`, but it directly returns a `Tensor`
/// instead of a tensor vector.
///
/// # Errors
/// Returns an error if:
/// - The number of dimensions exceeds 4.
/// - The data cannot be shaped into the specified dimensions.
///
/// # Examples
/// ```
/// use your_crate::vec_to_tensor;
/// use tract_onnx::prelude::*;
///
/// let dims = &[3, 3];
/// let data = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let result = vec_to_tensor(dims, data);
/// assert!(result.is_ok());
/// ```
pub fn vec_to_tensor<T: Debug + Datum + Copy>(dims: &[usize], data: &[T]) -> TractResult<Tensor> {
    if dims.len() > 4 {
        return Err(Error::msg(format!(
            "Invalid dimensions: dims has more than 4 elements (dims.len() = {})",
            dims.len()
        )));
    }

    let mut shape: [usize; 4] = [1; 4];
    for (i, &dim) in dims.iter().enumerate() {
        shape[i] = dim;
    }

    let tensor_data = Tensor::from_shape(&shape, data)?;

    Ok(tensor_data)
}

/// Converts a Tract tensor into a Rust vector.
///
/// This function extracts the data from a `Tensor` and returns it as a standard Rust vector.
///
/// # Errors
/// Returns an error if the tensor cannot be converted into a slice of the desired type.
///
/// # Examples
/// ```
/// use your_crate::tensor_to_vec;
/// use tract_onnx::prelude::*;
///
/// let tensor = Tensor::from_shape(&[2, 2], &[1, 2, 3, 4]).unwrap();
/// let result: Vec<i32> = tensor_to_vec(&tensor).unwrap();
/// assert_eq!(result, vec![1, 2, 3, 4]);
/// ```
pub fn tensor_to_vec<T: Datum>(tensor: &Tensor) -> TractResult<Vec<T>> {
    Ok(tensor.as_slice::<T>()?.to_vec())
}
