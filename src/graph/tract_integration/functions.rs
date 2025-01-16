use anyhow::Error;
use std::fmt::Debug;
use tract_data::internal::tract_smallvec::smallvec;
use tract_onnx::prelude::*;

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

pub fn tensor_to_vec<T: Datum>(tensor: &Tensor) -> TractResult<Vec<T>> {
    Ok(tensor.as_slice::<T>()?.to_vec())
}
