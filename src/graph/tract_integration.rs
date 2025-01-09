use anyhow::Error;
use tract_data::internal::tract_smallvec::smallvec;
use tract_onnx::prelude::*;
// use tract_onnx::tract_core;
// use tract_onnx::tract_core::ops::array::Gather;
// use tract_onnx::tract_core::ops::cnn::{Conv, MaxPool};
// use tract_onnx::tract_core::ops::nn::Softmax;
// use tract_onnx::tract_core::ops::nn::{Reduce, SoftmaxExp};
// use tract_onnx::{prelude::*, tract_hir::ops::konst::Const, tract_hir::ops::scan::Scan};

pub fn vec_to_eval_input(dims: &[usize], data: &[f32]) -> TractResult<TVec<TValue>> {
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

pub fn tensor_to_vec<T: Datum>(tensor: &Tensor) -> TractResult<Vec<T>> {
    Ok(tensor.as_slice::<T>()?.to_vec())
}
