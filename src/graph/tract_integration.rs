use anyhow::Error;
use tract_data::internal::tract_smallvec::smallvec;
use tract_onnx::prelude::*;
// use tract_onnx::tract_core;
// use tract_onnx::tract_core::ops::array::Gather;
// use tract_onnx::tract_core::ops::cnn::{Conv, MaxPool};
// use tract_onnx::tract_core::ops::nn::Softmax;
// use tract_onnx::tract_core::ops::nn::{Reduce, SoftmaxExp};
// use tract_onnx::{prelude::*, tract_hir::ops::konst::Const, tract_hir::ops::scan::Scan};
use std::fmt::Debug;
use tract_onnx::prelude::Datum;
pub use tract_onnx::tract_core::ops::nn::Reducer;

pub fn vec_to_eval_input<T: Debug + Datum + Copy>(
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CustomReducer {
    ArgMax(bool),
    ArgMin(bool),
    Max,
    Min,
    Prod,
    Sum,
    MeanOfSquares,
}

impl CustomReducer {
    /// Map between `CustomReducer`, `Reducer`, and their corresponding indices.
    const REDUCER_MAP: &'static [(Self, Reducer, usize)] = &[
        (Self::ArgMax(true), Reducer::ArgMax(true), 0),
        (Self::ArgMax(false), Reducer::ArgMax(false), 1),
        (Self::ArgMin(true), Reducer::ArgMin(true), 2),
        (Self::ArgMin(false), Reducer::ArgMin(false), 3),
        (Self::Max, Reducer::Max, 4),
        (Self::Min, Reducer::Min, 5),
        (Self::Prod, Reducer::Prod, 6),
        (Self::Sum, Reducer::Sum, 7),
        (Self::MeanOfSquares, Reducer::MeanOfSquares, 8),
    ];

    /// Get the index of a given `Reducer`.
    pub fn get_index_from_reducer(reducer: Reducer) -> usize {
        Self::REDUCER_MAP
            .iter()
            .find(|(_, original, _)| original == &reducer)
            .map(|(_, _, index)| *index)
            .expect("Invalid Reducer variant")
    }

    /// Get the `Reducer` corresponding to a given index.
    pub fn get_reducer_from_index(index: usize) -> Option<Reducer> {
        Self::REDUCER_MAP
            .iter()
            .find(|(_, _, idx)| *idx == index)
            .map(|(_, reducer, _)| *reducer)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum CustomDatumType {
    Binary,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

impl CustomDatumType {
    /// Map between `CustomDatumType`, `DatumType`, and their corresponding indices.
    const DATUM_MAP: &'static [(Self, DatumType, usize)] = &[
        (Self::Binary, DatumType::Bool, 0),
        (Self::U8, DatumType::U8, 1),
        (Self::U16, DatumType::U16, 2),
        (Self::U32, DatumType::U32, 3),
        (Self::U64, DatumType::U64, 4),
        (Self::I8, DatumType::I8, 5),
        (Self::I16, DatumType::I16, 6),
        (Self::I32, DatumType::I32, 7),
        (Self::I64, DatumType::I64, 8),
        (Self::F16, DatumType::F16, 9),
        (Self::F32, DatumType::F32, 10),
        (Self::F64, DatumType::F64, 11),
    ];

    /// Get the index of a given `DatumType`.
    pub fn get_index_from_datum_type(datum: DatumType) -> usize {
        Self::DATUM_MAP
            .iter()
            .find(|(_, original, _)| *original == datum)
            .map(|(_, _, index)| *index)
            .expect("Invalid DatumType variant")
    }

    /// Get the `DatumType` corresponding to a given index.
    pub fn get_datum_type_from_index(index: usize) -> Option<DatumType> {
        Self::DATUM_MAP
            .iter()
            .find(|(_, _, idx)| *idx == index)
            .map(|(_, datum, _)| *datum)
    }
}
