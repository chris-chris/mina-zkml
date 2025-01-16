use tract_onnx::prelude::*;
use tract_onnx::tract_core::ops::binary::BinMiniOp;
use tract_onnx::tract_core::ops::nn::Reducer;
use tract_onnx::tract_hir::internal::ElementWiseMiniOp;
use tract_onnx::tract_hir::ops::math::Rem;
use tract_onnx::tract_hir::ops::math::*;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CustomBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Rem,
    ShiftLeft,
    ShiftRight,
}

impl CustomBinOp {
    /// Map between names, indices, and `CustomBinOp` variants.
    pub const BIN_OP_MAP: &'static [(&'static str, &dyn BinMiniOp, usize)] = &[
        ("Add", &Add, 0),
        ("Sub", &Sub, 1),
        ("Mul", &Mul, 2),
        ("Div", &Div, 3),
        ("Pow", &Pow, 4),
        ("Max", &Max, 5),
        ("Min", &Min, 6),
        ("Rem", &Rem, 7),
        ("ShiftLeft", &ShiftLeft, 8),
        ("ShiftRight", &ShiftRight, 9),
    ];

    /// Get the index from a `BinMiniOp` by matching its name.
    pub fn get_index_from_op(op: &dyn BinMiniOp) -> Option<usize> {
        let op_name = op.name();
        Self::BIN_OP_MAP
            .iter()
            .find(|(name, _, _)| *name == op_name)
            .map(|(_, _, index)| *index)
    }

    /// Get the `CustomBinOp` from an index.
    pub fn get_op_from_index(index: &usize) -> Option<Box<dyn BinMiniOp>> {
        match index {
            0 => Some(Box::new(Add)),
            1 => Some(Box::new(Sub)),
            2 => Some(Box::new(Mul)),
            3 => Some(Box::new(Div)),
            4 => Some(Box::new(Pow)),
            5 => Some(Box::new(Max)),
            6 => Some(Box::new(Min)),
            7 => Some(Box::new(Rem)),
            8 => Some(Box::new(ShiftLeft)),
            9 => Some(Box::new(ShiftRight)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CustomElementWiseOp {
    Abs,
    Exp,
    Ln,
    Square,
    Sqrt,
    Recip,
    Rsqrt,
    Ceil,
    Floor,
    Round,
    Cos,
    Sin,
    Tan,
    Acos,
    Asin,
    Atan,
    Cosh,
    Sinh,
    Tanh,
    Erf,
    Atanh,
    Acosh,
    Asinh,
    Neg,
    Sign,
}

impl CustomElementWiseOp {
    /// Map between names, actual Tract operations, and indices for `CustomElementWiseOp`.
    pub const ELEMENTWISE_OP_MAP: &'static [(
        &'static str,
        fn() -> Box<dyn ElementWiseMiniOp>,
        usize,
    )] = &[
        ("Abs", || Box::new(Abs {}), 0),
        ("Exp", || Box::new(Exp {}), 1),
        ("Ln", || Box::new(Ln {}), 2),
        ("Square", || Box::new(Square {}), 3),
        ("Sqrt", || Box::new(Sqrt {}), 4),
        ("Recip", || Box::new(Recip {}), 5),
        ("Rsqrt", || Box::new(Rsqrt {}), 6),
        ("Ceil", || Box::new(Ceil {}), 7),
        ("Floor", || Box::new(Floor {}), 8),
        ("Round", || Box::new(Round {}), 9),
        ("Cos", || Box::new(Cos {}), 10),
        ("Sin", || Box::new(Sin {}), 11),
        ("Tan", || Box::new(Tan {}), 12),
        ("Acos", || Box::new(Acos {}), 13),
        ("Asin", || Box::new(Asin {}), 14),
        ("Atan", || Box::new(Atan {}), 15),
        ("Cosh", || Box::new(Cosh {}), 16),
        ("Sinh", || Box::new(Sinh {}), 17),
        ("Tanh", || Box::new(Tanh {}), 18),
        ("Erf", || Box::new(Erf {}), 19),
        ("Atanh", || Box::new(Atanh {}), 20),
        ("Acosh", || Box::new(Acosh {}), 21),
        ("Asinh", || Box::new(Asinh {}), 22),
        ("Neg", || Box::new(Neg {}), 23),
        ("Sign", || Box::new(Sign {}), 24),
    ];

    /// Get the index from an `ElementWiseMiniOp` by matching its name.
    pub fn get_index_from_op(op: &dyn ElementWiseMiniOp) -> Option<usize> {
        let op_name = op.name();
        Self::ELEMENTWISE_OP_MAP
            .iter()
            .find(|(name, _, _)| *name == op_name)
            .map(|(_, _, index)| *index)
    }

    /// Get the `CustomElementWiseOp` from an index.
    pub fn get_op_from_index(index: &usize) -> Option<Box<dyn ElementWiseMiniOp>> {
        Self::ELEMENTWISE_OP_MAP
            .iter()
            .find(|(_, _, i)| i == index)
            .map(|(_, op_fn, _)| op_fn())
    }
}
