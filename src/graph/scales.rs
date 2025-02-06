//! This library provides utilities for working with scaling factors and their conversions, including
//! mathematical operations on scales, as well as a helper structure for managing input, parameter,
//! and output scales with a rebase multiplier.
//!
//! # Overview
//!
//! The library introduces two main components:
//! - [`Scale`]: Represents a scaling factor as an integer, with utility methods for conversion and arithmetic operations.
//! - [`VarScales`]: A structure for managing input, parameter, and output scales, as well as rebasing values based on a multiplier.

use std::ops::{Add, Mul, Sub};

/// Represents a scaling factor as an integer, supporting arithmetic operations and conversions.
///
/// The `Scale` struct encapsulates an integer value representing a power of 2 scale factor.
/// It provides methods for converting between the integer representation and a multiplier.
///
/// # Panics
/// - None: All methods are designed to be safe.
///
/// # Errors
/// - None: This struct does not perform any fallible operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale(i32);

impl Scale {
    /// Creates a new `Scale` with the given integer value.
    pub fn new(scale: i32) -> Self {
        Scale(scale)
    }

    /// Returns the integer value of the scale.
    pub fn value(&self) -> i32 {
        self.0
    }

    /// Converts the scale to a multiplier (2^scale).
    pub fn to_multiplier(&self) -> f64 {
        2f64.powi(self.0)
    }

    /// Creates a `Scale` from a multiplier value.
    pub fn from_multiplier(multiplier: f64) -> Self {
        Scale((multiplier.log2()) as i32)
    }
}

impl Add for Scale {
    type Output = Scale;

    /// Adds two scales, resulting in a new `Scale`.
    fn add(self, other: Scale) -> Scale {
        Scale(self.0 + other.0)
    }
}

impl Sub for Scale {
    type Output = Scale;

    /// Subtracts one scale from another.
    fn sub(self, other: Scale) -> Scale {
        Scale(self.0 - other.0)
    }
}

impl Mul<i32> for Scale {
    type Output = Scale;

    /// Multiplies a scale by an integer factor.
    fn mul(self, rhs: i32) -> Scale {
        Scale(self.0 * rhs)
    }
}

/// Converts a `Scale` to its multiplier value.
pub fn scale_to_multiplier(scale: Scale) -> f64 {
    scale.to_multiplier()
}

/// Converts a multiplier value to a `Scale`.
pub fn multiplier_to_scale(multiplier: f64) -> Scale {
    Scale::from_multiplier(multiplier)
}

/// A structure to manage input, parameter, and output scales with a rebase multiplier.
///
/// The `VarScales` structure provides methods to retrieve the multiplier values for input,
/// parameter, and output scales, as well as to rebase and unrebase values.
#[derive(Debug, Clone)]
pub struct VarScales {
    pub input: Scale,
    pub params: Scale,
    pub output: Scale,
    pub rebase_multiplier: f64,
}

impl VarScales {
    /// Creates a new `VarScales` instance.
    pub fn new(input: Scale, params: Scale, output: Scale, rebase_multiplier: f64) -> Self {
        VarScales {
            input,
            params,
            output,
            rebase_multiplier,
        }
    }

    /// Returns the multiplier for the input scale.
    pub fn input_multiplier(&self) -> f64 {
        self.input.to_multiplier()
    }

    /// Returns the multiplier for the parameter scale.
    pub fn params_multiplier(&self) -> f64 {
        self.params.to_multiplier()
    }

    /// Returns the multiplier for the output scale.
    pub fn output_multiplier(&self) -> f64 {
        self.output.to_multiplier()
    }

    /// Rebases a value using the rebase multiplier.
    pub fn rebase(&self, value: f64) -> f64 {
        value * self.rebase_multiplier
    }

    /// Restores a rebased value to its original scale.
    pub fn unrebase(&self, value: f64) -> f64 {
        value / self.rebase_multiplier
    }
}

/// Creates a `VarScales` instance from multiplier values.
pub fn from_multipliers(input: f64, params: f64, output: f64, rebase_multiplier: f64) -> VarScales {
    VarScales::new(
        multiplier_to_scale(input),
        multiplier_to_scale(params),
        multiplier_to_scale(output),
        rebase_multiplier,
    )
}
