//! This library provides utilities for working with scaling factors and their conversions, including
//! mathematical operations on scales, as well as a helper structure for managing input, parameter,
//! and output scales with a rebase multiplier.
//!
//! # Overview
//!
//! The library introduces two main components:
//! - [`Scale`]: Represents a scaling factor as an integer, with utility methods for conversion and arithmetic operations.
//! - [`VarScales`]: A structure for managing input, parameter, and output scales, as well as rebasing values based on a multiplier.
//!
//! # Examples
//! Basic usage of the `Scale` type and the `VarScales` structure can be seen below.

use std::ops::{Add, Mul, Sub};

/// Represents a scaling factor as an integer, supporting arithmetic operations and conversions.
///
/// The `Scale` struct encapsulates an integer value representing a power of 2 scale factor.
/// It provides methods for converting between the integer representation and a multiplier.
///
/// # Examples
/// ```rust
/// use your_crate_name::Scale;
///
/// let scale = Scale::new(3);
/// assert_eq!(scale.value(), 3);
/// assert_eq!(scale.to_multiplier(), 8.0); // 2^3 = 8.0
///
/// let scale_from_multiplier = Scale::from_multiplier(8.0);
/// assert_eq!(scale_from_multiplier.value(), 3);
/// ```
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
    ///
    /// # Examples
    /// ```rust
    /// use your_crate_name::Scale;
    ///
    /// let scale = Scale::from_multiplier(16.0);
    /// assert_eq!(scale.value(), 4); // 2^4 = 16
    /// ```
    pub fn from_multiplier(multiplier: f64) -> Self {
        Scale((multiplier.log2()) as i32)
    }
}

impl Add for Scale {
    type Output = Scale;

    /// Adds two scales, resulting in a new `Scale`.
    ///
    /// # Examples
    /// ```rust
    /// use your_crate_name::Scale;
    ///
    /// let scale1 = Scale::new(2);
    /// let scale2 = Scale::new(3);
    /// assert_eq!((scale1 + scale2).value(), 5);
    /// ```
    fn add(self, other: Scale) -> Scale {
        Scale(self.0 + other.0)
    }
}

impl Sub for Scale {
    type Output = Scale;

    /// Subtracts one scale from another.
    ///
    /// # Examples
    /// ```rust
    /// use your_crate_name::Scale;
    ///
    /// let scale1 = Scale::new(5);
    /// let scale2 = Scale::new(3);
    /// assert_eq!((scale1 - scale2).value(), 2);
    /// ```
    fn sub(self, other: Scale) -> Scale {
        Scale(self.0 - other.0)
    }
}

impl Mul<i32> for Scale {
    type Output = Scale;

    /// Multiplies a scale by an integer factor.
    ///
    /// # Examples
    /// ```rust
    /// use your_crate_name::Scale;
    ///
    /// let scale = Scale::new(2);
    /// assert_eq!((scale * 3).value(), 6);
    /// ```
    fn mul(self, rhs: i32) -> Scale {
        Scale(self.0 * rhs)
    }
}

/// Converts a `Scale` to its multiplier value.
///
/// # Examples
/// ```rust
/// use your_crate_name::{Scale, scale_to_multiplier};
///
/// let scale = Scale::new(3);
/// assert_eq!(scale_to_multiplier(scale), 8.0); // 2^3 = 8
/// ```
pub fn scale_to_multiplier(scale: Scale) -> f64 {
    scale.to_multiplier()
}

/// Converts a multiplier value to a `Scale`.
///
/// # Examples
/// ```rust
/// use your_crate_name::{Scale, multiplier_to_scale};
///
/// let scale = multiplier_to_scale(8.0);
/// assert_eq!(scale.value(), 3);
/// ```
pub fn multiplier_to_scale(multiplier: f64) -> Scale {
    Scale::from_multiplier(multiplier)
}

/// A structure to manage input, parameter, and output scales with a rebase multiplier.
///
/// The `VarScales` structure provides methods to retrieve the multiplier values for input,
/// parameter, and output scales, as well as to rebase and unrebase values.
///
/// # Examples
/// ```rust
/// use your_crate_name::{Scale, VarScales};
///
/// let scales = VarScales::new(
///     Scale::new(2),
///     Scale::new(3),
///     Scale::new(4),
///     0.5,
/// );
/// assert_eq!(scales.input_multiplier(), 4.0); // 2^2 = 4
/// assert_eq!(scales.rebase(10.0), 5.0); // 10 * 0.5
/// ```
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
    ///
    /// # Examples
    /// ```rust
    /// use your_crate_name::VarScales;
    ///
    /// let scales = VarScales::new(
    ///     Scale::new(1),
    ///     Scale::new(2),
    ///     Scale::new(3),
    ///     0.5,
    /// );
    /// assert_eq!(scales.rebase(8.0), 4.0); // 8 * 0.5
    /// ```
    pub fn rebase(&self, value: f64) -> f64 {
        value * self.rebase_multiplier
    }

    /// Restores a rebased value to its original scale.
    ///
    /// # Examples
    /// ```rust
    /// use your_crate_name::VarScales;
    ///
    /// let scales = VarScales::new(
    ///     Scale::new(1),
    ///     Scale::new(2),
    ///     Scale::new(3),
    ///     0.5,
    /// );
    /// assert_eq!(scales.unrebase(4.0), 8.0); // 4 / 0.5
    /// ```
    pub fn unrebase(&self, value: f64) -> f64 {
        value / self.rebase_multiplier
    }
}

/// Creates a `VarScales` instance from multiplier values.
///
/// # Examples
/// ```rust
/// use your_crate_name::{Scale, VarScales, from_multipliers};
///
/// let scales = from_multipliers(4.0, 8.0, 16.0, 0.5);
/// assert_eq!(scales.input.value(), 2); // 2^2 = 4
/// assert_eq!(scales.rebase(10.0), 5.0);
/// ```
pub fn from_multipliers(input: f64, params: f64, output: f64, rebase_multiplier: f64) -> VarScales {
    VarScales::new(
        multiplier_to_scale(input),
        multiplier_to_scale(params),
        multiplier_to_scale(output),
        rebase_multiplier,
    )
}
