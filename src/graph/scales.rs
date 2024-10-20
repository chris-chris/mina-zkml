// src/lib.rs

use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale(i32);

impl Scale {
    pub fn new(scale: i32) -> Self {
        Scale(scale)
    }

    pub fn value(&self) -> i32 {
        self.0
    }

    pub fn to_multiplier(&self) -> f64 {
        2f64.powi(self.0)
    }

    pub fn from_multiplier(multiplier: f64) -> Self {
        Scale((multiplier.log2()) as i32)
    }
}

impl Add for Scale {
    type Output = Scale;

    fn add(self, other: Scale) -> Scale {
        Scale(self.0 + other.0)
    }
}

impl Sub for Scale {
    type Output = Scale;

    fn sub(self, other: Scale) -> Scale {
        Scale(self.0 - other.0)
    }
}

impl Mul<i32> for Scale {
    type Output = Scale;

    fn mul(self, rhs: i32) -> Scale {
        Scale(self.0 * rhs)
    }
}

pub fn scale_to_multiplier(scale: Scale) -> f64 {
    scale.to_multiplier()
}

pub fn multiplier_to_scale(multiplier: f64) -> Scale {
    Scale::from_multiplier(multiplier)
}

#[derive(Debug, Clone)]
pub struct VarScales {
    pub input: Scale,
    pub params: Scale,
    pub output: Scale,
    pub rebase_multiplier: f64,
}

impl VarScales {
    pub fn new(input: Scale, params: Scale, output: Scale, rebase_multiplier: f64) -> Self {
        VarScales {
            input,
            params,
            output,
            rebase_multiplier,
        }
    }

    pub fn input_multiplier(&self) -> f64 {
        self.input.to_multiplier()
    }

    pub fn params_multiplier(&self) -> f64 {
        self.params.to_multiplier()
    }

    pub fn output_multiplier(&self) -> f64 {
        self.output.to_multiplier()
    }

    pub fn rebase(&self, value: f64) -> f64 {
        value * self.rebase_multiplier
    }

    pub fn unrebase(&self, value: f64) -> f64 {
        value / self.rebase_multiplier
    }
}

pub fn from_multipliers(input: f64, params: f64, output: f64, rebase_multiplier: f64) -> VarScales {
    VarScales::new(
        multiplier_to_scale(input),
        multiplier_to_scale(params),
        multiplier_to_scale(output),
        rebase_multiplier,
    )
}