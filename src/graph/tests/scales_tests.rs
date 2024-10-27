#[cfg(test)]
mod tests {
    use super::super::scales::*;
    use std::f64::EPSILON;

    #[test]
    fn test_scale_creation() {
        let scale = Scale::new(2);
        assert_eq!(scale.value(), 2);
    }

    #[test]
    fn test_scale_arithmetic() {
        let scale1 = Scale::new(2);
        let scale2 = Scale::new(3);

        // Test addition
        let sum = scale1 + scale2;
        assert_eq!(sum.value(), 5);

        // Test subtraction
        let diff = scale2 - scale1;
        assert_eq!(diff.value(), 1);

        // Test multiplication
        let mult = scale1 * 3;
        assert_eq!(mult.value(), 6);
    }

    #[test]
    fn test_scale_multiplier_conversion() {
        let scale = Scale::new(2);
        let multiplier = scale.to_multiplier();
        assert!((multiplier - 4.0).abs() < EPSILON);

        let scale_back = Scale::from_multiplier(multiplier);
        assert_eq!(scale, scale_back);
    }

    #[test]
    fn test_var_scales() {
        let input_scale = Scale::new(2);
        let params_scale = Scale::new(3);
        let output_scale = Scale::new(4);
        let rebase_multiplier = 2.0;

        let var_scales = VarScales::new(
            input_scale,
            params_scale,
            output_scale,
            rebase_multiplier,
        );

        // Test multipliers
        assert!((var_scales.input_multiplier() - 4.0).abs() < EPSILON);
        assert!((var_scales.params_multiplier() - 8.0).abs() < EPSILON);
        assert!((var_scales.output_multiplier() - 16.0).abs() < EPSILON);

        // Test rebase operations
        let value = 10.0;
        let rebased = var_scales.rebase(value);
        assert!((rebased - 20.0).abs() < EPSILON);
        
        let unrebased = var_scales.unrebase(rebased);
        assert!((unrebased - value).abs() < EPSILON);
    }

    #[test]
    fn test_from_multipliers() {
        let input_mult = 4.0;
        let params_mult = 8.0;
        let output_mult = 16.0;
        let rebase_mult = 2.0;

        let var_scales = from_multipliers(
            input_mult,
            params_mult,
            output_mult,
            rebase_mult,
        );

        assert_eq!(var_scales.input.value(), 2);
        assert_eq!(var_scales.params.value(), 3);
        assert_eq!(var_scales.output.value(), 4);
        assert!((var_scales.rebase_multiplier - rebase_mult).abs() < EPSILON);
    }
}
