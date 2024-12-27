use crate::graph::model::{Model, ParsedNodes, VarVisibility, Visibility};
use crate::graph::utilities::*;
use std::collections::{BTreeMap, HashMap};

#[test]
fn test_conv_simple() {
    let mut nodes = BTreeMap::new();

    // Input node (id: 0)
    nodes.insert(0, create_input_node(0, vec![1, 1, 4, 4]));

    // Weight node (id: 1)
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![1, 1, 3, 3],
            vec![
                1.0, 0.0, -1.0, // Kernel row 1
                1.0, 0.0, -1.0, // Kernel row 2
                1.0, 0.0, -1.0, // Kernel row 3
            ],
        ),
    );

    // Bias node (id: 2)
    nodes.insert(2, create_const_node(2, vec![1], vec![0.0])); // No bias

    // Conv node (id: 3)
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![1, 1]);
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]);
    attributes.insert("dilations".to_string(), vec![1, 1]);

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 1, 2, 2],
            attributes,
        ),
    );

    // Graph setup
    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensor
    let input_tensor = vec![
        1.0, 2.0, 3.0, 4.0, // Row 1
        5.0, 6.0, 7.0, 8.0, // Row 2
        9.0, 10.0, 11.0, 12.0, // Row 3
        13.0, 14.0, 15.0, 16.0, // Row 4
    ];

    // Execute the graph
    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output
    let expected_output = vec![
        -6.0, -6.0, // Row 1
        -6.0, -6.0, // Row 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_conv_stride() {
    let mut nodes = BTreeMap::new();

    nodes.insert(0, create_input_node(0, vec![1, 1, 5, 5]));
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![1, 1, 3, 3],
            vec![
                1.0, 0.0, -1.0, // Kernel row 1
                1.0, 0.0, -1.0, // Kernel row 2
                1.0, 0.0, -1.0, // Kernel row 3
            ],
        ),
    );
    nodes.insert(2, create_const_node(2, vec![1], vec![0.0])); // No bias

    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![2, 2]);
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]);
    attributes.insert("dilations".to_string(), vec![1, 1]);

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 1, 2, 2],
            attributes,
        ),
    );

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    let input_tensor = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, // Row 1
        6.0, 7.0, 8.0, 9.0, 10.0, // Row 2
        11.0, 12.0, 13.0, 14.0, 15.0, // Row 3
        16.0, 17.0, 18.0, 19.0, 20.0, // Row 4
        21.0, 22.0, 23.0, 24.0, 25.0, // Row 5
    ];

    let result = model.graph.execute(&[input_tensor]).unwrap();

    let expected_output = vec![
        -6.0, -6.0, // Row 1
        -6.0, -6.0, // Row 2
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_conv_multiple_output_channels() {
    let mut nodes = BTreeMap::new();

    nodes.insert(0, create_input_node(0, vec![1, 1, 4, 4]));
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![2, 1, 3, 3],
            vec![
                1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, // Kernel for output channel 0
                -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, // Kernel for output channel 1
            ],
        ),
    );
    nodes.insert(2, create_const_node(2, vec![2], vec![0.0, 1.0])); // Bias for 2 channels

    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![1, 1]);
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]);
    attributes.insert("dilations".to_string(), vec![1, 1]);

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 2, 2, 2],
            attributes,
        ),
    );

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    let input_tensor = vec![
        1.0, 2.0, 3.0, 4.0, //
        5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, //
        13.0, 14.0, 15.0, 16.0, //
    ];

    let result = model.graph.execute(&[input_tensor]).unwrap();

    let expected_output = vec![-6.0, -6.0, -6.0, -6.0, 4.0, 4.0, 4.0, 4.0];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_conv_with_padding() {
    let mut nodes = BTreeMap::new();

    // Input tensor with shape [1, 1, 3, 3]
    nodes.insert(0, create_input_node(0, vec![1, 1, 3, 3]));
    // Kernel with shape [1, 1, 3, 3]
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![1, 1, 3, 3],
            vec![
                1.0, 0.0, -1.0, //
                0.5, 0.0, -0.5, //
                1.0, 0.0, -1.0, //
            ],
        ),
    );
    // Bias
    nodes.insert(2, create_const_node(2, vec![1], vec![1.0]));

    // Conv node attributes
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![1, 1]);
    attributes.insert("padding".to_string(), vec![1, 1, 1, 1]); // Padding of 1 on all sides
    attributes.insert("dilations".to_string(), vec![1, 1]);

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 1, 3, 3], // Output shape matches input due to padding
            attributes,
        ),
    );

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensor
    let input_tensor = vec![
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, //
    ];

    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output with padding included
    let expected_output = vec![-5.0, -2.0, 7.0, -11.5, -4.0, 13.5, -8.0, -2.0, 10.0];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_conv_with_dilation() {
    let mut nodes = BTreeMap::new();

    // Input tensor with shape [1, 1, 5, 5]
    nodes.insert(0, create_input_node(0, vec![1, 1, 5, 5]));
    // Kernel with shape [1, 1, 3, 3]
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![1, 1, 3, 3],
            vec![
                1.0, 0.0, -1.0, //
                0.5, 0.0, -0.5, //
                1.0, 0.0, -1.0, //
            ],
        ),
    );
    // Bias
    nodes.insert(2, create_const_node(2, vec![1], vec![0.0]));

    // Conv node attributes
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![1, 1]);
    attributes.insert("padding".to_string(), vec![0, 0, 0, 0]); // No padding
    attributes.insert("dilations".to_string(), vec![2, 2]); // Dilation factor of 2

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 1, 1, 1], // Smaller output due to dilation
            attributes,
        ),
    );

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Input tensor
    let input_tensor = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, //
        6.0, 7.0, 8.0, 9.0, 10.0, //
        11.0, 12.0, 13.0, 14.0, 15.0, //
        16.0, 17.0, 18.0, 19.0, 20.0, //
        21.0, 22.0, 23.0, 24.0, 25.0, //
    ];

    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Expected output with dilation
    let expected_output = vec![
        -10.0, //
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}

#[test]
fn test_conv_complex_case() {
    let mut nodes = BTreeMap::new();

    // Input tensor with shape [1, 2, 6, 6] (Batch size: 1, Channels: 2, Height: 6, Width: 6)
    nodes.insert(0, create_input_node(0, vec![1, 2, 6, 6]));

    // Kernel with shape [3, 2, 3, 3] (Output Channels: 3, Input Channels: 2, Height: 3, Width: 3)
    nodes.insert(
        1,
        create_const_node(
            1,
            vec![3, 2, 3, 3],
            vec![
                // Kernel for output channel 0
                1.0, -1.0, 0.5, 0.2, -0.2, 0.3, 0.4, -0.5, 0.6, // Channel 0
                -0.1, 0.9, -0.3, 0.7, -0.8, 0.1, -0.6, 0.4, -0.2, // Channel 1
                // Kernel for output channel 1
                0.2, -0.4, 0.6, -0.8, 0.1, -0.3, 0.9, 0.3, -0.2, // Channel 0
                0.5, -0.7, 0.8, 0.4, -0.9, 0.2, 0.6, -0.1, 0.3, // Channel 1
                // Kernel for output channel 2
                0.4, 0.5, -0.6, -0.2, 0.7, 0.8, -0.3, 0.9, -0.1, // Channel 0
                -0.5, 0.3, 0.2, 0.1, -0.4, 0.6, -0.8, 0.7, 0.4, // Channel 1
            ],
        ),
    );

    // Bias for each output channel
    nodes.insert(2, create_const_node(2, vec![3], vec![1.0, -1.0, 0.5]));

    // Conv node attributes
    let mut attributes = HashMap::new();
    attributes.insert("kernel_shape".to_string(), vec![3, 3]);
    attributes.insert("strides".to_string(), vec![2, 2]); // Stride of 2
    attributes.insert("padding".to_string(), vec![1, 1, 1, 1]); // Padding of 1 on all sides
    attributes.insert("dilations".to_string(), vec![2, 2]); // Dilation factor of 2

    nodes.insert(
        3,
        create_conv_node(
            3,
            vec![(0, 0), (1, 0), (2, 0)],
            vec![1, 3, 2, 2], // Output shape
            attributes,
        ),
    );

    let graph = ParsedNodes {
        nodes,
        inputs: vec![0],
        outputs: vec![(3, 0)],
    };

    let model = Model {
        graph,
        visibility: VarVisibility {
            input: Visibility::Public,
            output: Visibility::Public,
        },
    };

    // Randomized input tensor
    let input_tensor = vec![
        // Channel 0
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0, //
        7.0, -8.0, 9.0, -10.0, 11.0, -12.0, //
        -13.0, 14.0, -15.0, 16.0, -17.0, 18.0, //
        19.0, -20.0, 21.0, -22.0, 23.0, -24.0, //
        -25.0, 26.0, -27.0, 28.0, -29.0, 30.0, //
        31.0, -32.0, 33.0, -34.0, 35.0, -36.0, //
        // Channel 1
        -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, //
        -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, //
        13.0, -14.0, 15.0, -16.0, 17.0, -18.0, //
        -19.0, 20.0, -21.0, 22.0, -23.0, 24.0, //
        25.0, -26.0, 27.0, -28.0, 29.0, -30.0, //
        -31.0, 32.0, -33.0, 34.0, -35.0, 36.0, //
    ];

    let result = model.graph.execute(&[input_tensor]).unwrap();

    // Manually calculated expected output
    let expected_output = vec![
        -5.4,
        -22.800003,
        -6.6000023,
        -36.6,
        -0.99999905,
        1.7999997,
        -6.2,
        9.600004,
        -3.2999983,
        -12.900002,
        -8.899998,
        -26.899998,
    ];

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], expected_output);
}
