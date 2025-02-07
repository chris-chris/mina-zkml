# Roadmap

## High Priority Operators (P0)

| Operator | Type | Estimated Time |
|----------|------|----------------|
| DeConvPoly | Deconvolution | 2 weeks |
| IdentityPoly | Identity Operation | 2 days |
| LeakyReLUPoly | Activation Function | 1 week |
| ConcatPoly | Tensor Concatenation | 1 week |

Note: Time estimates consider implementation, testing, and optimization phases.

## Priority 1 Operators (P1)

| Operator | Type | Estimated Time |
|----------|------|----------------|
| HardSwishLookup | Activation Function | 1 week |
| GatherElementsPoly | Array Operation | 2 weeks |
| GatherNDPoly | Array Operation | 2 weeks |
| MoveAxisPoly | Array Operation | 1 week |
| FlattenPoly | Array Operation | 3 days |
| PadPoly | Array Operation | 1 week |
| SlicePoly | Array Operation | 1 week |
| ResizePoly | Array Operation | 1 week |
| TopKHybrid | Search Operation | 2 weeks |
| OneHotHybrid | Array Operation | 1 week |

Note: Complexity varies based on operation type and optimization requirements.

## Model Execution
- [ ] Implement `Tensor` interface, with better optimized execution code. Either from the `tract::Tensor` or `ndarray` crate, for interface, and operations from `tract::linalg` or `ndarray` crate.

## GPU Acceleration

Key Focus Areas:
1. **Basic Operations**
   - Possible CUDA integration for tensor operations.
   - Use ICICLE for GPU acceleration with cryptographic operations.
   - Estimated Performance Gain: 10-50x

2. **Advanced Operations**
   - GPU-optimized convolution algorithms
   - Batched processing support
   - Memory optimization strategies

## zk Optimization
- [ ] Implement, `Lookup Table`, and other optimizations mentioned in the `mina-book/mina-zkml` book, for the prover.
- [ ] Simplify the current `Operator` logic, ensure the operations and their corresponding `Circuits` are optimized.
