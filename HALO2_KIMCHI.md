# halo2_proofs

## halo2_proofs::poly

```
use halo2_proofs::poly::commitment::CommitmentScheme;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::poly::commitment::{CommitmentScheme, Params};
use halo2_proofs::poly::commitment::{ParamsProver, Verifier};
use halo2_proofs::poly::ipa::commitment::{IPACommitmentScheme, ParamsIPA};
use halo2_proofs::poly::ipa::multiopen::{ProverIPA, VerifierIPA};
use halo2_proofs::poly::ipa::strategy::AccumulatorStrategy as IPAAccumulatorStrategy;
use halo2_proofs::poly::ipa::strategy::SingleStrategy as IPASingleStrategy;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy as KZGAccumulatorStrategy;
use halo2_proofs::poly::kzg::{
commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::poly::commitment::{Blind, CommitmentScheme, Params};
```

## halo2_proofs::circuit

```
use halo2_proofs::circuit::Layouter;
use halo2_proofs::circuit::SimpleFloorPlanner;
use halo2_proofs::circuit::Value;
use halo2_proofs::{circuit::\*, plonk::\_};
```

## halo2_proofs::dev;

```
use halo2_proofs::dev::MockProver;
use halo2_proofs::dev::VerifyFailure;
```

## halo2_proofs::plonk

```
use halo2_proofs::plonk::{Circuit, ConstraintSystem, Error as PlonkError};
use halo2_proofs::plonk::{self, Circuit};
use halo2_proofs::plonk::VerifyingKey;
```

## halo2_proofs::arithmetic

```
use halo2_proofs::arithmetic::Field;
```

## halo2_proofs::transcript

```
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer};
```

# halo2curves

## halo2curves

```
use halo2curves::CurveAffine;
```

## use halo2curves::bn256

```
use halo2curves::bn256::{self, Fr as Fp, G1Affine};
use halo2curves::bn256::{Bn256, Fr};
use halo2curves::bn256::G1Affine;
```

## halo2curves::group

```
use halo2curves::group::ff::PrimeField;
use halo2curves::group::prime::PrimeCurveAffine;
use halo2curves::group::Curve;
```

## halo2curves::ff

```
use halo2curves::ff::{Field, PrimeField};
use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
```

## halo2curves::ff::Field

The Field trait defines a fundamental interface for working with elements of a finite field. A finite field is a mathematical structure consisting of a finite set of elements where addition, subtraction, multiplication, and division (except by zero) are defined. This trait encapsulates the essential operations and properties of finite field elements, allowing for consistent behavior across various finite field implementations.

### Key Features and Properties:

- Basic Operations Support: Implements addition (Add), subtraction (Sub), multiplication (Mul), division (via multiplicative inverse invert), and negation (Neg) operations.
- Constant Elements: Defines the additive identity (ZERO) and multiplicative identity (ONE).
- Random Element Generation: Allows generating random field elements using a random number generator (RngCore).
- Special Operations: Provides methods for squaring (square), cubing (cube), computing inverses (invert), calculating square roots (sqrt), and exponentiation (pow).
- Security: Supports constant-time operations to enhance security, preventing side-channel attacks. For example, the is_zero method operates in constant time to avoid timing attacks.

### Usage:

This trait is essential in zero-knowledge proof systems for performing mathematical operations on finite field elements. By abstracting these operations, it allows various field implementations to be used seamlessly within proving systems like Halo2. It ensures that the mathematical foundations required for cryptographic protocols are both robust and flexible.

## halo2curves::serde

```
use halo2curves::serde::SerdeObject;
```

# halo2_solidity_verifier

```
use halo2_solidity_verifier::encode_calldata; #[cfg(not(target_arch = "wasm32"))]
use halo2_solidity_verifier;
```

# snark_verifier

```
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use snark_verifier::system::halo2::compile;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use snark_verifier::system::halo2::Config;
```
