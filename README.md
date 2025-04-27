# type-generic-umfpack

A high-level type-generic Rust wrapper for the UMFPACK sparse linear system solver. 

## Overview

`type-generic-umfpack` provides a safe and ergonomic Rust interface to the UMFPACK library for solving unsymmetric sparse linear systems using the Unsymmetric MultiFrontal method.

## Features

- Support for both real and complex sparse matrices
- Automatic handling of symbolic and numeric factorization
- Compatible with Compressed Sparse Column (CSC) format
- Simple interface for solving linear systems of the form Ax = b
- Full control over UMFPACK parameters
- Support for various system types (A, A^T, etc.)

## Installation

Add `type-generic-umfpack` to your `Cargo.toml`:

```toml
[dependencies]
type-generic-umfpack = "0.1.0"
```

### Dependencies

This crate depends on the `umfpack-rs` crate, which requires:

- UMFPACK library (part of SuiteSparse)
- BLAS implementation

#### On macOS:

Install OpenBLAS via Homebrew:

```bash
brew install openblas
```

The build script will automatically use the Accelerate framework available on macOS.

#### On Linux:

Install the required libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libsuitesparse-dev libopenblas-dev

# Fedora
sudo dnf install suitesparse-devel openblas-devel
```

## Usage

### Basic Example

```rust
use type-generic-umfpack::UMFPackMatrix;
use type-generic-umfpack::UMFPACK;

// Create a sparse matrix in CSC format:
// [2 3 0]
// [0 1 4]
// [5 0 6]
let n = 3; // rows
let m = 3; // columns
let ap = vec![0, 2, 4, 6]; // column pointers
let ai = vec![0, 2, 0, 1, 1, 2]; // row indices
let ax = vec![2.0, 5.0, 3.0, 1.0, 4.0, 6.0]; // values

// Create the matrix
let mut matrix = UMFPackMatrix::new(n, m, ap, ai, ax);

// Factorize the matrix
matrix.factorize();

// Solve Ax = b
let b = vec![7.0, 9.0, 11.0];
let x = matrix.solve_system(&b, UMFPACK::A).unwrap();

// x should be approximately [1.0, 1.0, 2.0]
println!("Solution: {:?}", x);
```

### Complex Matrices

```rust
use type-generic-umfpack::UMFPackMatrix;
use nalgebra::Complex;

// Create a complex sparse matrix
let n = 3;
let m = 3;
let ap = vec![0, 2, 4, 6];
let ai = vec![0, 2, 0, 1, 1, 2];
let ax = vec![
    Complex::new(2.0, 0.1), Complex::new(5.0, 0.2),
    Complex::new(3.0, 0.3), Complex::new(1.0, 0.4),
    Complex::new(4.0, 0.5), Complex::new(6.0, 0.6),
];

let mut matrix = UMFPackMatrix::new(n, m, ap, ai, ax);

// Solve with a complex right-hand side
let b = vec![
    Complex::new(7.0, 0.5),
    Complex::new(9.0, 0.6),
    Complex::new(11.0, 0.7),
];

let x = matrix.inverse_times(&b).unwrap();
println!("Complex solution: {:?}", x);
```

### Control Parameters

```rust
use type-generic-umfpack::{UMFPackMatrix, Control};

// Create a sparse matrix
let n = 3;
let m = 3;
let ap = vec![0, 2, 4, 6];
let ai = vec![0, 2, 0, 1, 1, 2];
let ax = vec![2.0, 5.0, 3.0, 1.0, 4.0, 6.0];

// Create custom control parameters
let mut control = Control::new();
// Set custom parameters here

// Create matrix with custom control
let matrix = UMFPackMatrix::new(n, m, ap, ai, ax)
    .with_control(control);
```

## API Reference

### Main Types

- `UMFPackMatrix<I, F>`: Main sparse matrix type that handles factorization and solving
- `Symbolic`: Holds the symbolic factorization
- `Numeric`: Holds the numeric factorization
- `Control`: Controls parameters for UMFPACK operations
- `Info`: Contains information about UMFPACK operations
- `UMFPACK`: Enum specifying the type of system to solve

### Common Methods

- `UMFPackMatrix::new()`: Create a new sparse matrix
- `UMFPackMatrix::factorize()`: Perform both symbolic and numeric factorization
- `UMFPackMatrix::solve_system()`: Solve a linear system with the factorized matrix
- `UMFPackMatrix::inverse_times()`: Compute A^(-1)b for a given vector b

## License

Codes in this repo is licensed under the GPL-3.0 License. This license applies only to the code in this repository and not to its dependencies.

## Acknowledgements

- [umfpack-rs](https://github.com/su-z/umfpack-rs) for the Rust bindings to UMFPACK
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) by Timothy A. Davis for the underlying UMFPACK implementation
- [nalgebra](https://nalgebra.org/) for complex number support
