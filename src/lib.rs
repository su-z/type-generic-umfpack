//! # type-generic-umfpack
//! 
//! A Rust wrapper for the UMFPACK sparse linear system solver.
//! This crate provides a high-level interface to the UMFPACK library for solving
//! sparse linear systems of equations Ax = b using the Unsymmetric MultiFrontal method.

use std::fmt::Debug;

use nalgebra::Complex;
use umfpack::{di::*, zi::*, *};
use concat_idents::concat_idents;
pub use symbolic::Symbolic;
pub use numeric::Numeric;
pub use control::Control;
pub use info::Info;
pub use sys::UMFPACK;
use num_traits::Zero;

/// Trait defining the interface for UMFPACK sparse linear system operations.
///
/// The implementations support both real and complex numeric types
/// with different integer index types.
pub trait UMFPack<I, F> {
    /// Performs symbolic analysis for a sparse matrix.
    ///
    /// # Arguments
    /// * `n` - Number of rows in the matrix
    /// * `m` - Number of columns in the matrix
    /// * `ap` - Column pointers array in CSC format
    /// * `ai` - Row indices array in CSC format
    /// * `ax` - Nonzero values array
    /// * `symbolic` - Output symbolic analysis data structure
    /// * `control` - Optional control parameters
    /// * `info` - Optional information output
    ///
    /// # Returns
    /// UMFPACK status code (0 for success)
    fn analyze_symbolic(n: I, m: I, ap: &[I], ai: &[I], ax: &[F], symbolic: &mut Symbolic, control: Option<&Control>, info: Option<&mut Info>) -> I;
    
    /// Performs numeric factorization for a sparse matrix.
    ///
    /// # Arguments
    /// * `ap` - Column pointers array in CSC format
    /// * `ai` - Row indices array in CSC format
    /// * `ax` - Nonzero values array
    /// * `symbolic` - Symbolic analysis from analyze_symbolic
    /// * `numeric` - Output numeric factorization data structure
    /// * `control` - Optional control parameters
    /// * `info` - Optional information output
    ///
    /// # Returns
    /// UMFPACK status code (0 for success)
    fn analyze_numeric(ap: &[I], ai: &[I], ax: &[F], symbolic: &Symbolic, numeric: &mut Numeric, control: Option<&Control>, info: Option<&mut Info>) -> I;
    
    /// Solves a sparse linear system using the factorization.
    ///
    /// # Arguments
    /// * `sys` - Type of system to solve (Ax=b, A'x=b, etc.)
    /// * `ap` - Column pointers array in CSC format
    /// * `ai` - Row indices array in CSC format
    /// * `ax` - Nonzero values array
    /// * `x` - Output solution vector
    /// * `b` - Right-hand side vector
    /// * `numeric` - Numeric factorization from analyze_numeric
    /// * `control` - Optional control parameters
    /// * `info` - Optional information output
    ///
    /// # Returns
    /// UMFPACK status code (0 for success)
    fn solve(sys: UMFPACK, ap: &[I], ai: &[I], ax: &[F], x: &mut [F], b: &[F], numeric: &Numeric, control: Option<&Control>, info: Option<&mut Info>) -> I;
}

/// Internal macro to create UMFPACK function identifiers
macro_rules! umf_func {
    ($n: ident, $n1: ident) => {
        concat_idents!(name = umfpack_,$n,_,$n1 {name})
    };
}

/// Internal macro to implement UMFPack trait for various numeric types
macro_rules! impl_umfpack {
    ($I: ty, $F: ty, $n: ident) => {
        impl UMFPack<$I, $F> for ($I, $F) {
            fn analyze_numeric(ap: &[$I], ai: &[$I], ax: &[$F], symbolic: &Symbolic, numeric: &mut Numeric, control: Option<&Control>, info: Option<&mut Info>) -> $I {
                let nnz = *ap.last().unwrap() as usize;
                assert_eq!(nnz, ai.len());
                assert_eq!(nnz, ax.len());
                umf_func!($n,numeric)(ap, ai, ax, symbolic, numeric, control, info)
            }
            fn analyze_symbolic(n: $I, m: $I, ap: &[$I], ai: &[$I], ax: &[$F], symbolic: &mut Symbolic, control: Option<&Control>, info: Option<&mut Info>) -> $I {
                let nnz = *ap.last().unwrap() as usize;
                assert_eq!(nnz, ai.len());
                assert_eq!(nnz, ax.len());
                umf_func!($n,symbolic)(n, m, ap, ai, ax, symbolic, control, info)
            }
            fn solve(sys: UMFPACK, ap: &[$I], ai: &[$I], ax: &[$F], x: &mut [$F], b: &[$F], numeric: &Numeric, control: Option<&Control>, info: Option<&mut Info>) -> $I {
                let nnz = *ap.last().unwrap() as usize;
                assert_eq!(nnz, ai.len());
                assert_eq!(nnz, ax.len());
                umf_func!($n,solve)(sys, ap, ai, ax, x, b, numeric, control, info)
            }
        }
    };
}

// Implement for complex double precision and real double precision
impl_umfpack!(i32, Complex<f64>, zi);
impl_umfpack!(i32, f64, di);

/// Performs numeric factorization for a sparse matrix.
///
/// This is a convenience function that automatically dispatches to the appropriate
/// implementation based on the types of the input parameters.
///
/// # Arguments
/// * `ap` - Column pointers array in CSC format
/// * `ai` - Row indices array in CSC format
/// * `ax` - Nonzero values array
/// * `symbolic` - Symbolic analysis from analyze_symbolic
/// * `numeric` - Output numeric factorization data structure
/// * `control` - Optional control parameters
/// * `info` - Optional information output
///
/// # Returns
/// UMFPACK status code (0 for success)
pub fn analyze_numeric<I, F>(ap: &[I], ai: &[I], ax: &[F], symbolic: &Symbolic, numeric: &mut Numeric, control: Option<&Control>, info: Option<&mut Info>) -> I where (I,F): UMFPack<I, F> {
    <(I, F)>::analyze_numeric(ap, ai, ax, symbolic, numeric, control, info)
}

/// Performs symbolic analysis for a sparse matrix.
///
/// This is a convenience function that automatically dispatches to the appropriate
/// implementation based on the types of the input parameters.
///
/// # Arguments
/// * `n` - Number of rows in the matrix
/// * `m` - Number of columns in the matrix
/// * `ap` - Column pointers array in CSC format
/// * `ai` - Row indices array in CSC format
/// * `ax` - Nonzero values array
/// * `symbolic` - Output symbolic analysis data structure
/// * `control` - Optional control parameters
/// * `info` - Optional information output
///
/// # Returns
/// UMFPACK status code (0 for success)
pub fn analyze_symbolic<I, F>(n: I, m: I, ap: &[I], ai: &[I], ax: &[F], symbolic: &mut Symbolic, control: Option<&Control>, info: Option<&mut Info>) -> I where (I,F): UMFPack<I, F> {
    <(I,F)>::analyze_symbolic(n, m, ap, ai, ax, symbolic, control, info)
}

/// Solves a sparse linear system using the factorization.
///
/// This is a convenience function that automatically dispatches to the appropriate
/// implementation based on the types of the input parameters.
///
/// # Arguments
/// * `sys` - Type of system to solve (Ax=b, A'x=b, etc.)
/// * `ap` - Column pointers array in CSC format
/// * `ai` - Row indices array in CSC format
/// * `ax` - Nonzero values array
/// * `x` - Output solution vector
/// * `b` - Right-hand side vector
/// * `numeric` - Numeric factorization from analyze_numeric
/// * `control` - Optional control parameters
/// * `info` - Optional information output
///
/// # Returns
/// UMFPACK status code (0 for success)
pub fn solve<I, F>(sys: UMFPACK, ap: &[I], ai: &[I], ax: &[F], x: &mut [F], b: &[F], numeric: &Numeric, control: Option<&Control>, info: Option<&mut Info>) -> I where (I,F): UMFPack<I, F> {
    <(I, F)>::solve(sys, ap, ai, ax, x, b, numeric, control, info)
}

/// A sparse matrix data structure using UMFPACK for solving linear systems.
///
/// This structure stores a sparse matrix in Compressed Sparse Column (CSC) format
/// and provides methods to factorize and solve linear systems of the form Ax = b.
pub struct UMFPackMatrix<I, F>
where
    (I, F): UMFPack<I, F>,
{
    /// Number of rows in the matrix
    n: I,
    /// Number of columns in the matrix
    m: I,
    /// Column pointers array in CSC format
    ap: Vec<I>,
    /// Row indices array in CSC format
    ai: Vec<I>,
    /// Nonzero values array
    ax: Vec<F>,
    /// Symbolic factorization
    symbolic: Option<Symbolic>,
    /// Numeric factorization
    numeric: Option<Numeric>,
    /// Control parameters for UMFPACK
    control: Control,
}

impl<I, F> UMFPackMatrix<I, F>
where
    I: Copy + PartialEq + TryInto<usize> + Zero,
    F: Copy + Default,
    (I, F): UMFPack<I, F>,
    <I as TryInto<usize>>::Error: Debug
{
    /// Creates a new sparse matrix in CSC format.
    ///
    /// # Arguments
    /// * `n` - Number of rows
    /// * `m` - Number of columns
    /// * `ap` - Column pointers array (must have length m+1)
    /// * `ai` - Row indices array
    /// * `ax` - Nonzero values array
    ///
    /// # Returns
    /// A new UMFPackMatrix instance
    pub fn new(n: I, m: I, ap: Vec<I>, ai: Vec<I>, ax: Vec<F>) -> Self {
        let last_col_ptr = *ap.last().unwrap();
        assert_eq!(ap.len(), m.try_into().unwrap() + 1);
        assert_eq!(ai.len(), last_col_ptr.try_into().unwrap());
        assert_eq!(ax.len(), last_col_ptr.try_into().unwrap());

        Self {
            n,
            m,
            ap,
            ai,
            ax,
            symbolic: None,
            numeric: None,
            control: Control::new(),
        }
    }

    /// Sets custom control parameters for UMFPACK operations.
    ///
    /// # Arguments
    /// * `control` - UMFPACK control parameters
    pub fn with_control(mut self, control: Control) -> Self {
        self.control = control;
        self
    }

    /// Performs symbolic factorization of the matrix.
    ///
    /// This is the first step in solving a linear system with UMFPACK.
    ///
    /// # Returns
    /// UMFPACK status code (0 for success)
    pub fn factorize_symbolic(&mut self) -> I {
        let mut symbolic = Symbolic::new();
        let mut info = Info::new();

        let status = analyze_symbolic(
            self.n,
            self.m,
            &self.ap,
            &self.ai,
            &self.ax,
            &mut symbolic,
            Some(&self.control),
            Some(&mut info),
        );

        if status == I::zero() {
            self.symbolic = Some(symbolic);
        }

        status
    }

    /// Performs numeric factorization of the matrix.
    ///
    /// This is the second step in solving a linear system with UMFPACK.
    /// The symbolic factorization must be computed first.
    ///
    /// # Returns
    /// UMFPACK status code (0 for success)
    pub fn factorize_numeric(&mut self) -> I {
        if self.symbolic.is_none() {
            panic!("Symbolic factorization must be performed before numeric factorization.");
        }

        let mut numeric = Numeric::new();
        let mut info = Info::new();

        let status = analyze_numeric(
            &self.ap,
            &self.ai,
            &self.ax,
            self.symbolic.as_ref().unwrap(),
            &mut numeric,
            Some(&self.control),
            Some(&mut info),
        );

        if status == I::zero() {
            self.numeric = Some(numeric);
        }

        status
    }

    /// Performs complete factorization (both symbolic and numeric).
    ///
    /// This is a convenience method that performs both factorization steps.
    ///
    /// # Returns
    /// UMFPACK status code (0 for success)
    pub fn factorize(&mut self) -> I {
        let symbolic_status = self.factorize_symbolic();
        if symbolic_status != I::zero() {
            return symbolic_status;
        }

        self.factorize_numeric()
    }

    /// Solves the linear system Ax = b.
    ///
    /// The matrix must be factorized before calling this method.
    ///
    /// # Arguments
    /// * `b` - Right-hand side vector (must have length equal to matrix rows)
    /// * `system_type` - Type of system to solve (defaults to UMFPACK_A)
    ///
    /// # Returns
    /// Result containing either the solution vector or an error code
    pub fn solve_system(&self, b: &[F], system_type: UMFPACK) -> Result<Vec<F>, I> {
        if self.numeric.is_none() {
            panic!("Matrix must be factorized before solving.");
        }

        if b.len() != self.n.try_into().unwrap() {
            panic!("Right-hand side vector length must match matrix rows.");
        }

        let mut x = vec![F::default(); b.len()];
        let mut info = Info::new();

        let status = solve(
            system_type,
            &self.ap,
            &self.ai,
            &self.ax,
            &mut x,
            b,
            self.numeric.as_ref().unwrap(),
            Some(&self.control),
            Some(&mut info),
        );

        if status == I::zero() {
            Ok(x)
        } else {
            Err(status)
        }
    }

    /// Computes x = A^{-1}b for the given vector b.
    ///
    /// This is a convenience method that factorizes the matrix if needed
    /// and then solves the linear system.
    ///
    /// # Arguments
    /// * `b` - Right-hand side vector (must have length equal to matrix rows)
    ///
    /// # Returns
    /// Result containing either the solution vector or an error code
    pub fn inverse_times(&mut self, b: &[F]) -> Result<Vec<F>, I> {
        // Check if factorization is needed
        if self.numeric.is_none() {
            let status = self.factorize();
            if status != I::zero() {
                return Err(status);
            }
        }

        // Solve the system
        self.solve_system(b, UMFPACK::A)
    }
}

impl<I, F> Drop for UMFPackMatrix<I, F>
where
    (I, F): UMFPack<I, F>,
{
    fn drop(&mut self) {
        // Free UMFPACK resources
        self.numeric = None;
        self.symbolic = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;

    // Create a simple sparse matrix for testing:
    // [2 3 0]
    // [0 1 4]
    // [5 0 6]
    fn create_test_matrix_real() -> UMFPackMatrix<i32, f64> {
        // CSC format:
        // Column 0: entries at rows 0, 2
        // Column 1: entries at rows 0, 1 
        // Column 2: entries at rows 1, 2
        let n = 3;
        let m = 3;
        let ap = vec![0, 2, 4, 6];   // Column pointers
        let ai = vec![0, 2, 0, 1, 1, 2]; // Row indices
        let ax = vec![2.0, 5.0, 3.0, 1.0, 4.0, 6.0]; // Values

        UMFPackMatrix::new(n, m, ap, ai, ax)
    }

    // Create a simple complex sparse matrix for testing
    fn create_test_matrix_complex() -> UMFPackMatrix<i32, Complex<f64>> {
        let n = 3;
        let m = 3;
        let ap = vec![0, 2, 4, 6];
        let ai = vec![0, 2, 0, 1, 1, 2];
        let ax = vec![
            Complex::new(2.0, 0.1), Complex::new(5.0, 0.2),
            Complex::new(3.0, 0.3), Complex::new(1.0, 0.4),
            Complex::new(4.0, 0.5), Complex::new(6.0, 0.6),
        ];

        UMFPackMatrix::new(n, m, ap, ai, ax)
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = create_test_matrix_real();
        assert_eq!(matrix.n, 3);
        assert_eq!(matrix.m, 3);
        assert_eq!(matrix.ap, vec![0, 2, 4, 6]);
        assert_eq!(matrix.ai, vec![0, 2, 0, 1, 1, 2]);
        assert_eq!(matrix.ax, vec![2.0, 5.0, 3.0, 1.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_creation_invalid_dimensions() {
        // ap should have length m+1, but here it's too short
        let n = 3;
        let m = 3;
        let ap = vec![0, 2, 4]; // Missing the last element
        let ai = vec![0, 2, 0, 1, 1, 2];
        let ax = vec![2.0, 5.0, 3.0, 1.0, 4.0, 6.0];

        UMFPackMatrix::new(n, m, ap, ai, ax);
    }

    #[test]
    fn test_factorize_real() {
        let mut matrix = create_test_matrix_real();
        let status = matrix.factorize();
        assert_eq!(status, 0);
        assert!(matrix.symbolic.is_some());
        assert!(matrix.numeric.is_some());
    }

    #[test]
    fn test_factorize_complex() {
        let mut matrix = create_test_matrix_complex();
        let status = matrix.factorize();
        assert_eq!(status, 0);
        assert!(matrix.symbolic.is_some());
        assert!(matrix.numeric.is_some());
    }

    #[test]
    fn test_solve_real() {
        let mut matrix = create_test_matrix_real();
        matrix.factorize();

        // For the test matrix:
        // [2 3 0] [x1]   [5]
        // [0 1 4] [x2] = [9]
        // [5 0 6] [x3]   [17]
        // The solution is [1, 1, 2]
        let b = vec![5.0, 9.0, 17.0];
        let result = matrix.solve_system(&b, UMFPACK::A);
        
        assert!(result.is_ok());
        let x = result.unwrap();
        assert_eq!(x.len(), 3);
        
        // Check approximate equality with tolerance
        let tol = 1e-10;
        assert!((x[0] - 1.0).abs() < tol);
        assert!((x[1] - 1.0).abs() < tol);
        assert!((x[2] - 2.0).abs() < tol);
    }

    #[test]
    fn test_solve_complex() {
        let mut matrix = create_test_matrix_complex();
        matrix.factorize();

        // Create a complex right-hand side
        let b = vec![
            Complex::new(7.0, 0.5),
            Complex::new(9.0, 0.6),
            Complex::new(11.0, 0.7),
        ];
        
        let result = matrix.solve_system(&b, UMFPACK::A);
        assert!(result.is_ok());
        
        let x = result.unwrap();
        assert_eq!(x.len(), 3);
        
        // For complex matrices, we just check that we got a solution
        // The exact values depend on the specific complex values used
    }

    #[test]
    fn test_inverse_times() {
        let mut matrix = create_test_matrix_real();
        
        // Matrix isn't factorized yet, inverse_times should do it automatically
        let b = vec![5.0, 9.0, 17.0];
        let result = matrix.inverse_times(&b);
        
        assert!(result.is_ok());
        let x = result.unwrap();
        
        // Check approximate equality with tolerance
        let tol = 1e-10;
        assert!((x[0] - 1.0).abs() < tol);
        assert!((x[1] - 1.0).abs() < tol);
        assert!((x[2] - 2.0).abs() < tol);
        
        // Check that factorization occurred
        assert!(matrix.symbolic.is_some());
        assert!(matrix.numeric.is_some());
    }

    #[test]
    #[should_panic]
    fn test_solve_wrong_size() {
        let mut matrix = create_test_matrix_real();
        matrix.factorize();
        
        // Vector of wrong size
        let b = vec![1.0, 2.0]; // Only 2 elements, but matrix has 3 rows
        
        // This should panic
        let _ = matrix.solve_system(&b, UMFPACK::A);
    }

    #[test]
    fn test_different_system_types() {
        let mut matrix = create_test_matrix_real();
        matrix.factorize();
        
        let b = vec![7.0, 9.0, 11.0];
        
        // Test standard solve
        let result1 = matrix.solve_system(&b, UMFPACK::A);
        assert!(result1.is_ok());
        
        // Test transpose solve
        let result2 = matrix.solve_system(&b, UMFPACK::At);
        assert!(result2.is_ok());
        
        // Solutions should be different
        let x1 = result1.unwrap();
        let x2 = result2.unwrap();
        assert!(x1 != x2);
    }
}
