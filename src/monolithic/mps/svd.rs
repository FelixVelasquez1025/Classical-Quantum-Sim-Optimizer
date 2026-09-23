//! Bounded complex one-sided Jacobi SVD for normalized MPS two-site matrices.
//! Rotations act directly on columns, avoiding squared condition numbers from
//! diagonalizing A†A. The previous nalgebra complex SVD failed reconstruction
//! on rank-deficient circuit tensors (covered by the circuit regressions).
use nalgebra::DMatrix;
use num_complex::Complex64 as C;

type Factors = (DMatrix<C>, Vec<f64>, DMatrix<C>);

pub(super) fn decompose(matrix: DMatrix<C>) -> Result<Factors, String> {
    if matrix.nrows() < matrix.ncols() {
        let (u, s, vt) = tall(matrix.adjoint())?;
        Ok((vt.adjoint(), s, u.adjoint()))
    } else {
        tall(matrix)
    }
}

fn tall(mut a: DMatrix<C>) -> Result<Factors, String> {
    let (m, n) = (a.nrows(), a.ncols());
    let scale = a.iter().map(|z| z.norm()).fold(0., f64::max);
    if !scale.is_finite() || scale == 0. {
        return Err("MPS SVD encountered a zero or nonfinite matrix".into());
    }
    a /= C::new(scale, 0.);
    let mut v = DMatrix::<C>::identity(n, n);
    let tolerance = 4. * f64::EPSILON;
    let floor = f64::EPSILON * (m.max(n) as f64) * a.norm();
    let mut converged = false;
    for _ in 0..100 {
        let mut changed = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha: f64 = (0..m).map(|i| a[(i, p)].norm_sqr()).sum();
                let beta: f64 = (0..m).map(|i| a[(i, q)].norm_sqr()).sum();
                // Values at numerical rank precision are removed after convergence.
                if alpha.sqrt() <= floor || beta.sqrt() <= floor {
                    continue;
                }
                let gamma: C = (0..m).map(|i| a[(i, p)].conj() * a[(i, q)]).sum();
                let g = gamma.norm();
                if g <= tolerance * alpha.sqrt() * beta.sqrt() {
                    continue;
                }
                changed = true;
                let tau = (beta - alpha) / (2. * g);
                let t = if tau >= 0. { 1. } else { -1. } / (tau.abs() + tau.hypot(1.));
                let c = 1. / (1. + t * t).sqrt();
                let s = c * t;
                let phase = gamma / g;
                for i in 0..m {
                    let x = a[(i, p)];
                    let y = a[(i, q)];
                    a[(i, p)] = c * x - s * phase.conj() * y;
                    a[(i, q)] = s * phase * x + c * y;
                }
                for i in 0..n {
                    let x = v[(i, p)];
                    let y = v[(i, q)];
                    v[(i, p)] = c * x - s * phase.conj() * y;
                    v[(i, q)] = s * phase * x + c * y;
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("MPS Jacobi SVD failed to converge in 100 sweeps".into());
    }
    let norms: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| a[(i, j)].norm_sqr()).sum::<f64>().sqrt())
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    // Numerical rank removal is separate from intentional physical truncation.
    let rank = order
        .iter()
        .take_while(|&&j| norms[j] > floor)
        .count()
        .max(1);
    let u = DMatrix::from_fn(m, rank, |i, j| a[(i, order[j])] / norms[order[j]]);
    let vt = DMatrix::from_fn(rank, n, |i, j| v[(j, order[i])].conj());
    let s = order[..rank].iter().map(|&j| norms[j] * scale).collect();
    Ok((u, s, vt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn reconstructs_complex_rectangular_rank_deficient_and_degenerate_matrices() {
        let mut rng = ChaCha8Rng::seed_from_u64(81);
        for (m, n) in [(2, 2), (4, 8), (8, 4), (7, 7), (16, 32)] {
            for rank in [1, 2, m.min(n)] {
                let left = DMatrix::from_fn(m, rank, |_, _| {
                    C::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5)
                });
                let right = DMatrix::from_fn(rank, n, |_, _| {
                    C::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5)
                });
                let matrix = left * right;
                let (u, s, vt) = decompose(matrix.clone()).unwrap();
                let mut us = u.clone();
                for j in 0..s.len() {
                    us.column_mut(j).scale_mut(s[j]);
                }
                assert!((&us * &vt - &matrix).norm() < 2e-13 * matrix.norm());
                assert!((u.adjoint() * &u - DMatrix::identity(s.len(), s.len())).norm() < 1e-12);
                assert!((&vt * vt.adjoint() - DMatrix::identity(s.len(), s.len())).norm() < 1e-12);
                assert!(s.windows(2).all(|p| p[0] >= p[1]));
            }
        }
        let matrix = DMatrix::<C>::identity(8, 8) * C::new(0., 1.);
        let (u, s, vt) = decompose(matrix.clone()).unwrap();
        assert!((&u * &vt - matrix).norm() < 1e-14);
        assert!(s.iter().all(|&x| (x - 1.).abs() < 1e-14));
    }
}
