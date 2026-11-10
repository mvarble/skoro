#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
#![allow(clippy::needless_range_loop)] // dense linear algebra, by index
//! Elimination in square-root form, against the examples and statements of the note.

use std::sync::Arc;

use skoro_elimination::{Matrix, Problem, Rows, Schedule, Structure};

const DOUBLE: f64 = f64::EPSILON / 2.0;

/// A deterministic stream of numbers in `[-1, 1)`.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 52) as f64 - 1.0
    }

    fn take(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Rows from a matrix given row by row.
fn rows(a: &[&[f64]], b: &[f64]) -> Rows {
    let columns = a.first().map_or(0, |r| r.len());
    Rows {
        a: Matrix::new(a.len(), columns, a.concat()).unwrap(),
        b: b.to_vec(),
    }
}

/// Random rows for every factor of `structure`, `extra` more rows than columns each.
fn random_problem(structure: &Arc<Structure>, extra: usize, linear: bool, seed: u64) -> Problem {
    let mut stream = Stream(seed);
    let factors = structure
        .factors()
        .iter()
        .map(|blocks| {
            let width: usize = blocks.iter().map(|&k| structure.blocks()[k]).sum();
            let m = width + extra;
            Rows {
                a: Matrix::new(m, width, stream.take(m * width)).unwrap(),
                b: stream.take(m),
            }
        })
        .collect();
    let eta = linear.then(|| stream.take(structure.dimension()));
    Problem::new(structure.clone(), factors, eta, stream.next(), DOUBLE).unwrap()
}

/// The problem's rows over all of `D`, stacked, and its right-hand sides.
fn dense(problem: &Problem) -> (Vec<Vec<f64>>, Vec<f64>) {
    let structure = problem.structure();
    let d = structure.dimension();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (blocks, rows) in structure.factors().iter().zip(problem.factors()) {
        for r in 0..rows.b.len() {
            let mut row = vec![0.0; d];
            let mut column = 0;
            for &k in blocks {
                for c in structure.range(k) {
                    row[c] = rows.a.get(r, column);
                    column += 1;
                }
            }
            a.push(row);
            b.push(rows.b[r]);
        }
    }
    (a, b)
}

/// `AᵀA` and `Aᵀb`.
fn normal_equations(a: &[Vec<f64>], b: &[f64], d: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut gram = vec![vec![0.0; d]; d];
    let mut atb = vec![0.0; d];
    for (row, bi) in a.iter().zip(b) {
        for i in 0..d {
            atb[i] += row[i] * bi;
            for j in 0..d {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    (gram, atb)
}

/// `x` with `M x = v`, by Gaussian elimination with partial pivoting.
fn solve(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .zip(v)
        .map(|(r, v)| [r.clone(), vec![*v]].concat())
        .collect();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..=n {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let rest: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (a[i][n] - rest) / a[i][i];
    }
    x
}

fn inverse(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let columns: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            solve(
                m,
                &(0..n)
                    .map(|i| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    (0..n)
        .map(|i| (0..n).map(|j| columns[j][i]).collect())
        .collect()
}

fn close(got: &[f64], want: &[f64], tolerance: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= tolerance * w.abs().max(1.0),
            "{what}: {got:?} against {want:?}"
        );
    }
}

/// The dense minimizer of a problem: `(AᵀA) τ = Aᵀb - η`.
fn dense_minimizer(problem: &Problem) -> Vec<f64> {
    let d = problem.structure().dimension();
    let (a, b) = dense(problem);
    let (gram, mut rhs) = normal_equations(&a, &b, d);
    if let Some(eta) = problem.linear() {
        for (r, e) in rhs.iter_mut().zip(eta) {
            *r -= e;
        }
    }
    solve(&gram, &rhs)
}

/// The note's star: a hub, block 0, joined to each of `n` leaves by a factor of its own.
fn star(n: usize) -> Arc<Structure> {
    let factors = (1..=n).map(|leaf| vec![0, leaf]).collect();
    Arc::new(Structure::new(vec![2; n + 1], factors).unwrap())
}

/// The note's map: poses `x1, x2, x3` (blocks 0 to 2) and landmarks `l1, l2` (blocks 3 and 4).
/// A prior on `x1`, odometry between consecutive poses, a loop closure from `x1` to `x3`, `l1`
/// seen from `x1` and `x2`, and `l2` from `x2` and `x3`.
fn map() -> Arc<Structure> {
    let factors = vec![
        vec![0],
        vec![0, 1],
        vec![1, 2],
        vec![0, 2],
        vec![0, 3],
        vec![1, 3],
        vec![1, 4],
        vec![2, 4],
    ];
    Arc::new(Structure::new(vec![3, 3, 3, 2, 2], factors).unwrap())
}

#[test]
fn the_star_eliminates_its_leaves_first() {
    let n = 6;
    let structure = star(n);
    let schedule = Schedule::all(&structure).unwrap();
    assert_eq!(schedule.order().len(), n + 1);
    // The hub comes after the leaves, but for one it may share the last problem with: once the
    // others are gone, the hub and one leaf are a problem of two blocks either way.
    let hub = schedule.order().iter().position(|&k| k == 0).unwrap();
    assert!(
        hub >= n - 1,
        "the hub after the leaves: {:?}",
        schedule.order()
    );
    // No fill: every separator is at most one block.
    for i in 0..schedule.cliques() {
        assert!(
            schedule.separator(i).len() <= 1,
            "{:?}",
            schedule.separator(i)
        );
    }
    // The hub first instead leaves a factor over every leaf, and everything after it is one dense
    // front of every block.
    let hub_first = Schedule::in_order(&structure, (0..=n).collect()).unwrap();
    assert_eq!(hub_first.cliques(), 1);
    assert_eq!(hub_first.frontals(0), (0..=n).collect::<Vec<_>>());
}

#[test]
fn the_map_has_three_cliques() {
    let structure = map();
    let schedule = Schedule::in_order(&structure, vec![3, 4, 0, 1, 2]).unwrap();
    assert_eq!(schedule.cliques(), 3);
    let cliques: Vec<(Vec<usize>, Vec<usize>)> = (0..3)
        .map(|i| {
            (
                schedule.frontals(i).to_vec(),
                schedule.separator(i).to_vec(),
            )
        })
        .collect();
    assert!(cliques.contains(&(vec![3], vec![0, 1])), "{cliques:?}");
    assert!(cliques.contains(&(vec![4], vec![1, 2])), "{cliques:?}");
    assert_eq!(cliques[2], (vec![0, 1, 2], vec![]), "the root, last");
    assert!(schedule.kept().is_empty());
    // Merged into three fronts, the elimination gives the dense answers.
    let problem = random_problem(&structure, 1, true, 3);
    let elimination = problem.eliminate(&schedule, None).unwrap();
    close(
        &elimination.minimizer().unwrap(),
        &dense_minimizer(&problem),
        1e-10,
        "the minimizer",
    );
    let minimum = problem.value(&dense_minimizer(&problem));
    close(
        &[elimination.minimum().unwrap()],
        &[minimum],
        1e-10,
        "the minimum",
    );
}

#[test]
fn triangular_root_of_the_example() {
    let structure = Arc::new(Structure::new(vec![2], vec![vec![0]]).unwrap());
    let factor = rows(
        &[&[-2.0, 0.0], &[1.0, 5.0], &[2.0, 5.0]],
        &[-7.0, 16.0, -3.0],
    );
    let problem = Problem::new(structure.clone(), vec![factor], None, 0.0, DOUBLE).unwrap();
    let elimination = problem
        .eliminate(&Schedule::all(&structure).unwrap(), None)
        .unwrap();
    let conditional = &elimination.conditionals()[0];
    // R = [3 5; 0 -5] and d = (8, -5) in the note; with a positive diagonal, the second row's sign
    // is changed.
    close(conditional.r.data(), &[3.0, 5.0, 0.0, 5.0], 1e-14, "R");
    close(&conditional.rhs, &[8.0, 5.0], 1e-14, "d");
    close(
        &elimination.minimizer().unwrap(),
        &[1.0, 1.0],
        1e-14,
        "the minimizer",
    );
    // The minimum of ½|Aτ - b|² is ½ e² = 225 / 2.
    close(
        &[elimination.minimum().unwrap()],
        &[112.5],
        1e-14,
        "the minimum",
    );
}

#[test]
fn the_ill_conditioned_example_keeps_its_digits() {
    let e = 1e-8;
    let structure = Arc::new(Structure::new(vec![2], vec![vec![0]]).unwrap());
    let factor = rows(&[&[1.0, 1.0], &[e, 0.0], &[0.0, e]], &[2.0, e, e]);
    let problem = Problem::new(structure.clone(), vec![factor], None, 0.0, DOUBLE).unwrap();
    let elimination = problem
        .eliminate(&Schedule::all(&structure).unwrap(), None)
        .unwrap();
    close(
        &elimination.minimizer().unwrap(),
        &[1.0, 1.0],
        1e-14,
        "the minimizer",
    );
    // Cholesky of the normal equations, as computed, meets a zero pivot: 1 + ε² rounds to 1.
    let (a, b) = dense(&problem);
    let (gram, _) = normal_equations(&a, &b, 2);
    let l11 = gram[0][0].sqrt();
    let l21 = gram[1][0] / l11;
    let pivot = gram[1][1] - l21 * l21;
    assert!(
        pivot <= 0.0,
        "the normal equations are singular as computed: {pivot}"
    );
}

#[test]
fn eliminating_one_block_is_the_lemma() {
    // Block 0 is eliminated; blocks 1 and 2 are its separator. Every factor involves block 0.
    let structure = Arc::new(
        Structure::new(
            vec![2, 2, 1],
            vec![vec![0], vec![0, 1], vec![0, 2], vec![0, 1, 2]],
        )
        .unwrap(),
    );
    let problem = random_problem(&structure, 1, true, 11);
    let schedule = Schedule::eliminating(&structure, &[0]).unwrap();
    let elimination = problem.eliminate(&schedule, None).unwrap();
    let conditional = &elimination.conditionals()[0];
    assert_eq!(conditional.separator, [1, 2]);
    // φ(τ) = ½|R τ_k + T τ_S - (d - y)|² + φ'(τ_S) + γ, at random points.
    let mut stream = Stream(5);
    for _ in 0..5 {
        let tau = stream.take(5);
        let (k, s) = (&tau[0..2], &tau[2..5]);
        let w: Vec<f64> = conditional
            .r
            .apply(k)
            .iter()
            .zip(conditional.t.apply(s))
            .zip(&conditional.rhs)
            .map(|((r, t), d)| r + t - d)
            .collect();
        let mut value = 0.5 * w.iter().map(|w| w * w).sum::<f64>() + elimination.constant();
        for (blocks, rows) in elimination.left() {
            let x: Vec<f64> = blocks
                .iter()
                .flat_map(|&b| tau[structure.range(b)].to_vec())
                .collect();
            value += 0.5
                * rows
                    .a
                    .apply(&x)
                    .iter()
                    .zip(&rows.b)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
        }
        let eta = elimination.left_linear().unwrap();
        value += (2..5).map(|c| eta[c] * tau[c]).sum::<f64>();
        close(&[value], &[problem.value(&tau)], 1e-12, "the decomposition");
    }
    // The new factor's matrix is the Schur complement, and its gradient at zero the reduced one.
    let (a, b) = dense(&problem);
    let (gram, atb) = normal_equations(&a, &b, 5);
    let eta = problem.linear().unwrap();
    let g: Vec<f64> = (0..5).map(|i| eta[i] - atb[i]).collect();
    let block = |rows: std::ops::Range<usize>, cols: std::ops::Range<usize>| -> Vec<Vec<f64>> {
        rows.map(|i| cols.clone().map(|j| gram[i][j]).collect())
            .collect()
    };
    let (kk, sk) = (block(0..2, 0..2), block(2..5, 0..2));
    let kk_inv = inverse(&kk);
    let schur: Vec<f64> = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| {
            gram[2 + i][2 + j]
                - (0..2)
                    .flat_map(|p| (0..2).map(move |q| (p, q)))
                    .map(|(p, q)| sk[i][p] * kk_inv[p][q] * sk[j][q])
                    .sum::<f64>()
        })
        .collect();
    let (_, new) = &elimination.left()[0];
    let ata: Vec<f64> = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| {
            (0..new.a.rows())
                .map(|r| new.a.get(r, i) * new.a.get(r, j))
                .sum()
        })
        .collect();
    close(&ata, &schur, 1e-12, "the Schur complement");
    let reduced: Vec<f64> = (0..3)
        .map(|i| {
            g[2 + i]
                - (0..2)
                    .flat_map(|p| (0..2).map(move |q| (p, q)))
                    .map(|(p, q)| sk[i][p] * kk_inv[p][q] * g[q])
                    .sum::<f64>()
        })
        .collect();
    let left_eta = elimination.left_linear().unwrap();
    let gradient: Vec<f64> = (0..3)
        .map(|i| {
            left_eta[2 + i]
                - (0..new.a.rows())
                    .map(|r| new.a.get(r, i) * new.b[r])
                    .sum::<f64>()
        })
        .collect();
    close(&gradient, &reduced, 1e-12, "the gradient at zero");
}

#[test]
fn two_scalars() {
    // A prior τ₁ = 1 and an odometry τ₂ - τ₁ = 1, both of unit variance.
    let structure = Arc::new(Structure::new(vec![1, 1], vec![vec![0], vec![0, 1]]).unwrap());
    let factors = vec![rows(&[&[1.0]], &[1.0]), rows(&[&[-1.0, 1.0]], &[1.0])];
    let problem = Problem::new(structure.clone(), factors, None, 0.0, DOUBLE).unwrap();
    let first = problem
        .eliminate(&Schedule::eliminating(&structure, &[0]).unwrap(), None)
        .unwrap();
    let conditional = &first.conditionals()[0];
    let half = std::f64::consts::FRAC_1_SQRT_2;
    close(conditional.r.data(), &[2f64.sqrt()], 1e-15, "R");
    close(conditional.t.data(), &[-half], 1e-15, "T: τ₁ = τ₂ / 2");
    close(&conditional.rhs, &[0.0], 1e-15, "d");
    let (_, new) = &first.left()[0];
    let sign = new.a.get(0, 0).signum();
    close(
        &[sign * new.a.get(0, 0), sign * new.b[0]],
        &[half, 2f64.sqrt()],
        1e-15,
        "the new factor",
    );
    close(
        &[1.0 / new.a.get(0, 0).powi(2)],
        &[2.0],
        1e-14,
        "the marginal variance",
    );
    let whole = problem
        .eliminate(&Schedule::all(&structure).unwrap(), None)
        .unwrap();
    close(
        &whole.minimizer().unwrap(),
        &[1.0, 2.0],
        1e-14,
        "the minimizer",
    );
}

#[test]
fn a_linear_term_in_the_row_space_stays_there() {
    let structure = Arc::new(
        Structure::new(
            vec![2, 2, 1],
            vec![vec![0, 1], vec![0, 2], vec![0, 1, 2], vec![1, 2]],
        )
        .unwrap(),
    );
    let plain = random_problem(&structure, 2, false, 23);
    // η = Aᵀu, for a vector u of the rows' size.
    let (a, _) = dense(&plain);
    let mut stream = Stream(29);
    let u = stream.take(a.len());
    let eta: Vec<f64> = (0..5)
        .map(|j| a.iter().zip(&u).map(|(row, u)| row[j] * u).sum())
        .collect();
    let with = Problem::new(
        structure.clone(),
        plain.factors().to_vec(),
        Some(eta),
        plain.constant(),
        DOUBLE,
    )
    .unwrap();
    let schedule = Schedule::eliminating(&structure, &[0]).unwrap();
    let elimination = with.eliminate(&schedule, None).unwrap();
    // The linear term left on blocks 1 and 2 lies in the row space of the factors left there.
    let left_eta = elimination.left_linear().unwrap()[2..5].to_vec();
    let mut stacked: Vec<Vec<f64>> = Vec::new();
    for (blocks, rows) in elimination.left() {
        for r in 0..rows.a.rows() {
            let mut row = vec![0.0; 3];
            let mut column = 0;
            for &k in blocks {
                for c in structure.range(k) {
                    row[c - 2] = rows.a.get(r, column);
                    column += 1;
                }
            }
            stacked.push(row);
        }
    }
    let (gram, _) = normal_equations(&stacked, &vec![0.0; stacked.len()], 3);
    let v = solve(&gram, &left_eta);
    // left_eta = Mᵀ (M v') for the stacked M, if it is in M's row space: Mᵀ M v = left_eta.
    let back: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| gram[i][j] * v[j]).sum())
        .collect();
    close(&back, &left_eta, 1e-12, "in the row space");
    // Folding η = Aᵀu into the right-hand sides as b - u gives the same minimizer.
    let mut at = 0;
    let folded: Vec<Rows> = plain
        .factors()
        .iter()
        .map(|rows| {
            let n = rows.b.len();
            let b = rows
                .b
                .iter()
                .zip(&u[at..at + n])
                .map(|(b, u)| b - u)
                .collect();
            at += n;
            Rows {
                a: rows.a.clone(),
                b,
            }
        })
        .collect();
    let folded = Problem::new(structure.clone(), folded, None, 0.0, DOUBLE).unwrap();
    let all = Schedule::all(&structure).unwrap();
    close(
        &with.eliminate(&all, None).unwrap().minimizer().unwrap(),
        &folded.eliminate(&all, None).unwrap().minimizer().unwrap(),
        1e-12,
        "folded",
    );
    close(
        &with.eliminate(&all, None).unwrap().minimizer().unwrap(),
        &dense_minimizer(&with),
        1e-10,
        "dense",
    );
}
