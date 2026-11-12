#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Measured rotations in whatever sign they are given: a pose graph read from g2o's format, its
//! information converted to skoro's coordinates.

#[path = "../examples/common/g2o.rs"]
#[allow(dead_code)]
mod g2o;

use std::collections::BTreeMap;

use skoro::scope::{constant, quantity};
use skoro::tensor::{Points, Value};
use skoro::{Declares, NodeId, Observations, Q, Query, Snapshot, Target, Values, Writer};
use skoro_linearize::{Assembly, Error, LevenbergMarquardt, Linearizer, Status};
use skoro_normal::{Normal, Scale};
use skoro_spatial::{POSE, Pose, PoseMaps, Rotation, host};

/// `(γ(φ), ρ)`: the Cayley chart's step at the identity, in storage order.
fn cayley_step(tau: [f64; 6]) -> [f64; 7] {
    let s = (1.0 + (tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]) / 4.0).sqrt();
    [
        1.0 / s,
        tau[0] / 2.0 / s,
        tau[1] / 2.0 / s,
        tau[2] / 2.0 / s,
        tau[3],
        tau[4],
        tau[5],
    ]
}

/// g2o's error of the measurement `z` of the prediction `h`: the translation of `Δ = z⁻¹ h` and the
/// vector part of its quaternion, read off its rotation matrix with `w ≥ 0`.
fn g2o_error(z: [f64; 7], h: [f64; 7]) -> [f64; 6] {
    let delta = g2o::between(z, h);
    let q = host::from_matrix(host::matrix([delta[0], delta[1], delta[2], delta[3]]));
    let q = host::canonical(q);
    [delta[4], delta[5], delta[6], q[1], q[2], q[3]]
}

fn quadratic(m: &[f64; 36], v: &[f64; 6]) -> f64 {
    (0..6)
        .flat_map(|i| (0..6).map(move |j| (i, j)))
        .map(|(i, j)| v[i] * m[i * 6 + j] * v[j])
        .sum()
}

/// An edge line of g2o's format between two vertices, with the measurement `z` and the upper
/// triangle of `information`.
fn edge_line(z: [f64; 7], information: &[f64; 36]) -> String {
    let mut line = format!(
        "EDGE_SE3:QUAT 0 1 {} {} {} {} {} {} {}",
        z[4], z[5], z[6], z[1], z[2], z[3], z[0]
    );
    for i in 0..6 {
        for j in i..6 {
            line.push_str(&format!(" {}", information[i * 6 + j]));
        }
    }
    line
}

#[test]
fn g2o_information_is_converted() {
    // A diagonal information and one with a block coupling the translation to the rotation, both
    // symmetric positive definite, g2o's order: translation, then rotation.
    let mut diagonal = [0.0; 36];
    for (i, v) in [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        .into_iter()
        .enumerate()
    {
        diagonal[i * 6 + i] = v;
    }
    let mut coupled = diagonal;
    for (i, j, v) in [(0, 4, 30.0), (1, 3, -20.0), (2, 5, 40.0), (3, 4, 10.0)] {
        coupled[i * 6 + j] = v;
        coupled[j * 6 + i] = v;
    }
    let z = [0.9, 0.1, -0.3, 0.3, 1.0, 2.0, 3.0];
    let norm = (0.81f64 + 0.01 + 0.09 + 0.09).sqrt();
    let z = [
        z[0] / norm,
        z[1] / norm,
        z[2] / norm,
        z[3] / norm,
        z[4],
        z[5],
        z[6],
    ];
    for information in [diagonal, coupled] {
        let graph = g2o::read(&edge_line(z, &information)).unwrap();
        let edge = &graph.edges[0];
        // The measurement, reordered to (w, x, y, z) and normalized, its sign kept.
        for (got, want) in edge.measurement.iter().zip(&z) {
            assert!((got - want).abs() < 1e-12, "{:?}", edge.measurement);
        }
        // The precision: a quarter of the rotation block, half the coupling block, the
        // translation block, rotation first.
        for i in 0..6 {
            for j in 0..6 {
                let (gi, gj) = ((i + 3) % 6, (j + 3) % 6);
                let scale = [i, j]
                    .iter()
                    .map(|&k| if k < 3 { 0.5 } else { 1.0 })
                    .product::<f64>();
                let want = scale * information[gi * 6 + gj];
                assert_eq!(edge.precision[i * 6 + j], want, "({i}, {j})");
            }
        }
    }

    // The two quadratic forms differ at third order in the residual: halving it divides their
    // difference by about eight. The measurement is the prediction moved by τ in skoro's chart,
    // so that skoro's residual is τ exactly.
    let h = [0.8, -0.2, 0.5, 0.26, 0.4, -1.1, 2.0];
    let n = h[..4].iter().map(|v| v * v).sum::<f64>().sqrt();
    let h = [h[0] / n, h[1] / n, h[2] / n, h[3] / n, h[4], h[5], h[6]];
    let direction = [0.5, -0.3, 0.6, 0.2, -0.4, 0.3];
    let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    let lambda = g2o::precision(&coupled);
    let difference = |size: f64| {
        let tau = direction.map(|v| size * v / length);
        let z = host::pose_compose(h, cayley_step(tau));
        let e = g2o_error(z, h);
        quadratic(&lambda, &tau) - quadratic(&coupled, &e)
    };
    let (large, small) = (difference(0.1), difference(0.05));
    let ratio = large / small;
    assert!(
        (6.0..10.0).contains(&ratio),
        "{large:e} then {small:e}: a ratio of {ratio}"
    );
}

/// A pose graph read from g2o's format, as a model: vertex 0 a constant at its initial pose, every
/// other vertex a parameter scored by no term, and each edge a measured motion scored by the
/// normal of the vertices' relative motion with the edge's precision.
struct Model {
    snapshot: Snapshot,
    vertices: BTreeMap<usize, Q<Pose>>,
    edges: Vec<(Q<Pose>, NodeId)>,
}

fn model(graph: &g2o::Graph) -> Model {
    let mut writer = Writer::new();
    let (vertices, edges) = writer
        .build(|| {
            let vertices: BTreeMap<usize, Q<Pose>> = graph
                .vertices
                .iter()
                .map(|(&k, x)| {
                    let q = if k == 0 {
                        constant(Pose, *x)
                    } else {
                        quantity(format!("vertex/{k}"), Pose)
                    };
                    (k, q)
                })
                .collect();
            let edges: Vec<(Q<Pose>, NodeId)> = graph
                .edges
                .iter()
                .enumerate()
                .map(|(e, edge)| {
                    let z = quantity(format!("edge/{e}"), Pose);
                    let mean = vertices[&edge.from].between(vertices[&edge.to]);
                    let term =
                        Normal::on(Pose, mean, Scale::precision(&edge.precision, 6)).score(z);
                    (z, NodeId::from(term))
                })
                .collect();
            (vertices, edges)
        })
        .unwrap();
    Model {
        snapshot: writer.snapshot(),
        vertices,
        edges,
    }
}

/// The observations of the edges' measurements and the start at the vertices' initial poses.
fn given(graph: &g2o::Graph, model: &Model, target: &Target) -> (Observations, Values) {
    let mut observations = Observations::new();
    for ((z, _), edge) in model.edges.iter().zip(&graph.edges) {
        observations = observations.observe(*z, edge.measurement);
    }
    let mut start = Values::new();
    for (k, x) in &graph.vertices {
        if *k != 0 {
            let value = Value::Continuous(Points::from_host(x, 1, 7, target));
            start = start.with(model.vertices[k].id(), value);
        }
    }
    (observations, start)
}

/// The estimate of a graph: every vertex but the first, in order, and the final cost.
fn estimate(graph: &g2o::Graph) -> (Vec<[f64; 7]>, f64) {
    let target = Target::host();
    let model = model(graph);
    let (observations, start) = given(graph, &model, &target);
    let query = Query::log_density().given(observations);
    let linearizer = Linearizer::new(&model.snapshot, &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let minimized = LevenbergMarquardt::default()
        .minimize(&linearizer, &schedule, &start)
        .unwrap();
    assert_eq!(
        minimized.rows[0].status,
        Status::Converged,
        "{:?}",
        minimized.rows[0]
    );
    let cost = linearizer.objective(&minimized.point).unwrap()[0];
    let poses = graph
        .vertices
        .keys()
        .filter(|&&k| k != 0)
        .map(|k| {
            let Value::Continuous(points) = minimized.point.get(model.vertices[k].id()).unwrap()
            else {
                panic!("a pose is continuous");
            };
            points.to_host().try_into().unwrap()
        })
        .collect();
    (poses, cost)
}

/// The angle between two rotations, `2 atan2(|vec r|, |w(r)|)` of `r = q̄₁ q₂`.
fn angle(a: [f64; 4], b: [f64; 4]) -> f64 {
    let r = host::compose([a[0], -a[1], -a[2], -a[3]], b);
    2.0 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3])
        .sqrt()
        .atan2(r[0].abs())
}

fn quaternion(x: &[f64; 7]) -> [f64; 4] {
    [x[0], x[1], x[2], x[3]]
}

#[test]
fn a_one_turn_loop_closes() {
    for (noise, tolerance) in [(0.0, 1e-6), (0.01, 0.05)] {
        let graph = g2o::read(&g2o::one_turn_loop(24, noise, 20_261_111)).unwrap();
        let closure = graph.edges.last().unwrap();
        let (from, to) = (graph.vertices[&closure.from], graph.vertices[&closure.to]);

        // At the start, composed from the odometry, the loop closure is on the far side of its
        // prediction: the case the stereographic chart could not satisfy.
        let predicted = g2o::between(from, to);
        let inner: f64 = (0..4).map(|i| predicted[i] * closure.measurement[i]).sum();
        assert!(inner < 0.0, "the closure is on the near side: {inner}");

        // From there the estimate converges, and every relative rotation is within the noise of
        // its measurement: no turn is wound into it.
        let (poses, _) = estimate(&graph);
        let mut at = graph.vertices.clone();
        for (k, x) in graph.vertices.keys().filter(|&&k| k != 0).zip(&poses) {
            at.insert(*k, *x);
        }
        for edge in &graph.edges {
            let relative = g2o::between(at[&edge.from], at[&edge.to]);
            let misfit = angle(quaternion(&relative), quaternion(&edge.measurement));
            assert!(
                misfit < tolerance,
                "edge {} → {}: {misfit} rad",
                edge.from,
                edge.to
            );
        }
    }
}

/// Which edges or vertices to negate, by index.
type Choice<'a> = &'a dyn Fn(usize) -> bool;

#[test]
fn the_estimate_ignores_every_sign() {
    let text = g2o::one_turn_loop(24, 0.01, 20_261_112);
    let (reference, cost) = estimate(&g2o::read(&text).unwrap());
    let negate = |x: &mut [f64; 7]| {
        for v in &mut x[..4] {
            *v = -*v;
        }
    };
    // Every measurement, every start, and half of each, chosen by parity.
    let variants: [(Choice, Choice); 3] = [
        (&|_| true, &|_| false),
        (&|_| false, &|_| true),
        (&|i| i % 2 == 0, &|i| i % 3 == 1),
    ];
    for (edges, vertices) in variants {
        let mut graph = g2o::read(&text).unwrap();
        for (e, edge) in graph.edges.iter_mut().enumerate() {
            if edges(e) {
                negate(&mut edge.measurement);
            }
        }
        for (k, x) in graph.vertices.iter_mut() {
            if *k != 0 && vertices(*k) {
                negate(x);
            }
        }
        let (poses, other) = estimate(&graph);
        skoro::testing::close_up_to(&POSE, &poses.concat(), &reference.concat(), 1e-6).unwrap();
        assert!(
            (other - cost).abs() <= 1e-9 * cost.abs().max(1.0),
            "{other} against {cost}"
        );
    }
}

#[test]
fn a_half_turn_is_non_finite() {
    // A rotation observed at (0, 1, 0, 0), a half-turn from its mean (1, 0, 0, 0) at the start.
    let target = Target::host();
    let mut writer = Writer::new();
    let (x, z, term) = writer
        .build(|| {
            let x = quantity("x", Rotation);
            let z = quantity("z", Rotation);
            let term = Normal::on(Rotation, x, Scale::sd(0.1)).score(z);
            (x, z, NodeId::from(term))
        })
        .unwrap();
    let query = Query::log_density().given(Observations::new().observe(z, [0.0, 1.0, 0.0, 0.0]));
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let start = Values::new().with(
        x.id(),
        Value::Continuous(Points::from_host(
            &[1.0, 0.0, 0.0, 0.0],
            1,
            Rotation.space().storage(),
            &target,
        )),
    );
    assert_eq!(
        linearizer.at(&start).unwrap_err(),
        Error::NotFinite { term, row: 0 }
    );
}
