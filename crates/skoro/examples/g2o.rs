//! A 3D pose graph in g2o's format, estimated in whatever signs its quaternions are given.
//!
//! Given a path, the example reads that file; any 3D pose graph g2o writes will do. Without one, it
//! reads a one-turn loop of a hundred poses whose vertices were composed from the odometry, so that
//! the loop closure starts on the far side of its prediction. Vertex 0 is held at its initial pose,
//! every other vertex is a parameter, and each edge is a measured motion, scored by the normal of
//! the vertices' relative motion with its information converted to skoro's coordinates. The
//! example runs Levenberg–Marquardt from the vertices' initial poses and checks that it converged.

mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use skoro::scope::{constant, quantity};
use skoro::tensor::{Points, Value};
use skoro::{Observations, Q, Query, Target, Values, Writer};
use skoro_linearize::{Assembly, LevenbergMarquardt, Linearizer};
use skoro_normal::{Normal, Scale};
use skoro_spatial::{Pose, PoseMaps};

use common::g2o;
use common::report::{Result, converged};

/// The seed of the built-in loop, so that every run of the example is the same run.
const SEED: u64 = 20_261_111;

fn main() -> Result {
    let started = Instant::now();
    let target = Target::host();
    let text = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?,
        None => g2o::one_turn_loop(100, 0.01, SEED),
    };
    let graph = g2o::read(&text)?;
    if !graph.vertices.contains_key(&0) {
        return Err("the graph has no vertex 0 to hold".into());
    }
    println!(
        "{} vertices, {} edges, {} other lines skipped",
        graph.vertices.len(),
        graph.edges.len(),
        graph.skipped
    );

    // The model, the measurements and the start.
    let mut writer = Writer::new();
    let (vertices, measured) = writer.build(|| {
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
        let measured: Vec<Q<Pose>> = graph
            .edges
            .iter()
            .enumerate()
            .map(|(e, edge)| {
                let z = quantity(format!("edge/{e}"), Pose);
                let mean = vertices[&edge.from].between(vertices[&edge.to]);
                Normal::on(Pose, mean, Scale::precision(&edge.precision, 6)).score(z);
                z
            })
            .collect();
        (vertices, measured)
    })?;
    let mut observations = Observations::new();
    for (z, edge) in measured.iter().zip(&graph.edges) {
        observations = observations.observe(*z, edge.measurement);
    }
    let mut start = Values::new();
    for (k, x) in graph.vertices.iter().filter(|(k, _)| **k != 0) {
        start = start.with(
            vertices[k].id(),
            Value::Continuous(Points::from_host(x, 1, 7, &target)),
        );
    }

    // The mode.
    let query = Query::log_density().given(observations);
    let linearizer = Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual)?;
    let schedule = linearizer.schedule()?;
    let at_start = linearizer.objective(&start)?[0];
    let minimized = LevenbergMarquardt::default().minimize(&linearizer, &schedule, &start)?;
    converged(&minimized.rows, "the graph")?;
    let at_mode = linearizer.objective(&minimized.point)?[0];
    println!(
        "cost {at_start:.6e} at the start, {at_mode:.6e} at the mode, in {} iterations",
        minimized.rows[0].iterations
    );
    println!("solved in {:.1} s", started.elapsed().as_secs_f64());
    println!("every check passed");
    Ok(())
}
