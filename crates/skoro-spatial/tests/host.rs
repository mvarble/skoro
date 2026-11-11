#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The host maps of the rigid motions, against the graph's.

mod common;

use common::{Rows, close, on_host, quaternions, vectors, with_translations};
use skoro_core::{Declares, Q, QDyn, Vector};
use skoro_spatial::{Pose, host};

fn p(q: QDyn) -> Q<Pose> {
    q.narrow()
}

#[test]
fn host_pose_maps_agree_with_the_graph() {
    // Twenty random poses, with the quaternions of either sign: the fixed ones come with their
    // negatives among them, and the random ones are drawn from the whole sphere.
    let a = with_translations(&quaternions(20, 301), 1, 302);
    let b = with_translations(&quaternions(20, 303), 1, 304);
    let x: Vec<[f64; 3]> = vectors::<3>(a.len() / 7, 305, -5.0, 5.0);
    let rows = a.len() / 7;
    assert!(rows >= 20);
    let x_flat: Vec<f64> = x.iter().flatten().copied().collect();
    let inputs = [
        Rows::new(Pose.space(), a.clone()),
        Rows::new(Pose.space(), b.clone()),
        Rows::new(skoro_core::Space::Vector(3), x_flat),
    ];
    let composed = on_host(|q| (p(q[0]) * p(q[1])).untyped(), &inputs);
    let moved = on_host(
        |q| (p(q[0]) * q[2].narrow::<Vector<3>>()).untyped(),
        &inputs,
    );
    let (mut want_c, mut want_m) = (Vec::new(), Vec::new());
    for r in 0..rows {
        let ta: [f64; 7] = a[7 * r..7 * r + 7].try_into().unwrap();
        let tb: [f64; 7] = b[7 * r..7 * r + 7].try_into().unwrap();
        want_c.extend(host::pose_compose(ta, tb));
        want_m.extend(host::pose_transform(ta, x[r]));
    }
    close(&composed, &want_c, 1e-14);
    close(&moved, &want_m, 1e-14);
}
