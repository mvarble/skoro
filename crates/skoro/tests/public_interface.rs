#![allow(clippy::unwrap_used)] // a test crate
//! The umbrella crate's tests use `skoro`'s public interface alone, as a later entry's crate
//! would: none of them names the crates `skoro` is built from.

use std::path::Path;

fn sources(directory: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn tests_use_the_public_interface_alone() {
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files = Vec::new();
    sources(&tests, &mut files);
    assert!(files.len() > 1);
    let forbidden = ["skoro_core", "skoro_tensor", "extern crate"];
    for file in files {
        if file.ends_with("public_interface.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&file).unwrap();
        for word in forbidden {
            assert!(!text.contains(word), "{} names {word}", file.display());
        }
    }
}

#[test]
fn volumes_are_named_by_the_public_interface() {
    let real = skoro::Space::Real;
    assert!(real.has_measure());
    assert!(matches!(real.domain(), Ok(skoro::Domain::Global)));
    let volume: Option<&skoro::Volume> = None;
    assert!(volume.is_none());
}

#[test]
fn masses_are_named_by_the_public_interface() {
    fn nowhere(_: &[f64], _: &mut [f64]) {}
    let draw: skoro::UniformDraw = nowhere;
    let mass = skoro::Mass::new(0.0, draw);
    assert_eq!(mass.log_mass(), 0.0);
    let _: skoro::UniformDraw = mass.draw();
    // A flat declaration has neither a mass nor a place for one.
    let flat = skoro::Embedded::declare_flat("test.interface.flat", Vec::new(), 2).unwrap();
    assert!(flat.mass().is_none());
    assert!(flat.clone().with_mass(mass).is_err());
    let space = skoro::Space::Embedded(flat.clone());
    let no_mass = skoro::Error::NoMass {
        space: space.clone(),
    };
    assert_eq!(space.log_mass(), Err(no_mass.clone()));
    assert_eq!(space.draw_uniform(&[0.5; 2], &mut [0.0; 2]), Err(no_mass));
    assert_eq!(skoro::Space::BOOLEAN.log_mass(), Ok(2f64.ln()));
    let points = skoro::tensor::HostValue::new(
        Some(1),
        2,
        skoro::tensor::HostData::Continuous(vec![0.0, 0.0]),
    )
    .unwrap();
    let refused = skoro::testing::check_mass(&flat, &points, 1.0).unwrap_err();
    assert!(refused.contains("no mass"), "{refused}");
}
