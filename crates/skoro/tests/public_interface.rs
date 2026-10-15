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
