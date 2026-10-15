//! A complex vector of `N` entries, a marker of a crate's own: a flat embedded type of storage
//! `2N`, the real parts then the imaginary parts, with its product through `QMul`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use skoro::scope::{concatenate, project};
use skoro::{Composite, Declares, Embedded, Param, Q, QMul, Space, Typed};

#[derive(Clone, Copy, Debug)]
pub struct Complex<const N: usize>;

/// The declaration of each size used, made once.
static DECLARATIONS: LazyLock<Mutex<HashMap<usize, Arc<Embedded>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn declaration(n: usize) -> Arc<Embedded> {
    let mut declarations = DECLARATIONS.lock().unwrap();
    declarations
        .entry(n)
        .or_insert_with(|| {
            Embedded::declare_flat("complex", vec![Param::Integer(n as i64)], 2 * n).unwrap()
        })
        .clone()
}

impl<const N: usize> Typed for Complex<N> {
    fn admits(space: &Space) -> bool {
        *space == Space::Embedded(declaration(N))
    }
}

impl<const N: usize> Declares for Complex<N> {
    fn space(&self) -> Space {
        Space::Embedded(declaration(N))
    }
}

/// The entrywise product `(a + ib)(c + id) = (ac - bd) + i(ad + bc)`, in storage, projected back.
pub fn product<const N: usize>() -> Arc<Composite> {
    let space = Complex::<N>.space();
    Composite::define(
        "complex.mul",
        vec![Param::Integer(N as i64)],
        &[space.clone(), space],
        |q| {
            let (x, y) = (q[0].include(), q[1].include());
            let (a, b) = (x.slice(0..N), x.slice(N..2 * N));
            let (c, d) = (y.slice(0..N), y.slice(N..2 * N));
            let stored = concatenate(&[a * c - b * d, a * d + b * c], Space::Vector(2 * N));
            project(Complex::<N>, stored).untyped()
        },
    )
    .unwrap()
}

impl<const N: usize> QMul for Complex<N> {
    type Output = Complex<N>;

    fn mul(lhs: Q<Complex<N>>, rhs: Q<Complex<N>>) -> Q<Complex<N>> {
        product::<N>()
            .call(&[lhs.untyped(), rhs.untyped()])
            .narrow::<Complex<N>>()
    }
}
