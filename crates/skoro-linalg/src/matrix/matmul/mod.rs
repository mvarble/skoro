mod op;
mod sugar;

pub use sugar::*;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Matmul(pub usize, pub usize, pub usize);
