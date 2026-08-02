mod op;
mod sugar;

pub use sugar::*;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Transpose(pub usize, pub usize);
