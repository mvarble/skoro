use std::sync::atomic::{AtomicU64, Ordering};

/// A simple wrapper for `u64` which can only be obtained from a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(u64);

/// A generator of [`Id`] instances.
#[derive(Debug, Default)]
pub struct IdGenerator(AtomicU64);

impl Clone for IdGenerator {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

impl IdGenerator {
    /// Create a new generator starting at 0.
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Issue an [`Id`] and increment the internal state.
    pub fn issue_id(&self) -> Id {
        let id = self.0.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            panic!("Id maximum reached.");
        }
        Id(id)
    }

    /// Iterator interface of the generator.
    pub fn iter(&mut self) -> IdGeneratorIter<'_> {
        IdGeneratorIter(self)
    }
}

/// Iterator of [`IdGenerator`].
#[derive(Debug)]
pub struct IdGeneratorIter<'a>(&'a IdGenerator);

impl<'a> Iterator for IdGeneratorIter<'a> {
    type Item = Id;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.issue_id())
    }
}

impl std::ops::Deref for Id {
    type Target = u64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
