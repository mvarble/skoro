use std::fmt;
use std::sync::Arc;

/// One segment of an address: a name or an index.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Segment {
    Index(u64),
    Name(Arc<str>),
}

impl Segment {
    /// A name, even one that looks like a number; such a name is displayed quoted.
    pub fn name(name: impl Into<Arc<str>>) -> Segment {
        Segment::Name(name.into())
    }
}

impl From<u64> for Segment {
    fn from(index: u64) -> Segment {
        Segment::Index(index)
    }
}

impl From<usize> for Segment {
    fn from(index: usize) -> Segment {
        Segment::Index(index as u64)
    }
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Index(index) => write!(f, "{index}"),
            Segment::Name(name) if parses_as_name(name) => f.write_str(name),
            Segment::Name(name) => write!(f, "{name:?}"),
        }
    }
}

/// Whether `name`, written as it is in a path, parses back to the same single name.
fn parses_as_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '"']) && !name.bytes().all(|b| b.is_ascii_digit())
}

/// Where a quantity lives in a model: a path of segments.
///
/// Addresses order lexicographically by segment, so the addresses under a namespace are
/// contiguous in an ordered map.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Address(Arc<[Segment]>);

impl Address {
    pub fn new(segments: impl IntoIterator<Item = Segment>) -> Address {
        Address(segments.into_iter().collect())
    }

    /// The empty address, under which every other lies.
    pub fn root() -> Address {
        Address::new([])
    }

    pub fn segments(&self) -> &[Segment] {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// This address followed by `rest`.
    pub fn join(&self, rest: &Address) -> Address {
        Address::new(self.0.iter().chain(rest.0.iter()).cloned())
    }

    /// Whether this address lies under `prefix`, or is it.
    pub fn starts_with(&self, prefix: &Address) -> bool {
        self.0.starts_with(&prefix.0)
    }
}

/// A path: segments separated by `/`, a segment of ASCII digits being an index and any other a
/// name. Empty segments are skipped, so `""` is the root. A run of digits too long for a `u64`
/// is a name.
impl From<&str> for Address {
    fn from(path: &str) -> Address {
        Address::new(path.split('/').filter(|s| !s.is_empty()).map(|s| {
            match s.bytes().all(|b| b.is_ascii_digit()).then(|| s.parse()) {
                Some(Ok(index)) => Segment::Index(index),
                _ => Segment::name(s),
            }
        }))
    }
}

impl From<String> for Address {
    fn from(path: String) -> Address {
        Address::from(path.as_str())
    }
}

impl From<Segment> for Address {
    fn from(segment: Segment) -> Address {
        Address::new([segment])
    }
}

impl From<u64> for Address {
    fn from(index: u64) -> Address {
        Address::from(Segment::from(index))
    }
}

impl From<usize> for Address {
    fn from(index: usize) -> Address {
        Address::from(Segment::from(index))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, segment) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("/")?;
            }
            write!(f, "{segment}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_parses_digits_as_indices() {
        let address = Address::from("40/target/3");
        assert_eq!(
            address.segments(),
            [
                Segment::Index(40),
                Segment::name("target"),
                Segment::Index(3)
            ]
        );
        assert_eq!(Address::from(7usize), Address::new([Segment::Index(7)]));
        assert_eq!(
            Address::from("x3/3x").segments(),
            [Segment::name("x3"), Segment::name("3x")]
        );
        assert!(Address::from("").is_root());
        assert_eq!(Address::from("a//b/"), Address::from("a/b"));
        let huge = "99999999999999999999999";
        assert_eq!(Address::from(huge).segments(), [Segment::name(huge)]);
    }

    #[test]
    fn quoted_names_display_distinctly() {
        let name = Address::new([Segment::name("a"), Segment::name("3")]);
        let index = Address::new([Segment::name("a"), Segment::Index(3)]);
        assert_eq!(index.to_string(), "a/3");
        assert_eq!(name.to_string(), "a/\"3\"");
        assert_eq!(Address::new([Segment::name("a/b")]).to_string(), "\"a/b\"");
        assert_eq!(Address::from("40/target/3").to_string(), "40/target/3");
    }

    #[test]
    fn addresses_under_a_namespace_are_contiguous() {
        let mut addresses: Vec<Address> = ["a/2", "b", "a", "a/x/1", "ab", "a/1", "0"]
            .into_iter()
            .map(Address::from)
            .collect();
        addresses.sort();
        let prefix = Address::from("a");
        let under: Vec<bool> = addresses.iter().map(|a| a.starts_with(&prefix)).collect();
        let first = under.iter().position(|&u| u).unwrap();
        let count = under.iter().filter(|&&u| u).count();
        assert!(under[first..first + count].iter().all(|&u| u));
        assert_eq!(count, 4);
        assert_eq!(prefix.join(&"x/1".into()), Address::from("a/x/1"));
    }
}
