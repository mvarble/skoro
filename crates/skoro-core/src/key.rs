use crate::{Address, Segment};

/// A random key: draws are a function of a run's key and a quantity's address alone.
///
/// Every derivation is BLAKE3's keyed hash under the key, with its own domain tag first: `0x01`
/// folds a name, `0x02` an index, `0x03` splits, and `0x04` seeds the stream of variates.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Key([u8; 32]);

const NAME: u8 = 0x01;
const INDEX: u8 = 0x02;
const SPLIT: u8 = 0x03;
const UNIFORMS: u8 = 0x04;

impl Key {
    pub fn from_seed(seed: u64) -> Key {
        Key(blake3::derive_key(
            "skoro 2026-10-15 key from seed",
            &seed.to_le_bytes(),
        ))
    }

    /// Folds one segment into the key: `keyed_hash(self, [0x01] ++ le(len) ++ utf8)` for a name,
    /// `keyed_hash(self, [0x02] ++ le(index))` for an index.
    pub fn fold(&self, segment: &Segment) -> Key {
        let mut hasher = blake3::Hasher::new_keyed(&self.0);
        match segment {
            Segment::Name(name) => {
                hasher.update(&[NAME]);
                hasher.update(&(name.len() as u64).to_le_bytes());
                hasher.update(name.as_bytes());
            }
            Segment::Index(index) => {
                hasher.update(&[INDEX]);
                hasher.update(&index.to_le_bytes());
            }
        }
        Key(*hasher.finalize().as_bytes())
    }

    /// Folds the segments of `address` in order: the key of the quantity at that address.
    pub fn at(&self, address: &Address) -> Key {
        address
            .segments()
            .iter()
            .fold(self.clone(), |key, segment| key.fold(segment))
    }

    /// `n` keys, the `i`-th being `keyed_hash(self, [0x03] ++ le(i))`, whatever `n`.
    pub fn split(&self, n: usize) -> Vec<Key> {
        (0..n as u64)
            .map(|i| {
                let mut hasher = blake3::Hasher::new_keyed(&self.0);
                hasher.update(&[SPLIT]);
                hasher.update(&i.to_le_bytes());
                Key(*hasher.finalize().as_bytes())
            })
            .collect()
    }

    /// `n` variates in the open interval (0, 1), from the extendable output keyed by this key
    /// under tag `0x04`: each takes 8 bytes `u`, little-endian, as `((u >> 11) + 0.5) * 2^-53`.
    pub fn uniforms(&self, n: usize) -> Vec<f64> {
        let mut hasher = blake3::Hasher::new_keyed(&self.0);
        hasher.update(&[UNIFORMS]);
        let mut stream = hasher.finalize_xof();
        let mut bytes = [0u8; 8];
        (0..n)
            .map(|_| {
                stream.fill(&mut bytes);
                let u = u64::from_le_bytes(bytes);
                ((u >> 11) as f64 + 0.5) * f64::EPSILON / 2.0
            })
            .collect()
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(key: &Key) -> String {
        key.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn name_and_integer_segments_differ() {
        let key = Key::from_seed(0);
        assert_ne!(key.fold(&Segment::name("3")), key.fold(&Segment::Index(3)));
    }

    #[test]
    fn segments_fold_one_at_a_time() {
        let key = Key::from_seed(0);
        assert_ne!(key.at(&"ab/c".into()), key.at(&"a/bc".into()));
        assert_ne!(key.at(&"ab".into()), key.at(&"a/b".into()));
        assert_eq!(
            key.at(&"a/b".into()),
            key.fold(&Segment::name("a")).fold(&Segment::name("b"))
        );
        assert_eq!(key.at(&Address::root()), key);
    }

    /// Computed by an independent implementation of the derivation over the `blake3` Python
    /// package, and frozen: a change here changes every draw skoro has made.
    #[test]
    fn fold_matches_the_reference_vectors() {
        let key = Key::from_seed(7);
        assert_eq!(
            hex(&key),
            "bc1c66b5311ef8d53a290916cdd47dfad47bbb3c327c82defbb699c6e6dfbffe"
        );
        let at = key.at(&"40/target/3".into());
        assert_eq!(
            hex(&at),
            "a63a1820cbee4b331f05ee39be8a385a709d0cbddbcfd7b63199fc8b1eac95b0"
        );
        assert_eq!(
            hex(&key.split(2)[1]),
            "ec2fb9bbac1fe9cc1cbdad50c74df4038f3a4d0356492ff4176efc5beeb36982"
        );
        let expected: [f64; 3] = [0.3221308369554478, 0.6715094980977141, 0.7275466618005233];
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&at.uniforms(3)), bits(&expected));
    }

    #[test]
    fn split_keys_are_distinct() {
        let key = Key::from_seed(1);
        let mut keys = key.split(1000);
        assert_eq!(key.split(10), keys[..10]);
        keys.push(key);
        let mut bytes: Vec<_> = keys.iter().map(|k| *k.as_bytes()).collect();
        bytes.sort();
        bytes.dedup();
        assert_eq!(bytes.len(), keys.len());
    }

    #[test]
    fn uniforms_lie_in_the_open_interval() {
        let variates = Key::from_seed(2).uniforms(100_000);
        assert!(variates.iter().all(|&u| 0.0 < u && u < 1.0));
    }

    #[test]
    fn uniforms_have_uniform_moments() {
        let variates = Key::from_seed(3).uniforms(100_000);
        let n = variates.len() as f64;
        let mean = variates.iter().sum::<f64>() / n;
        let variance = variates.iter().map(|u| (u - mean).powi(2)).sum::<f64>() / n;
        // Five standard errors: sqrt(1/12 / n) for the mean, sqrt((1/80 - 1/144) / n) for the
        // variance.
        assert!((mean - 0.5).abs() < 5.0 * (1.0 / 12.0 / n).sqrt(), "{mean}");
        assert!(
            (variance - 1.0 / 12.0).abs() < 5.0 * ((1.0 / 80.0 - 1.0 / 144.0) / n).sqrt(),
            "{variance}"
        );
        // And the first half of the stream is the stream of the first half.
        assert_eq!(Key::from_seed(3).uniforms(10), variates[..10]);
    }
}
