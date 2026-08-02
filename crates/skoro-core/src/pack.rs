use skoro_backend::Element;

/// A trait which associates a number of float components with a data type.
pub trait NumComponents {
    fn num_components(&self) -> usize;
}

/// A trait which specifies that data can be represented as a contiguous blob of elements.
pub trait Packable<F: Element, D: Sized + 'static> {
    /// Pack an instance of the data into a buffer.
    fn pack(&self, data: &D, offset: usize, target: &mut [F]);

    /// Unpack a section of a buffer into the data.
    fn unpack(&self, source: &[F], offset: usize) -> D;
}

/// A trait which specifies how much discrete data is used to enumerate all objects in a set. It
/// also helps specify when this number is countably-infinite.
pub trait Cardinality {
    /// Returns `None` for countably-infinite supports (e.g. integers).
    fn cardinality(&self) -> Option<usize>;
}
