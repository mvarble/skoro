use skoro_backend::Backend;
use skoro_core::{kind::Kind, pack::Packable};
use skoro_graph::{point_map::PointMap, quantity::QuantityType};
use skoro_operation::Input;

use crate::q::Q;

/// Typed access into a [`PointMap`]: values are stored kind-erased
/// ([`Input`]), and these accessors recover the typed view through a `Q`
/// proxy.
pub trait PointMapAccess<B: Backend> {
    /// The batched value of the quantity, in kind-erased form.
    fn get<T>(&self, tag: &Q<B, T>) -> Option<&Input<B>>;

    /// The concrete quantity type stored for the quantity.
    fn get_quantity_type<T: 'static>(&self, tag: &Q<B, T>) -> Option<&T>;

    /// The value of the quantity, unpacked into vector of specified data type, one instance per
    /// batch element.
    fn get_unpacked<K, T, D>(&self, tag: &Q<B, T>) -> Option<Vec<D>>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static;
}

impl<B: Backend> PointMapAccess<B> for PointMap<B> {
    fn get<T>(&self, tag: &Q<B, T>) -> Option<&Input<B>> {
        self.get(&tag.id).map(|p| p.0)
    }

    fn get_quantity_type<T: 'static>(&self, tag: &Q<B, T>) -> Option<&T> {
        self.get(&tag.id).map(|(_, quantity_type)| {
            quantity_type
                .as_any()
                .downcast_ref()
                .expect("internal: quantity type of a Q proxy should always match")
        })
    }

    fn get_unpacked<K, T, D>(&self, tag: &Q<B, T>) -> Option<Vec<D>>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.get(&tag.id).map(|(input, quantity_type)| {
            let num_components = quantity_type.num_components();
            let point = K::as_point_ref(input)
                .expect("internal: point kind should match the quantity type's kind");
            let data = K::to_elements(point);
            let batches = data.len() / num_components;
            let quantity_type_ref: &T = quantity_type
                .as_any()
                .downcast_ref()
                .expect("internal: quantity type of a Q proxy should always match");
            let mut out = Vec::with_capacity(batches);
            for batch in 0..batches {
                out.push(quantity_type_ref.unpack(&data, batch * num_components));
            }
            out
        })
    }
}

impl<B: Backend, T> Q<B, T> {
    #[inline(always)]
    pub fn get_points<'a>(&self, points: &'a PointMap<B>) -> Option<&'a Input<B>> {
        points.get(&self.id).map(|p| p.0)
    }

    #[inline(always)]
    pub fn get_points_unpacked<K, D>(&self, points: &PointMap<B>) -> Option<Vec<D>>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        points.get_unpacked(self)
    }
}
