use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use skoro_backend::{
    Backend, CotangentFrame, CotangentVector, Point, TangentFrame, TangentVector, Tensor,
};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::{Id, IdGenerator},
    kind::{Kind, QuantityKind},
    kind_erased_tensor::QuantityValue,
    pack::Packable,
    sampler::Sampler,
};
use skoro_operation::{Input, ValueMap, ops::TangentConcat};
use skoro_riemann::euclidean::EuclideanD;
use skoro_riemann::real::Real;

use crate::compiler::{
    CompiledSequence, CompiledStep, ForwardInput, ForwardOptions, ForwardOutput, HvpForwardInput,
    HvpForwardOutput, JvpForwardInput, JvpForwardOutput, VjpForwardInput, VjpForwardOutput,
};
use crate::node::{
    ConstantNode, LogDensityNode, MappedNode, Node, ObservedNode, ParameterNode, StochasticNode,
};
use crate::operation::GraphOperation;
use crate::point_map::PointMap;
use crate::quantity::{DynContinuous, DynQuantityType, QuantityType};
use crate::sampler::{GraphSampler, KindedSampler};

/// A mutable probabilistic graphical model.
///
/// It stores variables, transformations, and their statistical properties.
#[derive(Clone)]
/// The four collections below carry an invariant between them: every id in
/// `log_likelihoods` names a [`Node::LogDensity`] in `nodes`, and `stochastic_to_log_density`
/// is the inverse of the `stochastic_id` those density nodes carry. Only the mutating methods
/// on this type maintain that invariant, so the fields are private and reachable through the
/// read accessors below.
///
/// **The density → stochastic direction is stored on the node, not in a map.**
/// [`LogDensityNode::stochastic_id`] is the single source of truth for which quantity a
/// density scores; [`stochastic_of`](Self::stochastic_of) reads it directly.
/// `stochastic_to_log_density` exists only to make the reverse lookup O(1) and is derived
/// state over that field — which is why every write to one happens beside a write to the
/// other, in the same method.
pub struct Graph<B: Backend> {
    /// Arena of all nodes.
    nodes: HashMap<Id, Node<B>>,
    /// Children edge data (parent edge data is local)
    children: HashMap<Id, HashSet<Id>>,
    /// Log-likelihood terms
    log_likelihoods: Vec<Id>,
    /// Reverse index over [`LogDensityNode::stochastic_id`]: the density scoring a given
    /// stochastic quantity. Derived — see the type-level note above.
    stochastic_to_log_density: HashMap<Id, Id>,
    /// Nodes this graph built for itself and never handed out.
    ///
    /// [`stacked_tangent`](Self::stacked_tangent) is the only writer: its anchors and per-block
    /// tangents are created inside the graph, so no `Q` will ever name one and nothing outside
    /// can be holding one. That makes them reclaimable the moment nothing points at them —
    /// which ordinary nodes are not, since a caller may still hold a childless leaf.
    ///
    /// **Captured, not derived.** The set is written at the moment the nodes are created, when
    /// the fact is known for certain, rather than recovered later by a walk that would have to
    /// guess. That is the same distinction `docs/skoro.md` §4.2 draws for `FactorRecord`: a
    /// derived index needs invalidating across the whole mutation API, a captured one does not.
    ///
    /// It also clones cleanly, unlike the topology `revision` counter `docs/skoro.md` §4.2 item 5
    /// warns must be dropped on clone. The difference is that a revision is only meaningful
    /// against one mutation history, while a private node is private in every copy of the graph
    /// that holds it — the ids come along with the nodes they name.
    private_nodes: HashSet<Id>,
    /// Generates unique IDs for nodes.
    id_generator: IdGenerator,
}

impl<B: Backend> Default for Graph<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> std::fmt::Debug for Graph<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("node_count", &self.nodes.len())
            .finish_non_exhaustive()
    }
}

/// The shared dyn quantity type of all [`LogDensityNode`]s (always real).
static REAL_QUANTITY_TYPE: DynContinuous<Real> = DynContinuous(Real);

/// Whether a parent node feeds a differentiable input to a consuming operation.
///
/// Constants, observations, cached mapped values, and all discrete quantities
/// are non-differentiable inputs.
fn is_differentiable_parent<B: Backend>(node: &Node<B>) -> bool {
    match node {
        Node::Constant(_) | Node::Observed(_) => false,
        Node::Mapped(MappedNode {
            value,
            quantity_type,
            ..
        }) => quantity_type.kind() == QuantityKind::Continuous && value.is_none(),
        Node::LogDensity(LogDensityNode { value, .. }) => value.is_none(),
        Node::Parameter(ParameterNode { quantity_type, .. })
        | Node::Stochastic(StochasticNode { quantity_type, .. }) => {
            quantity_type.kind() == QuantityKind::Continuous
        }
    }
}

/// Extract a continuous evaluation from a forward output, erroring if the node
/// is discrete.
fn take_float<B: Backend>(out: &mut ForwardOutput<B>, id: Id) -> SkoroResult<Point<B>> {
    let (input, _) = out
        .evaluations
        .remove(&id)
        .expect("evaluated node should be present");
    match input {
        Input::Float(tensor) => Ok(tensor),
        Input::Int(_) => Err(SkoroError::NotContinuous { id }),
    }
}

impl<B: Backend> Graph<B> {
    /// Create a new probabilistic execution graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            children: HashMap::new(),
            log_likelihoods: Vec::new(),
            id_generator: IdGenerator::new(),
            stochastic_to_log_density: HashMap::new(),
            private_nodes: HashSet::new(),
        }
    }

    /// The node stored at `id`, if it exists.
    pub fn node(&self, id: Id) -> Option<&Node<B>> {
        self.nodes.get(&id)
    }

    /// Whether a node with this id is present.
    pub fn contains(&self, id: Id) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Every node in the arena, in arbitrary order.
    pub fn nodes(&self) -> impl Iterator<Item = (&Id, &Node<B>)> {
        self.nodes.iter()
    }

    /// The number of nodes in the arena. The bound a retention policy watches.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The ids of every node that consumes `id` as a parent.
    pub fn children_of(&self, id: Id) -> Option<&HashSet<Id>> {
        self.children.get(&id)
    }

    /// The log-density terms that together score the model.
    pub fn log_likelihoods(&self) -> &[Id] {
        &self.log_likelihoods
    }

    /// Whether `id` is a live likelihood term — a density the objective still sums.
    ///
    /// This is the right "is the factor still there?" question, where [`contains`](Self::contains)
    /// is the wrong one: `unregister_density` deliberately leaves a demoted or replaced quantity's
    /// density in the arena, so membership says nothing about whether the term still scores.
    pub fn is_scoring(&self, id: Id) -> bool {
        self.log_likelihoods.contains(&id)
    }

    /// Whether `id` is a node this graph built for itself and never handed out — an anchor or
    /// per-block tangent of a [`stacked_tangent`](Self::stacked_tangent). Such nodes carry no
    /// handle, so a garbage collector may treat a childless one as reclaimable where an ordinary
    /// constant may still be named by an [`Id`] the caller kept.
    pub fn is_private(&self, id: Id) -> bool {
        self.private_nodes.contains(&id)
    }

    /// The log-density scoring `stochastic`, if it has one.
    pub fn density_of(&self, stochastic: Id) -> Option<Id> {
        self.stochastic_to_log_density.get(&stochastic).copied()
    }

    /// The stochastic quantity that `density` scores, if it belongs to one. A
    /// free-standing likelihood factor belongs to none and yields `None`, as does an id that
    /// is not a density at all.
    ///
    /// Read straight off the node rather than from an index, so there is no second copy to
    /// fall out of step with it.
    pub fn stochastic_of(&self, density: Id) -> Option<Id> {
        match self.nodes.get(&density) {
            Some(Node::LogDensity(LogDensityNode { stochastic_id, .. })) => *stochastic_id,
            _ => None,
        }
    }

    /// Density nodes that no longer score anything and are no longer likelihood terms.
    ///
    /// `unregister_density` deliberately leaves the node in the arena — it may still be
    /// referenced as a child, and removing it is the caller's decision — but until now there
    /// was no way to get its [`Id`] back: the tag that named it is dropped when the
    /// distribution is replaced, so the only recourse was scanning `nodes()` by hand.
    ///
    /// The orphan is not inert. It keeps its `parent_ids`, so its parents' `children` sets stay
    /// non-empty and [`remove`](Self::remove) is blocked on the whole cone above it. Pair this
    /// with [`remove_closure`](Self::remove_closure) to reclaim one.
    ///
    /// A caller that tracks handles has a better route and will rarely need this: `skoro-model`'s
    /// collector treats a density that no longer scores as ordinary garbage, so an orphan there is
    /// reclaimed along with the subgraph it was holding up. This stays because the graph itself
    /// knows nothing about handles, and so cannot make that judgement.
    pub fn orphaned_densities(&self) -> Vec<Id> {
        self.nodes
            .iter()
            .filter(|(id, node)| {
                matches!(
                    node,
                    Node::LogDensity(LogDensityNode {
                        stochastic_id: None,
                        ..
                    })
                ) && !self.log_likelihoods.contains(id)
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Remove `roots` together with everything downstream of them.
    ///
    /// [`remove`](Self::remove) is leaf-only and all-or-nothing: it rejects a batch that any
    /// surviving node still points into. That is the right primitive, but it makes every caller
    /// compute the downward closure itself, and retiring a tick of a filter needs exactly that
    /// closure at every retention policy. This is that walk, done once here.
    ///
    /// Returns the removed nodes. The closure includes `roots`; ids not in the arena are
    /// skipped rather than reported, so removing an already-removed subtree is a no-op.
    ///
    /// Afterwards, any [`private_nodes`](Self::private_nodes) entry left pointing at nothing is
    /// reclaimed too, and appears in the returned vector. Those are nodes the graph built for
    /// itself, upstream of what it handed out — the anchors of a
    /// [`stacked_tangent`](Self::stacked_tangent). A downstream walk cannot reach them and no
    /// `Q` names them, so without this they would be stranded for the life of the model, which
    /// is a leak that grows once per tick of a sliding window.
    pub fn remove_closure(&mut self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        let mut closure = HashSet::new();
        let mut queue: VecDeque<Id> = roots
            .iter()
            .copied()
            .filter(|id| self.nodes.contains_key(id))
            .collect();
        while let Some(id) = queue.pop_front() {
            if !closure.insert(id) {
                continue;
            }
            if let Some(children) = self.children.get(&id) {
                queue.extend(children.iter().copied());
            }
        }
        // Children before parents: `remove` checks that nothing outside the batch still names
        // a node, and it takes the batch as one closed set, so any order within it is legal —
        // but keeping the dependency order makes a partial failure easier to reason about.
        let ordered: Vec<Id> = self
            .topological_sort(&closure.iter().copied().collect::<Vec<_>>())
            .into_iter()
            .filter(|id| closure.contains(id))
            .rev()
            .collect();
        let mut removed = self.remove(&ordered)?;
        removed.extend(self.reclaim_private_nodes()?);
        Ok(removed)
    }

    /// Remove every private node nothing points at, to fixpoint.
    ///
    /// **The fixpoint is load-bearing, not defensive.** A private anchor is the parent of a
    /// private per-block tangent, so it is not childless until that tangent has gone. One pass
    /// over a `HashSet` would reclaim the anchor never, sometimes, or always depending on
    /// iteration order — the same intermittent shape as the dependency-order requirement on
    /// `check_cache`, and just as invisible in a single-member test.
    fn reclaim_private_nodes(&mut self) -> SkoroResult<Vec<Node<B>>> {
        let mut removed = Vec::new();
        loop {
            let orphans: Vec<Id> = self
                .private_nodes
                .iter()
                .copied()
                .filter(|id| {
                    self.nodes.contains_key(id)
                        && self.children.get(id).is_none_or(|kids| kids.is_empty())
                })
                .collect();
            if orphans.is_empty() {
                return Ok(removed);
            }
            removed.extend(self.remove(&orphans)?);
        }
    }

    /// Recompute every derived collection from `nodes` and report the first disagreement.
    ///
    /// `nodes` is the source of truth. `children`, `log_likelihoods` and
    /// `stochastic_to_log_density` are all derived from it and are maintained by hand across a
    /// dozen mutating methods, which is exactly the arrangement that historically drifts. This
    /// turns "the indices may leak" from a standing comment into something a test can fail on.
    ///
    /// Checked here:
    ///
    /// - `children` is the exact reverse of every node's `parent_ids`, with no empty sets left
    ///   behind;
    /// - every id in `log_likelihoods` names a [`Node::LogDensity`], and names it once;
    /// - `stochastic_to_log_density` is the exact inverse of the density nodes'
    ///   [`stochastic_id`](LogDensityNode::stochastic_id);
    /// - a density that scores a quantity is among the likelihood terms. The converse does not
    ///   hold: a free-standing likelihood factor scores no particular quantity;
    /// - every id in `private_nodes` still names a node. That set is *captured* rather than
    ///   derived, so nothing recomputes it — but a stale id in it would make
    ///   `reclaim_private_nodes` reach for a node that is gone, and, worse, would claim a
    ///   reissued id as the graph's to delete.
    ///
    /// Runs after every mutation in debug builds, and is public so a test can assert it
    /// directly after a sequence the mutators do not cover.
    pub fn validate_indices(&self) -> Result<(), String> {
        let mut derived: HashMap<Id, HashSet<Id>> = HashMap::new();
        for (id, node) in &self.nodes {
            for parent in node.parent_ids() {
                derived.entry(*parent).or_default().insert(*id);
            }
        }
        for (parent, actual) in &self.children {
            match derived.get(parent) {
                Some(expected) if expected == actual => {}
                Some(expected) => {
                    return Err(format!(
                        "children[{parent:?}] is {actual:?}, but parent_ids gives {expected:?}"
                    ));
                }
                None => {
                    return Err(format!(
                        "children[{parent:?}] is {actual:?}, but no node names {parent:?} as a parent"
                    ));
                }
            }
        }
        for (parent, expected) in &derived {
            if !self.children.contains_key(parent) {
                return Err(format!(
                    "children has no entry for {parent:?}, but {expected:?} name it as a parent"
                ));
            }
        }

        for (i, id) in self.log_likelihoods.iter().enumerate() {
            if !matches!(self.nodes.get(id), Some(Node::LogDensity(_))) {
                return Err(format!(
                    "log_likelihoods[{i}] = {id:?} does not name a LogDensity node"
                ));
            }
            if self.log_likelihoods[..i].contains(id) {
                return Err(format!("log_likelihoods names {id:?} more than once"));
            }
        }

        for (stochastic, density) in &self.stochastic_to_log_density {
            match self.nodes.get(density) {
                Some(Node::LogDensity(LogDensityNode {
                    stochastic_id: Some(claimed),
                    ..
                })) if claimed == stochastic => {}
                Some(Node::LogDensity(LogDensityNode { stochastic_id, .. })) => {
                    return Err(format!(
                        "stochastic_to_log_density[{stochastic:?}] = {density:?}, but that node claims {stochastic_id:?}"
                    ));
                }
                _ => {
                    return Err(format!(
                        "stochastic_to_log_density[{stochastic:?}] = {density:?}, which is not a LogDensity node"
                    ));
                }
            }
        }
        for (id, node) in &self.nodes {
            let Node::LogDensity(LogDensityNode {
                stochastic_id: Some(stochastic),
                ..
            }) = node
            else {
                continue;
            };
            if self.stochastic_to_log_density.get(stochastic) != Some(id) {
                return Err(format!(
                    "{id:?} scores {stochastic:?}, but stochastic_to_log_density does not point back at it"
                ));
            }
            if !self.log_likelihoods.contains(id) {
                return Err(format!(
                    "{id:?} scores {stochastic:?} but is not among the likelihood terms"
                ));
            }
        }

        for id in &self.private_nodes {
            if !self.nodes.contains_key(id) {
                return Err(format!("private_nodes names {id:?}, which is not a node"));
            }
        }

        Ok(())
    }

    /// [`validate_indices`](Self::validate_indices) as a debug assertion.
    ///
    /// Compiled away entirely in release. Every mutating method ends with a call to this, so a
    /// method that leaves an index inconsistent fails the test that exercised it rather than
    /// the unrelated one that reads the index later.
    #[inline]
    fn debug_validate_indices(&self) {
        #[cfg(debug_assertions)]
        if let Err(disagreement) = self.validate_indices() {
            panic!("graph index invariant violated: {disagreement}");
        }
    }

    /// Add a fixed, immutable quantity, continuous or discrete.
    pub fn add_constant<K, T, D>(&mut self, quantity_type: T, value: &D) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let value = K::wrap_unbatched_point(quantity_type.pack_value(value));
        self.add_constant_impl(quantity_type.into_dyn(), QuantityValue::Unbatched(value))
    }

    /// Add a fixed, immutable quantity that **differs across the batch**.
    ///
    /// One value per batch row, in order, rather than one value broadcast to every row. A batch
    /// is a set of independent problems (independent optimizations, independent conditioning),
    /// and this is how each row gets its own constant.
    ///
    /// The cost is that the value pins the batch: a graph containing a batch-`n` constant can
    /// only be evaluated at batch `n`, and any other batch size is a
    /// [`SkoroError::BatchSizeMismatch`] rather than a broadcast. Use
    /// [`add_constant`](Self::add_constant) for a value shared by every row.
    pub fn add_batched_constant<K, T, D>(&mut self, quantity_type: T, values: &[D]) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let value = K::wrap_point(quantity_type.pack_values(values));
        self.add_constant_impl(quantity_type.into_dyn(), QuantityValue::Batched(value))
    }

    /// [`add_batched_constant`](Self::add_batched_constant) over an already-erased quantity type
    /// and an already-packed value.
    ///
    /// The entry point for callers holding a quantity type only as `dyn` -- freezing a snapshot of
    /// a quantity whose manifold is not known statically, for instance.
    pub fn add_batched_constant_dyn(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        value: Input<B>,
    ) -> Id {
        self.add_constant_impl(quantity_type, QuantityValue::Batched(value))
    }

    fn add_constant_impl(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        value: QuantityValue<B>,
    ) -> Id {
        let id = self.id_generator.issue_id();
        self.nodes.insert(
            id,
            Node::Constant(ConstantNode {
                id,
                value,
                quantity_type,
            }),
        );
        self.debug_validate_indices();
        id
    }

    /// Add a free quantity, intended to be inferred, continuous or discrete.
    pub fn add_parameter<K, T>(&mut self, quantity_type: T) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.add_parameter_impl(quantity_type.into_dyn())
    }

    fn add_parameter_impl(&mut self, quantity_type: Arc<dyn DynQuantityType<B>>) -> Id {
        let id = self.id_generator.issue_id();
        self.nodes
            .insert(id, Node::Parameter(ParameterNode { id, quantity_type }));
        self.debug_validate_indices();
        id
    }

    /// Adds a stochastic quantity, continuous or discrete.
    ///
    /// The quantity type and the sampler must agree on the kind `K` (checked
    /// statically by the shared kind parameter).
    ///
    /// Returns an error if any of the supplied `parent_ids` does not exist.
    pub fn add_stochastic<K, T, S>(
        &mut self,
        quantity_type: T,
        sampler: S,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
        S: Sampler<B, K>,
    {
        self.add_stochastic_impl(
            quantity_type.into_dyn(),
            Arc::new(KindedSampler::<B, K, S>::new(sampler)),
            parent_ids,
        )
    }

    fn add_stochastic_impl(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        sampler: Arc<dyn GraphSampler<B>>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id> {
        let id = self.id_generator.issue_id();
        self.register_children(id, &parent_ids)?;
        self.nodes.insert(
            id,
            Node::Stochastic(StochasticNode {
                id,
                sampler,
                parent_ids,
                quantity_type,
            }),
        );
        self.debug_validate_indices();
        Ok(id)
    }

    /// Add a transformation of other quantities, continuous or discrete.
    ///
    /// Differentiable operations ([`DiffKernel`](skoro_diff::diff::DiffKernel))
    /// and forward-only discrete operations ([`ForwardOnly`](crate::operation::ForwardOnly))
    /// are both accepted through the unified [`GraphOperation`] trait.
    /// Errors with [`SkoroError::DanglingParent`] if any parent is not in the graph.
    pub fn add_mapped<K, T>(
        &mut self,
        quantity_type: T,
        op: impl GraphOperation<B>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.add_mapped_impl(quantity_type.into_dyn(), Arc::new(op), parent_ids)
    }

    /// [`add_mapped`](Self::add_mapped) over an already-erased quantity type and operation.
    ///
    /// The entry point for a caller building nodes from ids alone, where neither the manifold nor
    /// the operation has a name at the call site -- a factor over a set of quantities whose
    /// membership is only known at run time. Pairs with
    /// [`inverse_retraction_op`](Self::inverse_retraction_op).
    pub fn add_mapped_dyn(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        op: Box<dyn GraphOperation<B>>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id> {
        self.add_mapped_impl(quantity_type, Arc::from(op), parent_ids)
    }

    /// The manifold dimension of `id` and its inverse-retraction, as an erased operation.
    ///
    /// Both halves are needed together to build `x ⊖ p` from an id: the operation to register, and
    /// the dimension, which is the width of the resulting tangent -- and which is the *manifold*
    /// dimension, not the storage size, so an `SE3` quantity reports 6 rather than 12.
    ///
    /// Errors [`SkoroError::NotContinuous`] for a discrete quantity, which has no tangent space,
    /// and [`SkoroError::NodeNotFound`] for an unknown id.
    pub fn inverse_retraction_op(
        &self,
        id: Id,
    ) -> SkoroResult<(usize, Box<dyn GraphOperation<B>>)> {
        let quantity_type = self.get_quantity_type_impl(id)?;
        let op = quantity_type
            .inverse_retraction_op()
            .ok_or(SkoroError::NotContinuous { id })?;
        let dimension = quantity_type
            .as_manifold()
            .ok_or(SkoroError::NotContinuous { id })?
            .dimension();
        Ok((dimension, op))
    }

    /// The stacked tangent `δ = [x₁ ⊖ p₁ ; … ; x_k ⊖ p_k]` over a heterogeneous set of
    /// quantities, as one `EuclideanD(Σ dᵢ)` node.
    ///
    /// This is what lets a single factor score *several* quantities at once, which a joint
    /// likelihood over a Markov blanket needs and which no distribution can express -- a
    /// [`Distribution`](skoro_core) is over one quantity type. It is deliberately **not** a
    /// product manifold: the members keep their own ids, their own storage, and their own
    /// children, and only their *tangents* are stacked. A product would have to re-home them into
    /// one merged quantity, which is exactly what `demote_to_parameter` exists to avoid.
    ///
    /// Each `pᵢ` is frozen as a batch-pinned `Constant` of `xᵢ`'s own quantity type, taken from
    /// `at`. That is the point of the call rather than an implementation detail: a factor
    /// linearized at an estimate must not follow the quantity it summarizes, so the anchor is a
    /// snapshot and never a live node. Because the anchors are batch-pinned, a batch of `N`
    /// independent problems carries `N` independent linearization points -- and the graph is
    /// thereafter pinned to that batch.
    ///
    /// Widths are *manifold dimensions*: an `SE3` member contributes 6, not 12. The block order is
    /// the caller's, and is the order any matrix expressed in these coordinates must agree with.
    ///
    /// **Retiring one of these is a `remove_closure` at a scored quantity.** The `2k` internals
    /// are not reachable by a downstream walk — the anchors are upstream of everything, and a
    /// surviving member's per-block tangent is upstream of the concatenation — and no
    /// [`Q`](skoro_model) names them, so `cleanup()` would never queue them either. They are
    /// recorded in `private_nodes` instead, and
    /// [`remove_closure`](Self::remove_closure) reclaims each one as soon as nothing points at
    /// it. Only the concatenation is handed back, and it is deliberately *not* private: it is
    /// the node the caller holds.
    pub fn stacked_tangent(&mut self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Id> {
        if ids.is_empty() {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        }
        let mut widths = Vec::with_capacity(ids.len());
        let mut blocks = Vec::with_capacity(ids.len());
        for &id in ids {
            let (dimension, op) = self.inverse_retraction_op(id)?;
            let quantity_type = self.get_quantity_type_impl(id)?.clone_arc();
            let (value, _) = at.get(&id).ok_or(SkoroError::MissingPoint { id })?;
            let anchor = self.add_batched_constant_dyn(quantity_type, value.clone());
            // `inverse_retract(point, other)` yields `other ⊖ point`, so the anchor goes first:
            // the residual is the displacement *from* the linearization point.
            let block =
                self.add_mapped_dyn(EuclideanD(dimension).into_dyn(), op, vec![anchor, id])?;
            // Neither is ever handed out, so both are the graph's to reclaim. The
            // concatenation below is the caller's and stays out of the set.
            self.private_nodes.insert(anchor);
            self.private_nodes.insert(block);
            widths.push(dimension);
            blocks.push(block);
        }
        let total = widths.iter().sum();
        self.add_mapped_dyn(
            EuclideanD(total).into_dyn(),
            Box::new(TangentConcat(widths)),
            blocks,
        )
    }

    fn add_mapped_impl(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        op: Arc<dyn GraphOperation<B>>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id> {
        let id = self.id_generator.issue_id();
        self.register_children(id, &parent_ids)?;
        let value = self.check_cache(&parent_ids, &op)?;
        self.nodes.insert(
            id,
            Node::Mapped(MappedNode {
                id,
                op,
                parent_ids,
                quantity_type,
                value,
            }),
        );
        self.debug_validate_indices();
        Ok(id)
    }

    /// Mark a mapped quantity as a density for the stochastic quantity.
    pub fn mark_density(&mut self, id: Id, stochastic_id: Id) -> SkoroResult<()> {
        self.mark_likelihood_impl(id, Some(stochastic_id))
    }

    /// Mark a mapped quantity as a likelihood.
    pub fn mark_likelihood(&mut self, id: Id) -> SkoroResult<()> {
        self.mark_likelihood_impl(id, None)
    }

    fn mark_likelihood_impl(&mut self, id: Id, stochastic_id: Option<Id>) -> SkoroResult<()> {
        // The two density maps are inverses of one another, so both directions have to be
        // checked before anything is written:
        //
        // - **one density per quantity** — the reading [`SkoroError::DuplicateDensity`] names,
        //   and the one that was missing. The old guard keyed on the *density* id, so it never
        //   fired when a second density claimed an already-scored quantity. The `Mapped` arm
        //   below had no guard at all, and that is the arm a freshly built density arrives
        //   through.
        // - **one quantity per density** — rebinding an existing density to a different
        //   quantity would strand the old `stochastic_to_log_density` entry, since nothing
        //   removes it.
        //
        // Checked here rather than inside the arms so a rejected call leaves the graph
        // untouched, with no node to re-insert on the way out. The second direction reads the
        // node, which is still in the arena at this point — `stochastic_of` would do as well,
        // but this keeps both halves of the invariant visible side by side.
        if let Some(stochastic_id) = stochastic_id {
            if let Some(existing) = self.stochastic_to_log_density.get(&stochastic_id)
                && *existing != id
            {
                return Err(SkoroError::DuplicateDensity { stochastic_id });
            }
            if let Some(Node::LogDensity(LogDensityNode {
                stochastic_id: Some(existing),
                ..
            })) = self.nodes.get(&id)
                && *existing != stochastic_id
            {
                return Err(SkoroError::DuplicateDensity { stochastic_id });
            }
        }
        match self.nodes.remove(&id) {
            Some(Node::Mapped(MappedNode {
                id,
                op,
                parent_ids,
                quantity_type,
                value,
            })) => {
                if quantity_type.as_any().downcast_ref::<Real>().is_none() {
                    // re-insert the node we removed before erroring out
                    self.nodes.insert(
                        id,
                        Node::Mapped(MappedNode {
                            id,
                            op,
                            parent_ids,
                            quantity_type,
                            value,
                        }),
                    );
                    return Err(SkoroError::NotReal { id });
                }
                // Promoted unbound: `link_density` below writes the binding and both derived
                // collections together, for this arm and the next alike.
                self.nodes.insert(
                    id,
                    Node::LogDensity(LogDensityNode {
                        id,
                        op,
                        stochastic_id: None,
                        parent_ids,
                        value,
                    }),
                );
            }
            // Already a density — re-marking only changes what it is bound to.
            Some(node @ Node::LogDensity(_)) => {
                self.nodes.insert(id, node);
            }
            Some(node) => {
                // re-insert the node we removed before erroring
                self.nodes.insert(id, node);
                return Err(SkoroError::NotMapped { id });
            }
            None => return Err(SkoroError::NodeNotFound { id }),
        }
        self.link_density(id, stochastic_id);
        self.debug_validate_indices();
        Ok(())
    }

    /// Impose a posterior by observing a stochastic quantity at a specified value.
    ///
    /// Works for continuous and discrete stochastic quantities alike; the value
    /// is packed into the element type of the quantity's kind.
    ///
    /// Returns an error if the node does not exist, is not stochastic, or the
    /// supplied quantity type does not match.
    pub fn observe<K, T, D>(&mut self, id: Id, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        // validate the node kind and quantity type before mutating anything
        let already_observed = match self.nodes.get(&id) {
            Some(Node::Stochastic(StochasticNode { quantity_type, .. })) => {
                if quantity_type.as_any().downcast_ref::<T>().is_none() {
                    return Err(SkoroError::QuantityTypeMismatch { id });
                }
                false
            }
            Some(Node::Observed(ObservedNode { quantity_type, .. })) => {
                if quantity_type.as_any().downcast_ref::<T>().is_none() {
                    return Err(SkoroError::QuantityTypeMismatch { id });
                }
                true
            }
            Some(_) => return Err(SkoroError::NotStochastic { id }),
            None => return Err(SkoroError::NodeNotFound { id }),
        };
        match self.nodes.remove(&id) {
            Some(Node::Stochastic(StochasticNode {
                id,
                sampler,
                parent_ids,
                quantity_type,
            }))
            | Some(Node::Observed(ObservedNode {
                id,
                sampler,
                parent_ids,
                quantity_type,
                ..
            })) => {
                // pack the concrete value into storage form (type checked above)
                let value = {
                    let quantity_type_ref = quantity_type
                        .as_any()
                        .downcast_ref::<T>()
                        .expect("internal: quantity type validated above");
                    QuantityValue::Unbatched(K::wrap_unbatched_point(
                        quantity_type_ref.pack_value(value),
                    ))
                };

                // update the stochastic node to now be observed
                self.nodes.insert(
                    id,
                    Node::Observed(ObservedNode {
                        id,
                        value,
                        sampler,
                        parent_ids,
                        quantity_type,
                    }),
                );
            }
            _ => unreachable!(),
        };

        if already_observed {
            return Ok(());
        }

        // Fold every descendant that has become concrete, each exactly once, in an order
        // where a node is reached only after every parent of it has settled.
        //
        // **A plain `visited` set is not enough here**, which is the difference between this
        // walk and `clear_cached_descendants`. Clearing is order-independent and idempotent,
        // so that walk can skip anything it has already seen. Folding is neither: a node whose
        // parents lie on paths of different lengths can be dequeued *before* the last of them
        // folds, and refusing to revisit it would leave it unfolded forever. Ordering the walk
        // by dependency removes both the quadratic re-enqueueing and that hazard at once.
        for descendant in self.descendants_in_dependency_order(id) {
            match self.nodes.remove(&descendant) {
                // Already folded. Nothing an `observe` does can make a concrete node
                // non-concrete, so this stays true and the recompute is pure waste.
                Some(
                    node @ (Node::Mapped(MappedNode { value: Some(_), .. })
                    | Node::LogDensity(LogDensityNode { value: Some(_), .. })),
                ) => {
                    self.nodes.insert(descendant, node);
                }
                Some(Node::Mapped(mapped)) => {
                    let value = self.check_cache(&mapped.parent_ids, &mapped.op)?;
                    self.nodes
                        .insert(descendant, Node::Mapped(MappedNode { value, ..mapped }));
                }
                Some(Node::LogDensity(dens)) => {
                    let value = self.check_cache(&dens.parent_ids, &dens.op)?;
                    self.nodes.insert(
                        descendant,
                        Node::LogDensity(LogDensityNode { value, ..dens }),
                    );
                }
                // A leaf descendant folds nothing; put it back untouched.
                Some(other) => {
                    self.nodes.insert(descendant, other);
                }
                None => {}
            }
        }
        self.debug_validate_indices();
        Ok(())
    }

    /// Every descendant of `id`, ordered so that each follows all of its parents.
    ///
    /// Reachability first, then [`topological_sort`](Self::topological_sort) over that set —
    /// which orders *ancestors* first, so filtering its output back down to the reachable set
    /// gives exactly the dependency order the fold walk needs.
    fn descendants_in_dependency_order(&self, id: Id) -> Vec<Id> {
        let Some(children) = self.children.get(&id) else {
            return Vec::new();
        };
        let mut reachable = HashSet::new();
        let mut queue: VecDeque<Id> = children.iter().copied().collect();
        while let Some(next) = queue.pop_front() {
            if !reachable.insert(next) {
                continue;
            }
            if let Some(children) = self.children.get(&next) {
                queue.extend(children.iter().copied());
            }
        }
        let targets: Vec<Id> = reachable.iter().copied().collect();
        self.topological_sort(&targets)
            .into_iter()
            .filter(|id| reachable.contains(id))
            .collect()
    }

    /// Undo the effects of [`observe`](Self::observe).
    ///
    /// Returns an error if the node does not exist or is not stochastic.
    pub fn unobserve(&mut self, id: Id) -> SkoroResult<()> {
        match self.nodes.get(&id) {
            Some(Node::Observed(_)) => {}
            // Already unobserved; nothing to undo.
            Some(Node::Stochastic(_)) => {
                return Ok(());
            }
            Some(_) => return Err(SkoroError::NotStochastic { id }),
            None => return Err(SkoroError::NodeNotFound { id }),
        };
        let Some(Node::Observed(ObservedNode {
            id,
            sampler,
            parent_ids,
            quantity_type,
            ..
        })) = self.nodes.remove(&id)
        else {
            unreachable!()
        };
        self.nodes.insert(
            id,
            Node::Stochastic(StochasticNode {
                id,
                sampler,
                parent_ids,
                quantity_type,
            }),
        );
        self.clear_cached_descendants(id);
        self.debug_validate_indices();
        Ok(())
    }

    /// Drop the constant-folded values cached on every descendant of `id`.
    ///
    /// [`add_mapped`](Self::add_mapped) folds a subtree into
    /// [`MappedNode::value`](crate::node::MappedNode::value) when all of its parents
    /// are already concrete, and `is_differentiable_parent` treats such a node as
    /// non-differentiable. Any mutation that makes a node's value non-constant —
    /// [`unobserve`](Self::unobserve), [`demote_to_parameter`](Self::demote_to_parameter) —
    /// must therefore invalidate those folds, or the descendants are silently dropped
    /// from the tape and the derivatives come back wrong while the values stay right.
    fn clear_cached_descendants(&mut self, id: Id) {
        let Some(children) = self.children.get(&id) else {
            return;
        };
        let mut visit: VecDeque<Id> = children.iter().copied().collect();
        let mut visited = HashSet::new();
        while let Some(cid) = visit.pop_front() {
            visited.insert(cid);
            match self.nodes.get_mut(&cid) {
                Some(Node::Mapped(MappedNode { value, .. }))
                | Some(Node::LogDensity(LogDensityNode { value, .. }))
                    if value.is_some() =>
                {
                    value.take();
                    if let Some(children) = self.children.get(&cid) {
                        for cid in children {
                            if !visited.contains(cid) {
                                visit.push_back(*cid);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Update a stochastic node's sampling and density mechanisms.
    ///
    /// **Returns the parents it severed** — the old ones the new distribution did not keep. See
    /// [`demote_to_parameter`](Self::demote_to_parameter) for why they are reported rather than
    /// acted on.
    ///
    /// Returns an error if the node does not exist, is not stochastic, the
    /// sampler's kind does not match the node's, a new parent is dangling, or
    /// the update would introduce a cycle.
    pub fn update_stochastic<K, S>(
        &mut self,
        id: Id,
        new_sampler: S,
        new_parents: Vec<Id>,
    ) -> SkoroResult<Vec<Id>>
    where
        K: Kind<B>,
        S: Sampler<B, K>,
    {
        // the sampler's kind must match the node's quantity kind
        match self.nodes.get(&id) {
            Some(Node::Stochastic(StochasticNode { quantity_type, .. })) => {
                if quantity_type.kind() != K::KIND {
                    return Err(SkoroError::QuantityTypeMismatch { id });
                }
            }
            None => return Err(SkoroError::NodeNotFound { id }),
            _ => return Err(SkoroError::NotStochastic { id }),
        }
        self.update_stochastic_impl(
            id,
            Arc::new(KindedSampler::<B, K, S>::new(new_sampler)),
            new_parents,
        )
    }

    fn update_stochastic_impl(
        &mut self,
        id: Id,
        new_sampler: Arc<dyn GraphSampler<B>>,
        new_parents: Vec<Id>,
    ) -> SkoroResult<Vec<Id>> {
        // check that all the new parents are legal
        for &parent_id in &new_parents {
            if !self.nodes.contains_key(&parent_id) {
                return Err(SkoroError::DanglingParent { parent_id });
            }
        }

        // check if this creates cycle; error if so.
        let mut visit: VecDeque<Id> = new_parents.iter().copied().collect();
        let mut visited = HashSet::new();
        while let Some(pid) = visit.pop_front() {
            if pid == id {
                return Err(SkoroError::Cycle { id });
            }
            visited.insert(pid);
            if let Some(node) = self.nodes.get(&pid) {
                for &parent in node.parent_ids() {
                    if !visited.contains(&parent) {
                        visit.push_back(parent);
                    }
                }
            }
        }

        // Swap the edges. Reading the old parents first confines this to the node's own
        // edges; the previous version swept every children set in the graph to find them.
        let Some(Node::Stochastic(stoch)) = self.nodes.get(&id) else {
            unreachable!("node kind validated above");
        };
        let old_parents = stoch.parent_ids.clone();
        self.unregister_children(id, &old_parents);
        self.register_children(id, &new_parents)?;

        // update the node (kind validated above)
        if let Some(Node::Stochastic(stoch)) = self.nodes.get_mut(&id) {
            stoch.sampler = new_sampler;
            stoch.parent_ids = new_parents.clone();
        } else {
            unreachable!("node kind validated above");
        }
        let orphaned_density = self.unregister_density(id);
        self.debug_validate_indices();
        // The parents the swap dropped, less any the new distribution kept — only a node that
        // ended the call with one fewer child needs reconsidering — plus the density the old
        // distribution owned, which now scores nothing and holds that whole subgraph up.
        Ok(old_parents
            .into_iter()
            .filter(|old| !new_parents.contains(old))
            .chain(orphaned_density)
            .collect())
    }

    /// Unregister the log-density owned by a stochastic quantity, dropping it from the
    /// model's likelihood terms.
    ///
    /// The [`LogDensity`](Node::LogDensity) node itself is left in the arena: it may still be
    /// referenced as a child, and removing it is the caller's decision. **Its id is returned**,
    /// because orphaning it is a reachability change nothing else announces — it scores nothing
    /// afterwards, so it and the subgraph it holds up are garbage unless something still names
    /// them, which only a caller tracking handles can decide.
    ///
    /// The orphan's own [`stochastic_id`](LogDensityNode::stochastic_id) is cleared along with
    /// the maps, because [`remove`](Self::remove) reads that field to decide which map entries
    /// to drop. Leaving it set is not a cosmetic leak: if a *different* density is registered
    /// against the same stochastic quantity before the orphan is removed — which is exactly
    /// what "replace the distribution, then clean up the orphan" does — then removing the
    /// orphan would delete the live density's entry, and `density_of` would report `None` for
    /// a quantity that still has a density in `log_likelihoods`.
    fn unregister_density(&mut self, stochastic_id: Id) -> Option<Id> {
        let density = self
            .stochastic_to_log_density
            .get(&stochastic_id)
            .copied()?;
        // Retirement is "unbind, then stop scoring": `link_density(_, None)` releases the
        // reverse entry and clears the node's own field, and `drop_density_term` takes it out
        // of the sum. Neither collection is touched outside those two helpers.
        self.link_density(density, None);
        self.drop_density_term(density);
        Some(density)
    }

    /// Unbind and de-list a density, addressed **by the density's own id**.
    ///
    /// The density-keyed sibling of the stochastic-keyed `unregister_density`, which finds
    /// its target through the quantity being scored. Needed when the caller knows the
    /// density and the stochastic is irrelevant or already gone — a recorded factor whose
    /// residual was removed, say, where the factor's density must leave the objective in the
    /// same breath or `J` and the log-density would disagree about the model.
    ///
    /// Like `unregister_density`, the node itself is left in the arena: removing it is the
    /// caller's decision.
    pub fn unregister_density_node(&mut self, density: Id) {
        self.link_density(density, None);
        self.drop_density_term(density);
        self.debug_validate_indices();
    }

    /// Demote a stochastic quantity to a free [`Parameter`](Node::Parameter), at the
    /// same [`Id`].
    ///
    /// The node keeps its identity and its quantity type, and every edge from a child into it is
    /// preserved, so a referencing tag `Q` stays valid and everything downstream is untouched.
    /// What it loses is its sampler, its parent edges, and its own log-density.
    ///
    /// This is how a quantity is re-rooted when its marginal does not factorize: the joint is
    /// attached separately, as a likelihood factor over the demoted quantities via
    /// [`mark_likelihood`](Self::mark_likelihood). Where the marginal *does* factorize, prefer
    /// [`update_stochastic`](Self::update_stochastic), which keeps the quantity stochastic and
    /// gives it a real distribution.
    ///
    /// Demoting a node that is already a parameter is a no-op.
    ///
    /// **Returns the parents it severed.** Those nodes each just lost a child and may now be
    /// unreachable, which only a caller that tracks external handles can decide — so the ids
    /// are reported rather than acted on here. Dropping them on the floor is a leak that grows
    /// once per demotion, which is once per tick of a sliding window.
    ///
    /// Returns [`SkoroError::NodeNotFound`] if the node does not exist, or
    /// [`SkoroError::NotStochastic`] if it is neither stochastic nor already a parameter.
    pub fn demote_to_parameter(&mut self, id: Id) -> SkoroResult<Vec<Id>> {
        match self.nodes.get(&id) {
            Some(Node::Stochastic(_)) => {}
            // Already a free variable; nothing to demote.
            Some(Node::Parameter(_)) => return Ok(Vec::new()),
            Some(_) => return Err(SkoroError::NotStochastic { id }),
            None => return Err(SkoroError::NodeNotFound { id }),
        };
        let Some(Node::Stochastic(StochasticNode {
            id,
            parent_ids,
            quantity_type,
            ..
        })) = self.nodes.remove(&id)
        else {
            unreachable!("node kind validated above");
        };

        // Drop the edges from the old distribution's parameters into this node.
        self.unregister_children(id, &parent_ids);

        self.nodes
            .insert(id, Node::Parameter(ParameterNode { id, quantity_type }));
        let mut orphaned = parent_ids;
        orphaned.extend(self.unregister_density(id));
        // Defensive, and cheap: today `check_cache` folds nothing downstream of a
        // stochastic node (it yields `None` for both `Stochastic` and `Parameter`
        // parents), so this finds nothing to clear. It guards the invariant rather
        // than a live failure — the node's value is no longer determined by its
        // parents, and a fold rule that ever admitted stochastic parents would make
        // every descendant stale.
        self.clear_cached_descendants(id);
        self.debug_validate_indices();
        Ok(orphaned)
    }

    /// Removes leaf nodes, verifying they have no children which would be kept in the graph.
    ///
    /// Returns an error if a node does not exist, or is still referenced as a
    /// parent of a node that is not also being removed.
    pub fn remove(&mut self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        // Validate no dangling pointers.
        for id in ids {
            if let Some(children) = self.children.get(id)
                && let Some(child_id) = children.iter().find(|id| !ids.contains(id))
            {
                return Err(SkoroError::NodeStillReferenced {
                    id: *id,
                    child_id: *child_id,
                });
            }
        }
        // Perform removal.
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            // Not a parent-edge write: this drops the departing node's *own* children set,
            // which the loop above already proved is empty of survivors. Edges into this node
            // from its parents are `unregister_children`'s job, below.
            self.children.remove(id);
            let Some(node) = self.nodes.remove(id) else {
                return Err(SkoroError::NodeNotFound { id: *id });
            };
            self.private_nodes.remove(id);
            let parent_ids = node.parent_ids().to_vec();
            self.unregister_children(*id, &parent_ids);
            if let Node::LogDensity(LogDensityNode { stochastic_id, .. }) = &node {
                self.forget_density_node(*id, *stochastic_id);
            }
            out.push(node);
        }
        self.debug_validate_indices();
        Ok(out)
    }

    /// Get the quantity type associated with a node by its Id.
    ///
    /// Returns [`SkoroError::NodeNotFound`] if the node does not exist, or
    /// [`SkoroError::QuantityTypeMismatch`] if the requested type does not match
    /// the type stored in the graph.
    pub fn get_quantity_type<K, T>(&self, id: Id) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.get_quantity_type_impl(id)?
            .as_any()
            .downcast_ref::<T>()
            .cloned()
            .ok_or(SkoroError::QuantityTypeMismatch { id })
    }

    fn get_quantity_type_impl(&self, id: Id) -> SkoroResult<&dyn DynQuantityType<B>> {
        let node = self.nodes.get(&id).ok_or(SkoroError::NodeNotFound { id })?;
        match node {
            Node::Constant(ConstantNode { quantity_type, .. })
            | Node::Parameter(ParameterNode { quantity_type, .. })
            | Node::Mapped(MappedNode { quantity_type, .. })
            | Node::Stochastic(StochasticNode { quantity_type, .. })
            | Node::Observed(ObservedNode { quantity_type, .. }) => Ok(quantity_type.as_ref()),
            Node::LogDensity(_) => Ok(&REAL_QUANTITY_TYPE),
        }
    }

    /// The manifold structure of a continuous node, for tangent-space sizing.
    ///
    /// [`Node::LogDensity`] has no stored quantity type but is always real-valued, so it
    /// reports dimension 1 — which is what lets a layout name density nodes as rows.
    pub fn get_manifold(&self, id: Id) -> SkoroResult<&dyn skoro_riemann::DynManifold<B>> {
        self.get_quantity_type_impl(id)?
            .as_manifold()
            .ok_or(SkoroError::NotContinuous { id })
    }

    /// Evaluate the log-density of the model at the specified values.
    pub fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        let compiled_sequence = self.compile_backward_from(&self.log_likelihoods);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        let forward_output = compiled_sequence.forward(ForwardInput {
            points,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })?;
        let mut log_density = Point::zeros([batch_size, 1], &Default::default());
        for log_likelihood_id in &self.log_likelihoods {
            log_density = log_density
                + forward_output
                    .evaluations
                    .get(log_likelihood_id)
                    .expect("Log-likelihood should be evaluated.")
                    .0
                    .clone()
                    .unwrap_float();
        }
        Ok(log_density)
    }

    /// The evaluation of [`log_density`](Self::log_density) with the first-order gradient
    /// calculation.
    pub fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        let compiled_sequence = self.compile_backward_from(&self.log_likelihoods);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };

        let cotangents: HashMap<Id, CotangentFrame<B>> = self
            .log_likelihoods
            .iter()
            .map(|id| (*id, Tensor::ones([batch_size, 1, 1], &Default::default())))
            .collect();

        let vjp_output = compiled_sequence.vjp_forward(VjpForwardInput {
            points,
            cotangents: &cotangents,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })?;
        let mut log_density = Point::zeros([batch_size, 1], &Default::default());
        for log_likelihood_id in &self.log_likelihoods {
            log_density = log_density
                + vjp_output
                    .primal
                    .evaluations
                    .get(log_likelihood_id)
                    .expect("Log-likelihood should be evaluated.")
                    .0
                    .clone()
                    .unwrap_float();
        }
        let cotangents = vjp_output
            .cotangents
            .into_iter()
            .map(|(id, v)| (id, v.squeeze_dim(2)))
            .collect();
        Ok(PointEvalOrder1 {
            value: log_density,
            cotangents,
        })
    }

    /// The evaluation of [`log_density`](Self::log_density) with the first-order gradient
    /// calculation and second-order Hessian-product calculation.
    pub fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        let compiled_sequence = self.compile_backward_from(&self.log_likelihoods);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };

        let ord1_cotangents: HashMap<Id, CotangentFrame<B>> = self
            .log_likelihoods
            .iter()
            .map(|id| (*id, Tensor::ones([batch_size, 1, 1], &Default::default())))
            .collect();

        let ord2_cotangents: HashMap<Id, CotangentFrame<B>> = self
            .log_likelihoods
            .iter()
            .map(|id| (*id, Tensor::zeros([batch_size, 1, 1], &Default::default())))
            .collect();

        let hvp_output = compiled_sequence.hvp_forward(HvpForwardInput {
            points,
            tangents,
            ord1_cotangents: &ord1_cotangents,
            ord2_cotangents: &ord2_cotangents,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })?;
        let mut log_density = Point::zeros([batch_size, 1], &Default::default());
        for log_likelihood_id in &self.log_likelihoods {
            log_density = log_density
                + hvp_output
                    .primal
                    .evaluations
                    .get(log_likelihood_id)
                    .expect("Log-likelihood should be evaluated.")
                    .0
                    .clone()
                    .unwrap_float();
        }
        let ord1_cotangents = hvp_output
            .ord1_cotangents
            .into_iter()
            .map(|(id, v)| (id, v.squeeze_dim(2)))
            .collect();
        let ord2_cotangents = hvp_output
            .ord2_cotangents
            .into_iter()
            .map(|(id, v)| (id, v.squeeze_dim(2)))
            .collect();
        Ok(PointEvalOrder2 {
            value: log_density,
            tangents: hvp_output.tangents,
            ord1_cotangents,
            ord2_cotangents,
        })
    }

    /// Stochastically generate all values in the model.
    pub fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>> {
        let empty = HashMap::new();
        self.generate_from(&empty, batch_size)
    }

    /// Stochastically generate all values in the model.
    pub fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>> {
        let ids: Vec<Id> = self.nodes.keys().copied().collect();
        let compiled_sequence = self.compile_backward_from(&ids);
        compiled_sequence.forward(ForwardInput {
            points,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: true,
            },
        })
    }

    /// Simple interface for evaluating multiple mapped quantities.
    pub fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>> {
        let compiled_sequence = self.compile_backward_from(ids);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        compiled_sequence.forward(ForwardInput {
            points,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })
    }

    /// The JVP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients, instead
    /// use [`evaluate_ids_ord1`](Self::evaluate_ids_ord1).
    pub fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>> {
        let compiled_sequence = self.compile_backward_from(ids);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        compiled_sequence.jvp_forward(JvpForwardInput {
            points,
            tangents,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })
    }

    /// The VJP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients, instead
    /// use [`evaluate_ids_ord1`](Self::evaluate_ids_ord1).
    pub fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>> {
        let compiled_sequence = self.compile_backward_from(ids);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        compiled_sequence.vjp_forward(VjpForwardInput {
            points,
            cotangents,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })
    }

    /// The HVP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ids_ord2`](Self::evaluate_ids_ord2).
    pub fn evaluate_ids_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<HvpForwardOutput<B>> {
        let compiled_sequence = self.compile_backward_from(ids);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        compiled_sequence.hvp_forward(HvpForwardInput {
            points,
            tangents,
            ord1_cotangents,
            ord2_cotangents,
            options: ForwardOptions {
                batch_size,
                use_model_samplers: false,
            },
        })
    }

    /// The evaluation [`evaluate_ids`](Self::evaluate_ids) with the first-order gradient calculation.
    pub fn evaluate_ids_ord1(
        &self,
        points: &ValueMap<B>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder1<B>> {
        let compiled_sequence = self.compile_backward_from(ids);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        let cotangents: HashMap<Id, CotangentFrame<B>> = ids
            .iter()
            .map(|id| {
                Ok((
                    *id,
                    Tensor::ones(
                        [batch_size, self.get_manifold(*id)?.dimension(), 1],
                        &Default::default(),
                    ),
                ))
            })
            .collect::<SkoroResult<_>>()?;
        Ok(compiled_sequence
            .vjp_forward(VjpForwardInput {
                points,
                cotangents: &cotangents,
                options: ForwardOptions {
                    batch_size,
                    use_model_samplers: false,
                },
            })?
            .into())
    }

    /// The evaluation [`evaluate_ids`](Self::evaluate_ids) with the first-order gradient and second-order
    /// Hessian-product calculation.
    pub fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>> {
        let compiled_sequence = self.compile_backward_from(ids);
        let batch_size = if let Some((_, point)) = points.iter().next() {
            point.dims()[0]
        } else {
            1
        };
        let ord1_cotangents: HashMap<Id, CotangentFrame<B>> = ids
            .iter()
            .map(|id| {
                Ok((
                    *id,
                    Tensor::ones(
                        [batch_size, self.get_manifold(*id)?.dimension(), 1],
                        &Default::default(),
                    ),
                ))
            })
            .collect::<SkoroResult<_>>()?;
        let ord2_cotangents: HashMap<Id, CotangentFrame<B>> = ids
            .iter()
            .map(|id| {
                Ok((
                    *id,
                    Tensor::zeros(
                        [batch_size, self.get_manifold(*id)?.dimension(), 1],
                        &Default::default(),
                    ),
                ))
            })
            .collect::<SkoroResult<_>>()?;
        Ok(compiled_sequence
            .hvp_forward(HvpForwardInput {
                points,
                tangents,
                ord1_cotangents: &ord1_cotangents,
                ord2_cotangents: &ord2_cotangents,
                options: ForwardOptions {
                    batch_size,
                    use_model_samplers: false,
                },
            })?
            .into())
    }

    /// Simple interface for evaluating a single mapped quantity. This should not be used multiple
    /// times in sequence, since it may lead to inefficient recalculation. In that event, instead
    /// use [`evaluate_ids`](Self::evaluate_ids).
    ///
    /// Returns [`SkoroError::NotContinuous`] if the quantity is discrete.
    pub fn evaluate(&self, points: &ValueMap<B>, id: Id) -> SkoroResult<Point<B>> {
        let mut out = self.evaluate_ids(points, &[id])?;
        take_float(&mut out, id)
    }

    /// The JVP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    pub fn evaluate_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        id: Id,
    ) -> SkoroResult<PointJvpEval<B>> {
        let mut out = self.evaluate_ids_jvp(points, tangents, &[id])?;
        Ok(PointJvpEval {
            value: take_float(&mut out.primal, id)?,
            tangents: out.tangents,
        })
    }

    /// The VJP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    pub fn evaluate_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        id: Id,
    ) -> SkoroResult<PointVjpEval<B>> {
        let mut out = self.evaluate_ids_vjp(points, cotangents, &[id])?;
        Ok(PointVjpEval {
            value: take_float(&mut out.primal, id)?,
            cotangents: out.cotangents,
        })
    }

    /// The HVP evaluation [`evaluate`](Self::evaluate); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ord2`](Self::evaluate_ord2).
    pub fn evaluate_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        id: Id,
    ) -> SkoroResult<PointHvpEval<B>> {
        let mut out =
            self.evaluate_ids_hvp(points, tangents, ord1_cotangents, ord2_cotangents, &[id])?;
        Ok(PointHvpEval {
            value: take_float(&mut out.primal, id)?,
            tangents: out.tangents,
            ord1_cotangents: out.ord1_cotangents,
            ord2_cotangents: out.ord2_cotangents,
        })
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient calculation.
    pub fn evaluate_ord1(&self, points: &ValueMap<B>, id: Id) -> SkoroResult<PointEvalOrder1<B>> {
        let mut out = self.evaluate_ids_ord1(points, &[id])?;
        Ok(PointEvalOrder1 {
            value: take_float(&mut out.primal, id)?,
            cotangents: out.cotangents,
        })
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient and second-order
    /// Hessian-product calculation.
    pub fn evaluate_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        id: Id,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        let mut out = self.evaluate_ids_ord2(points, tangents, &[id])?;
        Ok(PointEvalOrder2 {
            value: take_float(&mut out.primal, id)?,
            tangents: out.tangents,
            ord1_cotangents: out.ord1_cotangents,
            ord2_cotangents: out.ord2_cotangents,
        })
    }

    fn register_children(&mut self, id: Id, parent_ids: &[Id]) -> SkoroResult<()> {
        // Two parents pinning *different* batch sizes describe a node that can never be
        // evaluated, so reject it here rather than at the first pass. Checked before the
        // mutation loop below, so a rejected call leaves no half-registered edges behind.
        //
        // Deliberately partial: a parent whose batch is pinned only indirectly -- through a
        // mapped node that did not fold, because it has a free variable above it -- reports no
        // pin here. That residue is caught in full at evaluation time, where every pinned value
        // in the sequence checks against the same `ForwardOptions::batch_size`.
        let mut pinned: Option<(Id, usize)> = None;
        for &parent_id in parent_ids {
            let Some(batch) = self.pinned_batch_of(parent_id) else {
                continue;
            };
            match pinned {
                Some((_, expected)) if expected != batch => {
                    return Err(SkoroError::BatchSizeMismatch {
                        id: parent_id,
                        expected,
                        found: batch,
                    });
                }
                Some(_) => {}
                None => pinned = Some((parent_id, batch)),
            }
        }

        for &parent_id in parent_ids {
            if !self.nodes.contains_key(&parent_id) {
                return Err(SkoroError::DanglingParent { parent_id });
            }
            match self.children.entry(parent_id) {
                Entry::Occupied(mut entry) => entry.get_mut().insert(id),
                Entry::Vacant(entry) => entry.insert(HashSet::new()).insert(id),
            };
        }
        Ok(())
    }

    /// The batch a node's stored value pins the graph to, if it pins one.
    ///
    /// `None` for a node with no stored value at all -- a parameter, an unfolded mapped node --
    /// which is a statement that nothing is known here, not that the node is batch-agnostic.
    fn pinned_batch_of(&self, id: Id) -> Option<usize> {
        match self.nodes.get(&id)? {
            Node::Constant(ConstantNode { value, .. })
            | Node::Observed(ObservedNode { value, .. }) => value.pinned_batch(),
            Node::Mapped(MappedNode { value, .. })
            | Node::LogDensity(LogDensityNode { value, .. }) => {
                value.as_ref().and_then(QuantityValue::pinned_batch)
            }
            Node::Parameter(_) | Node::Stochastic(_) => None,
        }
    }

    /// Drop the edges from `parent_ids` into `id`, forgetting any parent left childless.
    ///
    /// The inverse of [`register_children`](Self::register_children), and the one place a
    /// child edge is torn down. `update_stochastic_impl`, `demote_to_parameter` and `remove`
    /// each carried their own copy of this loop; `validate_indices` checks the result of all
    /// three against `parent_ids`, so they had to agree exactly.
    fn unregister_children(&mut self, id: Id, parent_ids: &[Id]) {
        for parent_id in parent_ids {
            let Some(children) = self.children.get_mut(parent_id) else {
                continue;
            };
            children.remove(&id);
            if children.is_empty() {
                self.children.remove(parent_id);
            }
        }
    }

    /// Bind `density` to `stochastic` — or to nothing — and make it a live likelihood term.
    ///
    /// The single place a density's registration is written. All three pieces move together:
    /// [`LogDensityNode::stochastic_id`], which is the source of truth; the
    /// `stochastic_to_log_density` reverse index; and membership in `log_likelihoods`. Any
    /// previous binding is released first, so re-marking a density as a bare likelihood does
    /// not strand the old reverse entry.
    ///
    /// A no-op on an id that is not a density. Callers reach this only through
    /// `mark_likelihood_impl`, which has already established the node kind.
    fn link_density(&mut self, density: Id, stochastic: Option<Id>) {
        let previous = match self.nodes.get_mut(&density) {
            Some(Node::LogDensity(dens)) => std::mem::replace(&mut dens.stochastic_id, stochastic),
            _ => return,
        };
        if let Some(previous) = previous
            && Some(previous) != stochastic
        {
            self.stochastic_to_log_density.remove(&previous);
        }
        if let Some(stochastic) = stochastic {
            self.stochastic_to_log_density.insert(stochastic, density);
        }
        if !self.log_likelihoods.contains(&density) {
            self.log_likelihoods.push(density);
        }
    }

    /// Remove `density` from the likelihood terms. The one place that vector shrinks.
    fn drop_density_term(&mut self, density: Id) {
        if let Some(i) = self.log_likelihoods.iter().position(|id| *id == density) {
            self.log_likelihoods.remove(i);
        }
    }

    /// Forget a density's registrations as its node leaves the arena.
    ///
    /// The node carries its own `stochastic_id` out with it, so only the two derived
    /// collections need clearing. Distinct from
    /// [`unregister_density`](Self::unregister_density), which leaves the node in place.
    fn forget_density_node(&mut self, density: Id, stochastic: Option<Id>) {
        self.drop_density_term(density);
        if let Some(stochastic) = stochastic {
            self.stochastic_to_log_density.remove(&stochastic);
        }
    }

    /// The constant-folded value of an operation over `parent_ids`, if every parent is
    /// already concrete.
    ///
    /// Errors [`SkoroError::NodeNotFound`] on a parent that is not in the arena. That used to
    /// be an `expect`, and it is reachable rather than internal: a retired parent leaves its
    /// children naming an id the arena no longer has.
    fn check_cache(
        &self,
        parent_ids: &[Id],
        op: &Arc<dyn GraphOperation<B>>,
    ) -> SkoroResult<Option<QuantityValue<B>>> {
        let parents: Vec<&Node<B>> = parent_ids
            .iter()
            .map(|parent_id| {
                self.nodes
                    .get(parent_id)
                    .ok_or(SkoroError::NodeNotFound { id: *parent_id })
            })
            .collect::<SkoroResult<_>>()?;
        let parent_values: Vec<Option<(Id, &QuantityValue<B>)>> = parents
            .iter()
            .map(|parent| match parent {
                Node::Constant(ConstantNode { id, value, .. })
                | Node::Observed(ObservedNode { id, value, .. }) => Some((*id, value)),
                Node::Mapped(MappedNode { id, value, .. })
                | Node::LogDensity(LogDensityNode { id, value, .. }) => {
                    value.as_ref().map(|value| (*id, value))
                }
                _ => None,
            })
            .collect();
        let parent_diffs: Vec<bool> = parents
            .iter()
            .map(|p| is_differentiable_parent(p))
            .collect();

        // determine this node's stochasticity by its parents
        let is_stochastic = parent_values.iter().any(|value| value.is_none());

        // if the value is not stochastic, we may cache it
        if is_stochastic {
            return Ok(None);
        }

        // A parent that pins a batch forces the fold to run at that batch, and the folded value
        // to stay rank-2. Folding at 1 and unbatching -- which is what this did before batch-pinned
        // values existed -- would silently keep row 0 and discard the rest.
        //
        // Disagreeing parents are rejected in `register_children`, so the first pin found is the
        // only one.
        let pinned = parent_values
            .iter()
            .flatten()
            .find_map(|(_, value)| value.pinned_batch());
        let batch = pinned.unwrap_or(1);

        let n = parent_values.len();
        let mut points: Vec<Option<Input<B>>> = parent_values
            .into_iter()
            .map(|value| {
                let (id, value) = value.expect("internal: stochastic parents returned above");
                value.materialize(batch, id)
            })
            .collect::<SkoroResult<Vec<_>>>()?
            .into_iter()
            .map(Some)
            .collect();
        points.push(None);
        op.lower((0..n).collect(), parent_diffs, n)
            .forward(&mut points);
        Ok(points[n].take().map(|out| match pinned {
            None => QuantityValue::Unbatched(out.unbatch()),
            Some(_) => QuantityValue::Batched(out),
        }))
    }

    /// Returns a topologically sorted list of node Ids ending at the given execution leaves.
    ///
    /// In other words, we hand this function Ids of target nodes we want our operations to *end*,
    /// and it will return an ordered vector of Ids of nodes which we walk to get there.
    pub fn topological_sort(&self, targets: &[Id]) -> Vec<Id> {
        let mut visited = HashSet::new();
        let mut ordered = Vec::new();
        let mut stack = Vec::new();

        for &target in targets {
            stack.push((target, true));
        }

        while let Some((current, first_visit)) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            if first_visit {
                stack.push((current, false));

                if let Some(node) = self.nodes.get(&current) {
                    for &parent in node.parent_ids().iter().rev() {
                        if !visited.contains(&parent) {
                            stack.push((parent, true));
                        }
                    }
                }
            } else {
                visited.insert(current);
                ordered.push(current);
            }
        }

        ordered
    }

    /// Compile a subgraph into a sequence of operations.
    pub fn compile_backward_from(&self, targets: &[Id]) -> CompiledSequence<B> {
        // get the nodes in the order they appear
        let ids = self.topological_sort(targets);

        // initialize states we fold over
        let mut steps = Vec::new();
        let mut id_to_index = HashMap::new();
        let mut max_index: usize = 0;

        // visit each node and prepare the sequence
        for id in ids {
            let node = self.nodes.get(&id).unwrap();
            match node {
                Node::Constant(ConstantNode {
                    id,
                    value,
                    quantity_type,
                })
                | Node::Observed(ObservedNode {
                    id,
                    value,
                    quantity_type,
                    ..
                }) => {
                    steps.push(CompiledStep::Constant {
                        id: *id,
                        value: value.clone(),
                        quantity_type: quantity_type.clone(),
                    });
                    id_to_index.insert(id, max_index);
                    max_index += 1;
                }
                Node::Parameter(ParameterNode { id, quantity_type }) => {
                    steps.push(CompiledStep::Parameter {
                        id: *id,
                        quantity_type: quantity_type.clone(),
                    });
                    id_to_index.insert(id, max_index);
                    max_index += 1;
                }
                Node::Mapped(MappedNode {
                    id,
                    op,
                    parent_ids,
                    value,
                    quantity_type,
                }) => {
                    let input_indices = parent_ids
                        .iter()
                        .map(|id| *id_to_index.get(id).unwrap())
                        .collect();
                    let input_diffs: Vec<bool> = parent_ids
                        .iter()
                        .map(|id| {
                            is_differentiable_parent(
                                self.nodes.get(id).expect("Parent node should be in graph."),
                            )
                        })
                        .collect();
                    let differentiable = op.is_differentiable();

                    steps.push(CompiledStep::Operation {
                        id: *id,
                        constant_inputs: !differentiable || !input_diffs.iter().any(|b| *b),
                        forward_only: !differentiable,
                        op: op.lower(input_indices, input_diffs, max_index),
                        value: value.clone(),
                        quantity_type: quantity_type.clone(),
                    });
                    id_to_index.insert(id, max_index);
                    max_index += 1;
                }
                Node::LogDensity(LogDensityNode {
                    id,
                    op,
                    parent_ids,
                    value,
                    ..
                }) => {
                    let input_indices = parent_ids
                        .iter()
                        .map(|id| *id_to_index.get(id).unwrap())
                        .collect();
                    let input_diffs: Vec<bool> = parent_ids
                        .iter()
                        .map(|id| {
                            is_differentiable_parent(
                                self.nodes.get(id).expect("Parent node should be in graph."),
                            )
                        })
                        .collect();
                    let differentiable = op.is_differentiable();

                    steps.push(CompiledStep::Operation {
                        id: *id,
                        constant_inputs: !differentiable || !input_diffs.iter().any(|b| *b),
                        forward_only: !differentiable,
                        op: op.lower(input_indices, input_diffs, max_index),
                        value: value.clone(),
                        quantity_type: Arc::new(DynContinuous(Real)),
                    });
                    id_to_index.insert(id, max_index);
                    max_index += 1;
                }
                Node::Stochastic(StochasticNode {
                    id,
                    sampler,
                    parent_ids,
                    quantity_type,
                    ..
                }) => {
                    let input_indices = parent_ids
                        .iter()
                        .map(|id| *id_to_index.get(id).unwrap())
                        .collect();

                    steps.push(CompiledStep::Sampler {
                        id: *id,
                        sampler: sampler.lower(input_indices, max_index),
                        quantity_type: quantity_type.clone(),
                    });
                    id_to_index.insert(id, max_index);
                    max_index += 1;
                }
            };
        }

        CompiledSequence::new(steps)
    }
}

#[derive(Debug, Clone)]
pub struct PointJvpEval<B: Backend> {
    pub value: Point<B>,
    pub tangents: HashMap<Id, TangentFrame<B>>,
}

#[derive(Debug, Clone)]
pub struct PointVjpEval<B: Backend> {
    pub value: Point<B>,
    pub cotangents: HashMap<Id, CotangentFrame<B>>,
}

#[derive(Debug, Clone)]
pub struct PointHvpEval<B: Backend> {
    pub value: Point<B>,
    pub tangents: HashMap<Id, TangentVector<B>>,
    pub ord1_cotangents: HashMap<Id, CotangentFrame<B>>,
    pub ord2_cotangents: HashMap<Id, CotangentFrame<B>>,
}

#[derive(Debug, Clone)]
pub struct PointEvalOrder1<B: Backend> {
    pub value: Point<B>,
    pub cotangents: HashMap<Id, CotangentVector<B>>,
}

#[derive(Debug, Clone)]
pub struct PointEvalOrder2<B: Backend> {
    pub value: Point<B>,
    pub tangents: HashMap<Id, TangentVector<B>>,
    pub ord1_cotangents: HashMap<Id, CotangentVector<B>>,
    pub ord2_cotangents: HashMap<Id, CotangentVector<B>>,
}

#[derive(Debug, Clone)]
pub struct EvalOrder1<B: Backend> {
    pub primal: ForwardOutput<B>,
    pub cotangents: HashMap<Id, CotangentVector<B>>,
}

impl<B: Backend> From<VjpForwardOutput<B>> for EvalOrder1<B> {
    fn from(value: VjpForwardOutput<B>) -> Self {
        Self {
            primal: value.primal,
            cotangents: value
                .cotangents
                .into_iter()
                .map(|(id, frame)| (id, frame.squeeze_dim(2)))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalOrder2<B: Backend> {
    pub primal: ForwardOutput<B>,
    pub tangents: HashMap<Id, TangentVector<B>>,
    pub ord1_cotangents: HashMap<Id, CotangentVector<B>>,
    pub ord2_cotangents: HashMap<Id, CotangentVector<B>>,
}

impl<B: Backend> From<HvpForwardOutput<B>> for EvalOrder2<B> {
    fn from(value: HvpForwardOutput<B>) -> Self {
        Self {
            primal: value.primal,
            tangents: value.tangents,
            ord1_cotangents: value
                .ord1_cotangents
                .into_iter()
                .map(|(id, frame)| (id, frame.squeeze_dim(2)))
                .collect(),
            ord2_cotangents: value
                .ord2_cotangents
                .into_iter()
                .map(|(id, frame)| (id, frame.squeeze_dim(2)))
                .collect(),
        }
    }
}
