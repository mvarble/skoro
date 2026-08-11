use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use skoro_backend::{
    Backend, CotangentFrame, CotangentVector, Point, TangentFrame, TangentVector, Tensor,
};
use skoro_core::{
    constant_value::ConstantValue,
    error::{SkoroError, SkoroResult},
    id::{Id, IdGenerator},
    kind::{Kind, QuantityKind},
    pack::Packable,
    sampler::Sampler,
};
use skoro_operation::{Input, ValueMap};
use skoro_riemann::real::Real;

use crate::compiler::{
    CompiledSequence, CompiledStep, ForwardInput, ForwardOptions, ForwardOutput, HvpForwardInput,
    HvpForwardOutput, JvpForwardInput, JvpForwardOutput, VjpForwardInput, VjpForwardOutput,
};
use crate::node::{
    ConstantNode, LogDensityNode, MappedNode, Node, ObservedNode, ParameterNode, StochasticNode,
};
use crate::operation::GraphOperation;
use crate::quantity::{DynContinuous, DynQuantityType, QuantityType};
use crate::sampler::{GraphSampler, KindedSampler};

/// A mutable probabilistic graphical model.
///
/// It stores variables, transformations, and their statistical properties.
#[derive(Clone)]
/// The five collections below carry an invariant between them: every id in
/// `log_likelihoods` names a [`Node::LogDensity`] in `nodes`, and the two density maps
/// are inverses of one another over the subset of densities that belong to a stochastic
/// quantity. Only the mutating methods on this type maintain that invariant, so the
/// fields are private and reachable through the read accessors below.
pub struct Graph<B: Backend> {
    /// Arena of all nodes.
    nodes: HashMap<Id, Node<B>>,
    /// Children edge data (parent edge data is local)
    children: HashMap<Id, HashSet<Id>>,
    /// Log-likelihood terms
    log_likelihoods: Vec<Id>,
    /// A map from Ids of log-densities to their stochastic quantities.
    log_density_to_stochastic: HashMap<Id, Id>,
    /// A map from Ids of stochastic quantities to their log-densities.
    stochastic_to_log_density: HashMap<Id, Id>,
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
            log_density_to_stochastic: HashMap::new(),
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

    /// The log-density scoring `stochastic`, if it has one.
    pub fn density_of(&self, stochastic: Id) -> Option<Id> {
        self.stochastic_to_log_density.get(&stochastic).copied()
    }

    /// The stochastic quantity that `density` scores, if it belongs to one. A
    /// free-standing likelihood factor belongs to none and yields `None`.
    pub fn stochastic_of(&self, density: Id) -> Option<Id> {
        self.log_density_to_stochastic.get(&density).copied()
    }

    /// Add a fixed, immutable quantity, continuous or discrete.
    pub fn add_constant<K, T, D>(&mut self, quantity_type: T, value: &D) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let value = K::wrap_unbatched_point(quantity_type.pack_value(value));
        self.add_constant_impl(quantity_type.into_dyn(), value)
    }

    fn add_constant_impl(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        value: ConstantValue<B>,
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

    fn add_mapped_impl(
        &mut self,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        op: Arc<dyn GraphOperation<B>>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id> {
        let id = self.id_generator.issue_id();
        self.register_children(id, &parent_ids)?;
        let value = self.check_cache(&parent_ids, &op);
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
                self.nodes.insert(
                    id,
                    Node::LogDensity(LogDensityNode {
                        id,
                        op,
                        stochastic_id,
                        parent_ids,
                        value,
                    }),
                );
                if !self.log_likelihoods.contains(&id) {
                    self.log_likelihoods.push(id);
                }
                if let Some(stochastic_id) = stochastic_id {
                    self.log_density_to_stochastic.insert(id, stochastic_id);
                    self.stochastic_to_log_density.insert(stochastic_id, id);
                }
            }
            Some(Node::LogDensity(mut dens)) => {
                match &stochastic_id {
                    Some(stochastic_id) => {
                        match self.log_density_to_stochastic.entry(id) {
                            Entry::Vacant(entry) => {
                                entry.insert(*stochastic_id);
                                let _ = self.stochastic_to_log_density.insert(*stochastic_id, id);
                            }
                            Entry::Occupied(entry) => {
                                if *entry.get() != *stochastic_id {
                                    // re-insert the node we removed before erroring
                                    let stochastic_id = *stochastic_id;
                                    self.nodes.insert(id, Node::LogDensity(dens));
                                    return Err(SkoroError::DuplicateDensity { stochastic_id });
                                }
                            }
                        };
                    }
                    None => {
                        let stochastic_id = self.log_density_to_stochastic.remove(&id);
                        if let Some(stochastic_id) = stochastic_id {
                            self.stochastic_to_log_density.remove(&stochastic_id);
                        }
                    }
                };
                dens.stochastic_id = stochastic_id;
                self.nodes.insert(id, Node::LogDensity(dens));
            }
            Some(node) => {
                // re-insert the node we removed before erroring
                self.nodes.insert(id, node);
                return Err(SkoroError::NotMapped { id });
            }
            None => return Err(SkoroError::NodeNotFound { id }),
        }
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
                    K::wrap_unbatched_point(quantity_type_ref.pack_value(value))
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

        if let Some(children_ids) = self.children.get(&id) {
            let mut visit: VecDeque<Id> = children_ids.iter().copied().collect();
            while let Some(ancestor_id) = visit.pop_front() {
                // if the ancestor is not a mapped node, continue
                if !matches!(
                    self.nodes
                        .get(&ancestor_id)
                        .expect("A node's children should be present in the graph."),
                    Node::Mapped(_) | Node::LogDensity(_)
                ) {
                    continue;
                }

                // If the ancestor is a mapped node, we see if we can cache the value.
                // This requires removing the node, since we cannot mutably access it while
                // accessing its parents.
                match self.nodes.remove(&ancestor_id).unwrap() {
                    Node::Mapped(mapped) => {
                        let value = self.check_cache(&mapped.parent_ids, &mapped.op);

                        if value.is_some()
                            && let Some(children_ids) = self.children.get(&ancestor_id)
                        {
                            children_ids.iter().for_each(|id| visit.push_back(*id));
                        }

                        self.nodes
                            .insert(ancestor_id, Node::Mapped(MappedNode { value, ..mapped }));
                    }
                    Node::LogDensity(dens) => {
                        let value = self.check_cache(&dens.parent_ids, &dens.op);

                        if value.is_some()
                            && let Some(children_ids) = self.children.get(&ancestor_id)
                        {
                            children_ids.iter().for_each(|id| visit.push_back(*id));
                        }

                        self.nodes.insert(
                            ancestor_id,
                            Node::LogDensity(LogDensityNode { value, ..dens }),
                        );
                    }
                    _ => unreachable!(),
                };
            }
        }
        Ok(())
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
    /// Returns an error if the node does not exist, is not stochastic, the
    /// sampler's kind does not match the node's, a new parent is dangling, or
    /// the update would introduce a cycle.
    pub fn update_stochastic<K, S>(
        &mut self,
        id: Id,
        new_sampler: S,
        new_parents: Vec<Id>,
    ) -> SkoroResult<()>
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
    ) -> SkoroResult<()> {
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

        // update the node and the children registry
        if let Some(Node::Stochastic(stoch)) = self.nodes.get_mut(&id) {
            // remove this node from the children registry
            let mut remove = Vec::new();
            for (pid, children) in self.children.iter_mut() {
                children.remove(&id);
                if children.is_empty() && !new_parents.contains(pid) {
                    remove.push(*pid);
                }
            }

            // add this node back into the children registry
            for &pid in &new_parents {
                match self.children.entry(pid) {
                    Entry::Occupied(mut entry) => entry.get_mut().insert(id),
                    Entry::Vacant(entry) => entry.insert(HashSet::new()).insert(id),
                };
            }

            // remove the children sets of bereaved parents
            for id in remove {
                self.children.remove(&id);
            }

            // update the node (kind validated above)
            stoch.sampler = new_sampler;
            stoch.parent_ids = new_parents;
        } else {
            unreachable!("node kind validated above");
        }
        self.unregister_density(id);
        Ok(())
    }

    /// Unregister the log-density owned by a stochastic quantity, dropping it from the
    /// model's likelihood terms.
    ///
    /// The [`LogDensity`](Node::LogDensity) node itself is left in the arena: it may still be
    /// referenced as a child, and removing it is the caller's decision. A caller replacing a
    /// distribution wholesale is expected to clean up the orphan.
    fn unregister_density(&mut self, stochastic_id: Id) {
        let Some(log_density_id) = self.stochastic_to_log_density.remove(&stochastic_id) else {
            return;
        };
        self.log_density_to_stochastic.remove(&log_density_id);
        if let Some((i, _)) = self
            .log_likelihoods
            .iter()
            .enumerate()
            .find(|(_, id)| **id == log_density_id)
        {
            self.log_likelihoods.remove(i);
        }
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
    /// Returns [`SkoroError::NodeNotFound`] if the node does not exist, or
    /// [`SkoroError::NotStochastic`] if it is neither stochastic nor already a parameter.
    pub fn demote_to_parameter(&mut self, id: Id) -> SkoroResult<()> {
        match self.nodes.get(&id) {
            Some(Node::Stochastic(_)) => {}
            // Already a free variable; nothing to demote.
            Some(Node::Parameter(_)) => return Ok(()),
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
        for parent_id in &parent_ids {
            let mut bereaved = false;
            if let Some(children) = self.children.get_mut(parent_id) {
                children.remove(&id);
                bereaved = children.is_empty();
            }
            if bereaved {
                self.children.remove(parent_id);
            }
        }

        self.nodes
            .insert(id, Node::Parameter(ParameterNode { id, quantity_type }));
        self.unregister_density(id);
        // Defensive, and cheap: today `check_cache` folds nothing downstream of a
        // stochastic node (it yields `None` for both `Stochastic` and `Parameter`
        // parents), so this finds nothing to clear. It guards the invariant rather
        // than a live failure — the node's value is no longer determined by its
        // parents, and a fold rule that ever admitted stochastic parents would make
        // every descendant stale.
        self.clear_cached_descendants(id);
        Ok(())
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
            self.children.remove(id);
            let Some(node) = self.nodes.remove(id) else {
                return Err(SkoroError::NodeNotFound { id: *id });
            };
            for pid in node.parent_ids() {
                // TODO: check if I need to remove children
                let mut remove = false;
                if let Some(children) = self.children.get_mut(pid) {
                    children.remove(id);
                    if children.is_empty() {
                        remove = true;
                    }
                }
                if remove {
                    self.children.remove(pid);
                }
            }
            if let Node::LogDensity(LogDensityNode { stochastic_id, .. }) = &node {
                if let Some((i, _)) = self
                    .log_likelihoods
                    .iter()
                    .enumerate()
                    .find(|(_, iid)| *iid == id)
                {
                    self.log_likelihoods.remove(i);
                }
                if let Some(stochastic_id) = stochastic_id {
                    self.log_density_to_stochastic.remove(id);
                    self.stochastic_to_log_density.remove(stochastic_id);
                }
            }
            out.push(node);
        }
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
                    .into_float();
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
                    .into_float();
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
                    .into_float();
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

    fn check_cache(
        &self,
        parent_ids: &[Id],
        op: &Arc<dyn GraphOperation<B>>,
    ) -> Option<ConstantValue<B>> {
        let parents: Vec<&Node<B>> = parent_ids
            .iter()
            .map(|parent_id| {
                self.nodes
                    .get(parent_id)
                    .expect("Parent node should be in graph.")
            })
            .collect();
        let parent_values: Vec<Option<Input<B>>> = parents
            .iter()
            .map(|parent| match parent {
                Node::Constant(ConstantNode { value, .. })
                | Node::Observed(ObservedNode { value, .. }) => Some(value.expand(1)),
                Node::Mapped(MappedNode { value, .. })
                | Node::LogDensity(LogDensityNode { value, .. }) => {
                    value.as_ref().map(|value| value.expand(1))
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
            None
        } else {
            let n = parent_values.len();
            let mut points = parent_values;
            points.push(None);
            op.lower((0..n).collect(), parent_diffs, n)
                .forward(&mut points);
            points[n].take().map(ConstantValue::unbatch)
        }
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
