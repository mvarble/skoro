//! Embedded types: declared by a retraction `R`, its inverse `Q` and, optionally, a projection
//! `P`, written as composites, and reached through the boundary operations, whose derivative maps
//! are passes over those composites' programs.
//!
//! Write `𝓡` and `𝓠` for the programs of `R` and `Q`, each with two roots, and `𝓟` for that of
//! `P`, with one. For the inclusion `ι` at a point stored as `x`, `B u` is
//! `𝓡.tangent((x, 0); (0, u))`, `Bᵀ α` is the cotangent of `𝓡` at `(x, 0)` at its second input,
//! and the second-order map is the `β` of `𝓡.hessian((x, 0); (B u, 0); α)` at its second input.
//! The retraction `⊕` runs `𝓡` and the projection `π` runs `𝓟`, and their maps compose those of
//! `𝓡` or `𝓟` with `L(y)`, read off `𝓠` on the diagonal `(y, y)`, as the implementation
//! companion's tables say, line by line.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use skoro_tensor::{Frame, Points, Target, Value};

use crate::{
    Composite, Error, FirstOrder, Forward, Frames, Kernel, Lowered, Lowering, NodeId, Operation,
    Param, Pass, Program, Pullback, SecondOrder, SecondPullback, Space, Values,
};

/// A curved type of storage `n` and dimension `d`, declared by its retraction and its inverse.
pub struct Embedded {
    name: &'static str,
    params: Vec<Param>,
    storage: usize,
    dimension: usize,
    retraction: Arc<Composite>,
    inverse: Arc<Composite>,
    projection: Option<Arc<Composite>>,
    flat: bool,
}

impl Embedded {
    /// `retraction`, `Vector(n)` and `Vector(d)` to `Vector(n)`, is the `R` of the mathematics;
    /// `inverse`, two of `Vector(n)` to `Vector(d)`, its `Q`; `projection`, `Vector(n)` to
    /// `Vector(n)`, its `P`, when the type has one. Any other signature is refused
    /// (`InvalidDeclaration`).
    pub fn declare(
        name: &'static str,
        params: Vec<Param>,
        storage: usize,
        dimension: usize,
        retraction: Arc<Composite>,
        inverse: Arc<Composite>,
        projection: Option<Arc<Composite>>,
    ) -> Result<Arc<Embedded>, Error> {
        let (n, d) = (Space::Vector(storage), Space::Vector(dimension));
        let invalid = |reason: &str| Error::InvalidDeclaration {
            name: name.to_string(),
            reason: reason.to_string(),
        };
        if storage == 0 || dimension == 0 || dimension > storage {
            return Err(invalid("its dimension is not between one and its storage"));
        }
        if retraction.inputs() != [n.clone(), d.clone()] || *retraction.output() != n {
            return Err(invalid(
                "its retraction does not take Vector(n) and Vector(d) to Vector(n)",
            ));
        }
        if inverse.inputs() != [n.clone(), n.clone()] || *inverse.output() != d {
            return Err(invalid(
                "its inverse does not take two of Vector(n) to Vector(d)",
            ));
        }
        if let Some(p) = &projection
            && (p.inputs() != [n.clone()] || *p.output() != n)
        {
            return Err(invalid(
                "its projection does not take Vector(n) to Vector(n)",
            ));
        }
        Ok(Arc::new(Embedded {
            name,
            params,
            storage,
            dimension,
            retraction,
            inverse,
            projection,
            flat: false,
        }))
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn params(&self) -> &[Param] {
        &self.params
    }

    pub fn storage(&self) -> usize {
        self.storage
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// `R`.
    pub fn retraction(&self) -> &Arc<Composite> {
        &self.retraction
    }

    /// `Q`.
    pub fn inverse(&self) -> &Arc<Composite> {
        &self.inverse
    }

    /// `P`, when the type declares one.
    pub fn projection(&self) -> Option<&Arc<Composite>> {
        self.projection.as_ref()
    }

    /// Whether `skoro.project` maps onto the type.
    pub fn has_projection(&self) -> bool {
        self.flat || self.projection.is_some()
    }
}

/// Two declarations are one type when their names and parameters are.
impl PartialEq for Embedded {
    fn eq(&self, other: &Embedded) -> bool {
        self.name == other.name && self.params == other.params
    }
}

impl Eq for Embedded {}

impl Hash for Embedded {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.params.hash(state);
    }
}

impl fmt::Debug for Embedded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{:?}[{}, {}]",
            self.name, self.params, self.storage, self.dimension
        )
    }
}

/// Which boundary operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Boundary {
    Include,
    Retract,
    Project,
}

/// The inclusion `ι`: a value of the type, read as the `Vector(n)` it is stored as.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Include(Arc<Embedded>);

/// The retraction `⊕`: a value of the type and `Vector(d)`, to the type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Retract(Arc<Embedded>);

/// The projection `π`: a `Vector(n)`, to the type (`NoProjection` when it declares none).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Project(Arc<Embedded>);

impl Include {
    pub fn new(declaration: Arc<Embedded>) -> Include {
        Include(declaration)
    }
}

impl Retract {
    pub fn new(declaration: Arc<Embedded>) -> Retract {
        Retract(declaration)
    }
}

impl Project {
    pub fn new(declaration: Arc<Embedded>) -> Project {
        Project(declaration)
    }
}

fn is_type(space: &Space, declaration: &Arc<Embedded>) -> bool {
    matches!(space, Space::Embedded(e) if e == declaration)
}

impl Operation for Include {
    fn name(&self) -> &'static str {
        "skoro.include"
    }

    fn params(&self) -> Vec<Param> {
        vec![Param::Space(Space::Embedded(self.0.clone()))]
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        match inputs {
            [x] if is_type(x, &self.0) => Ok(Space::Vector(self.0.storage)),
            _ => Err(Error::SpaceMismatch {
                operation: self.name(),
                inputs: inputs.to_vec(),
            }),
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        lower(Boundary::Include, &self.0, lowering)
    }
}

impl Operation for Retract {
    fn name(&self) -> &'static str {
        "skoro.retract"
    }

    fn params(&self) -> Vec<Param> {
        vec![Param::Space(Space::Embedded(self.0.clone()))]
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        match inputs {
            [x, Space::Vector(d)] if is_type(x, &self.0) && *d == self.0.dimension => Ok(x.clone()),
            _ => Err(Error::SpaceMismatch {
                operation: self.name(),
                inputs: inputs.to_vec(),
            }),
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        lower(Boundary::Retract, &self.0, lowering)
    }
}

impl Operation for Project {
    fn name(&self) -> &'static str {
        "skoro.project"
    }

    fn params(&self) -> Vec<Param> {
        vec![Param::Space(Space::Embedded(self.0.clone()))]
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        let space = Space::Embedded(self.0.clone());
        if !self.0.has_projection() {
            return Err(Error::NoProjection { space });
        }
        match inputs {
            [Space::Vector(n)] if *n == self.0.storage => Ok(space),
            _ => Err(Error::SpaceMismatch {
                operation: self.name(),
                inputs: inputs.to_vec(),
            }),
        }
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        self.output(inputs)?;
        lower(Boundary::Project, &self.0, lowering)
    }
}

/// The kernel of a boundary operation on one target, of the most its inner passes admit. It
/// compiles the composites its maps run: `R` for `ι`, `R` and `Q` for `⊕`, `P` and `Q` for `π`.
fn lower(
    boundary: Boundary,
    declaration: &Arc<Embedded>,
    lowering: &mut Lowering<'_>,
) -> Result<Lowered, Error> {
    let mut kernel = BoundaryKernel {
        boundary,
        d: declaration.dimension,
        target: lowering.target().clone(),
        flat: declaration.flat,
        r: None,
        q: None,
        p: None,
    };
    if !declaration.flat {
        if boundary != Boundary::Project {
            kernel.r = Some(Inner::new(lowering.compile(&declaration.retraction)?));
        }
        if boundary != Boundary::Include {
            kernel.q = Some(Inner::new(lowering.compile(&declaration.inverse)?));
        }
        if boundary == Boundary::Project {
            let Some(projection) = &declaration.projection else {
                return Err(Error::NoProjection {
                    space: Space::Embedded(declaration.clone()),
                });
            };
            kernel.p = Some(Inner::new(lowering.compile(projection)?));
        }
    }
    let kernel = Arc::new(kernel);
    Ok(Lowered::Kernel(match kernel.order() {
        2 => Kernel::Two(kernel),
        1 => Kernel::One(kernel),
        _ => Kernel::Zero(kernel),
    }))
}

/// A compiled inner program of `K` roots and one target, and its passes.
#[derive(Clone)]
struct Inner<const K: usize> {
    program: Arc<Program>,
    roots: [NodeId; K],
    output: NodeId,
}

/// What an inner pass gives at each root, `None` read as zero by the caller.
type AtRoots<const K: usize> = [Option<Frame>; K];

fn admitted<T>(result: Result<T, Error>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => unreachable!("an inner pass admitted when the kernel was lowered: {error}"),
    }
}

impl<const K: usize> Inner<K> {
    fn new(program: Arc<Program>) -> Inner<K> {
        let free = program.free();
        if free.len() != K {
            unreachable!("a declaration's composite takes {K} inputs, checked when declared")
        }
        let roots = std::array::from_fn(|i| free[i].node);
        let output = program.targets()[0];
        Inner {
            program,
            roots,
            output,
        }
    }

    fn admits(&self, pass: Pass<'_>) -> bool {
        self.program.admits(&pass).is_ok()
    }

    /// Whether the program admits its first-order passes, and its Hessian passes.
    fn order(&self) -> usize {
        let out = [self.output];
        if self.admits(Pass::Hessian {
            seeded: &self.roots,
            targets: &out,
        }) {
            2
        } else if self.admits(Pass::Tangent {
            seeded: &self.roots,
        }) && self.admits(Pass::Cotangent { targets: &out })
        {
            1
        } else {
            0
        }
    }

    fn values(&self, at: [&Value; K]) -> Values {
        let mut values = Values::new();
        for (root, value) in self.roots.iter().zip(at) {
            values = values.with(*root, value.clone());
        }
        values
    }

    fn frames(&self, frames: [Option<&Frame>; K]) -> Frames {
        let mut out = Frames::new();
        for (root, frame) in self.roots.iter().zip(frames) {
            if let Some(frame) = frame {
                out.insert(*root, frame.clone());
            }
        }
        out
    }

    fn forward(&self, at: [&Value; K]) -> Value {
        let mut values = admitted(self.program.forward(&self.values(at)));
        match values.remove(self.output) {
            Some(value) => value,
            None => unreachable!("a program computes its target"),
        }
    }

    fn tangent(&self, at: [&Value; K], tangents: [Option<&Frame>; K]) -> Frame {
        let (_, frames) = admitted(
            self.program
                .tangent(&self.values(at), &self.frames(tangents)),
        );
        match frames.get(self.output) {
            Some(frame) => frame.clone(),
            None => unreachable!("a tangent pass gives a frame at its target"),
        }
    }

    fn cotangent(&self, at: [&Value; K], seed: &Frame) -> AtRoots<K> {
        let seeds = Frames::new().with(self.output, seed.clone());
        let (_, frames) = admitted(self.program.cotangent(&self.values(at), &seeds));
        self.roots.map(|root| frames.get(root).cloned())
    }

    /// `(α, β)` at each root.
    fn hessian(
        &self,
        at: [&Value; K],
        velocities: [Option<&Frame>; K],
        seed: &Frame,
    ) -> (AtRoots<K>, AtRoots<K>) {
        let seeds = Frames::new().with(self.output, seed.clone());
        let second = admitted(self.program.hessian(
            &self.values(at),
            &self.frames(velocities),
            &seeds,
        ));
        (
            self.roots.map(|root| second.alpha.get(root).cloned()),
            self.roots.map(|root| second.beta.get(root).cloned()),
        )
    }
}

#[derive(Clone)]
struct BoundaryKernel {
    boundary: Boundary,
    d: usize,
    target: Target,
    /// A flat declaration's maps are the identity, and its kernels compile no programs.
    flat: bool,
    /// The programs of `R`, `Q` and `P` that the boundary operation runs.
    r: Option<Inner<2>>,
    q: Option<Inner<2>>,
    p: Option<Inner<1>>,
}

fn sum(a: Option<Frame>, b: Option<Frame>, target: &Target) -> Option<Frame> {
    match (a, b) {
        (Some(a), Some(b)) => Some(
            match Frame::from_burn(a.into_burn() + b.into_burn(), target) {
                Ok(frame) => frame,
                Err(error) => unreachable!("frames of one kernel share its target: {error}"),
            },
        ),
        (a, b) => a.or(b),
    }
}

fn continuous(value: &Value) -> &Points {
    match value {
        Value::Continuous(points) => points,
        Value::Discrete(_) => unreachable!("checked when the node was added: a continuous input"),
    }
}

fn compiled<const K: usize>(inner: &Option<Inner<K>>) -> &Inner<K> {
    match inner {
        Some(inner) => inner,
        None => unreachable!("a boundary kernel compiles the programs its maps run"),
    }
}

impl BoundaryKernel {
    fn r(&self) -> &Inner<2> {
        compiled(&self.r)
    }

    fn q(&self) -> &Inner<2> {
        compiled(&self.q)
    }

    fn p(&self) -> &Inner<1> {
        compiled(&self.p)
    }

    fn order(&self) -> usize {
        if self.flat {
            return 2;
        }
        match self.boundary {
            Boundary::Include => {
                let r = self.r();
                let (x, out) = ([r.roots[0]], [r.output]);
                if r.admits(Pass::Hessian {
                    seeded: &x,
                    targets: &out,
                }) {
                    2
                } else {
                    r.order().min(1)
                }
            }
            Boundary::Retract => self.r().order().min(self.q().order()),
            Boundary::Project => self.p().order().min(self.q().order()),
        }
    }

    fn zero_tangent(&self, rows: usize) -> Value {
        Value::Continuous(Points::zeros(rows, self.d, &self.target))
    }

    /// `B u`, at `x`.
    fn basis(&self, x: &Value, u: &Frame) -> Frame {
        let zero = self.zero_tangent(u.rows());
        self.r().tangent([x, &zero], [None, Some(u)])
    }

    /// `Bᵀ g`, at `x`.
    fn basis_transpose(&self, x: &Value, g: &Frame) -> Frame {
        let zero = self.zero_tangent(g.rows());
        match self.r().cotangent([x, &zero], g) {
            [_, Some(frame)] => frame,
            _ => unreachable!("the cotangent pass reaches the tangent input"),
        }
    }

    /// `L(y) ẏ`: the tangent of `𝓠` on the diagonal `(y, y)`, along its second input.
    fn chart(&self, y: &Value, dy: &Frame) -> Frame {
        self.q().tangent([y, y], [None, Some(dy)])
    }

    /// `L(y)ᵀ α`: the cotangent of `𝓠` on the diagonal at its second input.
    fn chart_transpose(&self, y: &Value, alpha: &Frame) -> Frame {
        match self.q().cotangent([y, y], alpha) {
            [_, Some(frame)] => frame,
            _ => unreachable!("the cotangent pass reaches the second input"),
        }
    }

    fn value(&self, inputs: &[Value]) -> Value {
        match self.boundary {
            Boundary::Include => inputs[0].clone(),
            Boundary::Project if self.flat => inputs[0].clone(),
            Boundary::Project => self.p().forward([&inputs[0]]),
            Boundary::Retract if self.flat => {
                let (x, t) = (continuous(&inputs[0]), continuous(&inputs[1]));
                let y = x.clone().into_burn() + t.clone().into_burn();
                match Points::from_burn(y, &self.target) {
                    Ok(points) => Value::Continuous(points),
                    Err(error) => unreachable!("a flat retraction keeps its target: {error}"),
                }
            }
            Boundary::Retract => self.r().forward([&inputs[0], &inputs[1]]),
        }
    }

    /// The velocity in storage of the first input, `ẋ = B u` for `ι` and `⊕` and `ẇ` for `π`,
    /// and that of the output in storage, `ẏ`, for `⊕` and `π`.
    fn storage_velocities(
        &self,
        inputs: &[Value],
        velocities: &[Option<Frame>],
    ) -> (Option<Frame>, Option<Frame>) {
        let velocity = |j: usize| velocities.get(j).cloned().flatten();
        if self.flat {
            return (None, None);
        }
        match self.boundary {
            Boundary::Include => (velocity(0).map(|u| self.basis(&inputs[0], &u)), None),
            Boundary::Retract => {
                let dx = velocity(0).map(|u| self.basis(&inputs[0], &u));
                let v = velocity(1);
                let dy = (dx.is_some() || v.is_some()).then(|| {
                    self.r()
                        .tangent([&inputs[0], &inputs[1]], [dx.as_ref(), v.as_ref()])
                });
                (dx, dy)
            }
            Boundary::Project => {
                let dw = velocity(0);
                let dy = dw
                    .as_ref()
                    .map(|dw| self.p().tangent([&inputs[0]], [Some(dw)]));
                (dw, dy)
            }
        }
    }

    fn push(&self, inputs: &[Value], y: &Value, tangents: &[Option<Frame>]) -> Option<Frame> {
        if self.flat {
            return match self.boundary {
                Boundary::Include | Boundary::Project => tangents[0].clone(),
                Boundary::Retract => sum(tangents[0].clone(), tangents[1].clone(), &self.target),
            };
        }
        let (dx, dy) = self.storage_velocities(inputs, tangents);
        match self.boundary {
            Boundary::Include => dx,
            Boundary::Retract | Boundary::Project => dy.map(|dy| self.chart(y, &dy)),
        }
    }

    fn recorded(
        &self,
        inputs: &[Value],
        differentiable: &[bool],
        velocities: &[Option<Frame>],
    ) -> (Value, BoundaryPullback) {
        let y = self.value(inputs);
        let (dx, dy) = self.storage_velocities(inputs, velocities);
        let pullback = BoundaryPullback {
            kernel: Arc::new(self.clone()),
            inputs: inputs.to_vec(),
            y: y.clone(),
            differentiable: differentiable.to_vec(),
            v: velocities.get(1).cloned().flatten(),
            dx,
            dy,
        };
        (y, pullback)
    }
}

impl Forward for BoundaryKernel {
    fn forward(&self, inputs: &[Value]) -> Value {
        self.value(inputs)
    }
}

impl FirstOrder for BoundaryKernel {
    fn tangent(&self, inputs: &[Value], tangents: &[Option<Frame>]) -> (Value, Option<Frame>) {
        let y = self.value(inputs);
        let pushed = self.push(inputs, &y, tangents);
        (y, pushed)
    }

    fn record(&self, inputs: &[Value], differentiable: &[bool]) -> (Value, Box<dyn Pullback>) {
        let (y, pullback) = self.recorded(inputs, differentiable, &[]);
        (y, Box::new(pullback))
    }
}

impl SecondOrder for BoundaryKernel {
    fn record_second(
        &self,
        inputs: &[Value],
        differentiable: &[bool],
        velocities: &[Option<Frame>],
    ) -> (Value, Option<Frame>, Box<dyn SecondPullback>) {
        let (y, pullback) = self.recorded(inputs, differentiable, velocities);
        let velocity = self.push(inputs, &y, velocities);
        (y, velocity, Box::new(pullback))
    }
}

/// What a boundary kernel recorded: its inputs, its output, and the velocities it moved along.
struct BoundaryPullback {
    kernel: Arc<BoundaryKernel>,
    inputs: Vec<Value>,
    y: Value,
    differentiable: Vec<bool>,
    /// The tangent input's velocity, for `⊕`.
    v: Option<Frame>,
    /// The first input's velocity in storage, `ẋ = B u` or `ẇ`, and for `⊕` and `π`, `ẏ`.
    dx: Option<Frame>,
    dy: Option<Frame>,
}

impl BoundaryPullback {
    fn gate(&self, j: usize, frame: Option<Frame>) -> Option<Frame> {
        if self.differentiable.get(j).copied().unwrap_or(false) {
            frame
        } else {
            None
        }
    }

    /// `m = (D L(y)[ẏ])ᵀ α`: the `β` of `𝓠` on the diagonal, both inputs moving along `ẏ`.
    fn chart_turn(&self, alpha: &Frame) -> Option<Frame> {
        let (y, q) = (&self.y, self.kernel.q());
        self.dy.as_ref().and_then(|dy| {
            let (_, [_, m]) = q.hessian([y, y], [Some(dy), Some(dy)], alpha);
            m
        })
    }
}

impl Pullback for BoundaryPullback {
    fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let k = &self.kernel;
        if k.flat {
            return (0..self.inputs.len())
                .map(|j| self.gate(j, Some(alpha.clone())))
                .collect();
        }
        match k.boundary {
            Boundary::Include => {
                vec![self.gate(0, Some(k.basis_transpose(&self.inputs[0], alpha)))]
            }
            Boundary::Retract => {
                let h = k.chart_transpose(&self.y, alpha);
                let [g1, g2] = k.r().cotangent([&self.inputs[0], &self.inputs[1]], &h);
                let into_x = g1.map(|g1| k.basis_transpose(&self.inputs[0], &g1));
                vec![self.gate(0, into_x), self.gate(1, g2)]
            }
            Boundary::Project => {
                let h = k.chart_transpose(&self.y, alpha);
                let [g] = k.p().cotangent([&self.inputs[0]], &h);
                vec![self.gate(0, g)]
            }
        }
    }
}

impl SecondPullback for BoundaryPullback {
    fn second(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let k = &self.kernel;
        let target = &k.target;
        // A flat declaration's second-order maps are zero.
        if k.flat {
            return vec![None; self.inputs.len()];
        }
        let x = &self.inputs[0];
        match k.boundary {
            Boundary::Include => {
                let zero = k.zero_tangent(alpha.rows());
                // (D B(x)[ẋ])ᵀ α: the β of 𝓡 at (x, 0), moving x along ẋ, at its second input.
                let hvp = self.dx.as_ref().and_then(|dx| {
                    let (_, [_, beta]) = k.r().hessian([x, &zero], [Some(dx), None], alpha);
                    beta
                });
                vec![self.gate(0, hvp)]
            }
            Boundary::Retract => {
                let (r, t, zero) = (k.r(), &self.inputs[1], k.zero_tangent(alpha.rows()));
                let h = k.chart_transpose(&self.y, alpha);
                let [g1, _] = r.cotangent([x, t], &h);
                let m = self.chart_turn(alpha);
                // (ġ1, ġ2): the cotangent of 𝓡 of m, and the β of 𝓡 along (ẋ, v) seeded with h.
                let [m1, m2] = match &m {
                    Some(m) => r.cotangent([x, t], m),
                    None => [None, None],
                };
                let [b1, b2] = if self.dx.is_some() || self.v.is_some() {
                    r.hessian([x, t], [self.dx.as_ref(), self.v.as_ref()], &h).1
                } else {
                    [None, None]
                };
                let (dg1, dg2) = (sum(m1, b1, target), sum(m2, b2, target));
                // hvp_x = Bᵀ ġ1 + (D B(x)[ẋ])ᵀ g1; hvp_t = ġ2.
                let turn = match (&self.dx, &g1) {
                    (Some(dx), Some(g1)) => {
                        let (_, [_, beta]) = r.hessian([x, &zero], [Some(dx), None], g1);
                        beta
                    }
                    _ => None,
                };
                let hvp_x = sum(dg1.map(|g| k.basis_transpose(x, &g)), turn, target);
                vec![self.gate(0, hvp_x), self.gate(1, dg2)]
            }
            Boundary::Project => {
                // (D² P(w)[ẇ])ᵀ L(y)ᵀ α, the β of 𝓟 along ẇ seeded with h, plus DP(w)ᵀ m.
                let p = k.p();
                let h = k.chart_transpose(&self.y, alpha);
                let curvature = self.dx.as_ref().and_then(|dw| {
                    let (_, [beta]) = p.hessian([x], [Some(dw)], &h);
                    beta
                });
                let pulled = self.chart_turn(alpha).and_then(|m| {
                    let [g] = p.cotangent([x], &m);
                    g
                });
                vec![self.gate(0, sum(curvature, pulled, target))]
            }
        }
    }
}
