//! Embedded types: declared by a retraction `R` and its inverse `Q`, written as composites, and
//! reached through the boundary operations, whose derivative maps are passes over those
//! composites' programs.
//!
//! Write `𝓡` and `𝓠` for the programs of `R` and `Q`, each with two roots. For the inclusion `ι`
//! at a point stored as `x`, `B u` is `𝓡.tangent((x, 0); (0, u))`, `Bᵀ α` is the cotangent of `𝓡`
//! at `(x, 0)` at its second input, and the second-order map is the `β` of
//! `𝓡.hessian((x, 0); (B u, 0); α)` at its second input. The retraction `⊕` runs `𝓡`, and its maps
//! compose those of `𝓡` with `L(y)`, read off `𝓠` on the diagonal `(y, y)`, as the implementation
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
    flat: bool,
}

impl Embedded {
    /// `retraction`, `Vector(n)` and `Vector(d)` to `Vector(n)`, is the `R` of the mathematics;
    /// `inverse`, two of `Vector(n)` to `Vector(d)`, its `Q`. Any other signature is refused
    /// (`InvalidDeclaration`).
    pub fn declare(
        name: &'static str,
        params: Vec<Param>,
        storage: usize,
        dimension: usize,
        retraction: Arc<Composite>,
        inverse: Arc<Composite>,
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
        if inverse.inputs() != [n.clone(), n] || *inverse.output() != d {
            return Err(invalid(
                "its inverse does not take two of Vector(n) to Vector(d)",
            ));
        }
        Ok(Arc::new(Embedded {
            name,
            params,
            storage,
            dimension,
            retraction,
            inverse,
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
}

/// The inclusion `ι`: a value of the type, read as the `Vector(n)` it is stored as.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Include(Arc<Embedded>);

/// The retraction `⊕`: a value of the type and `Vector(d)`, to the type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Retract(Arc<Embedded>);

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

/// The kernel of a boundary operation on one target, of the most its inner passes admit.
fn lower(
    boundary: Boundary,
    declaration: &Arc<Embedded>,
    lowering: &mut Lowering<'_>,
) -> Result<Lowered, Error> {
    let target = lowering.target().clone();
    let (r, q) = if declaration.flat {
        (None, None)
    } else {
        let r = Inner::new(lowering.compile(&declaration.retraction)?);
        let q = Inner::new(lowering.compile(&declaration.inverse)?);
        (Some(r), Some(q))
    };
    let kernel = Arc::new(BoundaryKernel {
        boundary,
        n: declaration.storage,
        d: declaration.dimension,
        target,
        r,
        q,
    });
    Ok(Lowered::Kernel(match kernel.order() {
        2 => Kernel::Two(kernel),
        1 => Kernel::One(kernel),
        _ => Kernel::Zero(kernel),
    }))
}

/// A compiled inner program of two roots and one target, and its passes.
struct Inner {
    program: Arc<Program>,
    roots: [NodeId; 2],
    output: NodeId,
}

/// What an inner pass gives at each root, `None` read as zero by the caller.
type AtRoots = [Option<Frame>; 2];

fn admitted<T>(result: Result<T, Error>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => unreachable!("an inner pass admitted when the kernel was lowered: {error}"),
    }
}

impl Inner {
    fn new(program: Arc<Program>) -> Inner {
        let free = program.free();
        let (Some(a), Some(b)) = (free.first(), free.get(1)) else {
            unreachable!("a declaration's composites take two inputs")
        };
        let roots = [a.node, b.node];
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
        let (all, out) = (self.roots, [self.output]);
        if self.admits(Pass::Hessian {
            seeded: &all,
            targets: &out,
        }) {
            2
        } else if self.admits(Pass::Tangent { seeded: &all })
            && self.admits(Pass::Cotangent { targets: &out })
        {
            1
        } else {
            0
        }
    }

    fn values(&self, a: &Value, b: &Value) -> Values {
        Values::new()
            .with(self.roots[0], a.clone())
            .with(self.roots[1], b.clone())
    }

    fn frames(&self, frames: [Option<&Frame>; 2]) -> Frames {
        let mut out = Frames::new();
        for (root, frame) in self.roots.iter().zip(frames) {
            if let Some(frame) = frame {
                out.insert(*root, frame.clone());
            }
        }
        out
    }

    fn forward(&self, a: &Value, b: &Value) -> Value {
        let mut values = admitted(self.program.forward(&self.values(a, b)));
        match values.remove(self.output) {
            Some(value) => value,
            None => unreachable!("a program computes its target"),
        }
    }

    fn tangent(&self, a: &Value, b: &Value, tangents: [Option<&Frame>; 2]) -> Frame {
        let (_, frames) = admitted(
            self.program
                .tangent(&self.values(a, b), &self.frames(tangents)),
        );
        match frames.get(self.output) {
            Some(frame) => frame.clone(),
            None => unreachable!("a tangent pass gives a frame at its target"),
        }
    }

    fn cotangent(&self, a: &Value, b: &Value, seed: &Frame) -> AtRoots {
        let seeds = Frames::new().with(self.output, seed.clone());
        let (_, frames) = admitted(self.program.cotangent(&self.values(a, b), &seeds));
        self.roots.map(|root| frames.get(root).cloned())
    }

    /// `(α, β)` at each root.
    fn hessian(
        &self,
        a: &Value,
        b: &Value,
        velocities: [Option<&Frame>; 2],
        seed: &Frame,
    ) -> (AtRoots, AtRoots) {
        let seeds = Frames::new().with(self.output, seed.clone());
        let second = admitted(self.program.hessian(
            &self.values(a, b),
            &self.frames(velocities),
            &seeds,
        ));
        (
            self.roots.map(|root| second.alpha.get(root).cloned()),
            self.roots.map(|root| second.beta.get(root).cloned()),
        )
    }
}

struct BoundaryKernel {
    boundary: Boundary,
    n: usize,
    d: usize,
    target: Target,
    /// The programs of `R` and `Q`; `None` for a flat declaration, whose maps are the identity.
    r: Option<Inner>,
    q: Option<Inner>,
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

impl BoundaryKernel {
    fn order(&self) -> usize {
        match (&self.r, &self.q, self.boundary) {
            (None, _, _) => 2,
            (Some(r), _, Boundary::Include) => {
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
            (Some(r), Some(q), Boundary::Retract) => r.order().min(q.order()),
            (Some(_), None, _) => 0,
        }
    }

    fn inner(&self) -> (&Inner, &Inner) {
        match (&self.r, &self.q) {
            (Some(r), Some(q)) => (r, q),
            _ => unreachable!("a curved declaration has its programs"),
        }
    }

    fn zero_tangent(&self, rows: usize) -> Value {
        Value::Continuous(Points::zeros(rows, self.d, &self.target))
    }

    /// `B u`, at `x`.
    fn basis(&self, x: &Value, u: &Frame) -> Frame {
        let (r, _) = self.inner();
        let zero = self.zero_tangent(u.rows());
        r.tangent(x, &zero, [None, Some(u)])
    }

    /// `Bᵀ g`, at `x`.
    fn basis_transpose(&self, x: &Value, g: &Frame) -> Frame {
        let (r, _) = self.inner();
        let zero = self.zero_tangent(g.rows());
        match r.cotangent(x, &zero, g) {
            [_, Some(frame)] => frame,
            _ => unreachable!("the cotangent pass reaches the tangent input"),
        }
    }

    /// `L(y) ẏ`: the tangent of `𝓠` on the diagonal `(y, y)`, along its second input.
    fn chart(&self, y: &Value, dy: &Frame) -> Frame {
        let (_, q) = self.inner();
        q.tangent(y, y, [None, Some(dy)])
    }

    /// `L(y)ᵀ α`: the cotangent of `𝓠` on the diagonal at its second input.
    fn chart_transpose(&self, y: &Value, alpha: &Frame) -> Frame {
        let (_, q) = self.inner();
        match q.cotangent(y, y, alpha) {
            [_, Some(frame)] => frame,
            _ => unreachable!("the cotangent pass reaches the second input"),
        }
    }

    fn value(&self, inputs: &[Value]) -> Value {
        match (self.boundary, &self.r) {
            (Boundary::Include, _) => inputs[0].clone(),
            (Boundary::Retract, None) => {
                let (x, t) = (continuous(&inputs[0]), continuous(&inputs[1]));
                let y = x.clone().into_burn() + t.clone().into_burn();
                match Points::from_burn(y, &self.target) {
                    Ok(points) => Value::Continuous(points),
                    Err(error) => unreachable!("a flat retraction keeps its target: {error}"),
                }
            }
            (Boundary::Retract, Some(r)) => r.forward(&inputs[0], &inputs[1]),
        }
    }

    fn push(&self, inputs: &[Value], y: &Value, tangents: &[Option<Frame>]) -> Option<Frame> {
        let target = &self.target;
        match (self.boundary, &self.r) {
            (Boundary::Include, None) => tangents[0].clone(),
            (Boundary::Include, Some(_)) => tangents[0].as_ref().map(|u| self.basis(&inputs[0], u)),
            (Boundary::Retract, None) => sum(tangents[0].clone(), tangents[1].clone(), target),
            (Boundary::Retract, Some(r)) => {
                let dx = tangents[0].as_ref().map(|u| self.basis(&inputs[0], u));
                if dx.is_none() && tangents[1].is_none() {
                    return None;
                }
                let dy = r.tangent(&inputs[0], &inputs[1], [dx.as_ref(), tangents[1].as_ref()]);
                Some(self.chart(y, &dy))
            }
        }
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

impl BoundaryKernel {
    fn recorded(
        &self,
        inputs: &[Value],
        differentiable: &[bool],
        velocities: &[Option<Frame>],
    ) -> (Value, BoundaryPullback) {
        let y = self.value(inputs);
        let velocity = |j: usize| velocities.get(j).cloned().flatten();
        let (dx, dy) = match (self.boundary, &self.r) {
            (_, None) => (None, None),
            (Boundary::Include, Some(_)) => (velocity(0).map(|u| self.basis(&inputs[0], &u)), None),
            (Boundary::Retract, Some(r)) => {
                let dx = velocity(0).map(|u| self.basis(&inputs[0], &u));
                let v = velocity(1);
                let dy = (dx.is_some() || v.is_some())
                    .then(|| r.tangent(&inputs[0], &inputs[1], [dx.as_ref(), v.as_ref()]));
                (dx, dy)
            }
        };
        let pullback = BoundaryPullback {
            kernel: self.clone_shallow(),
            inputs: inputs.to_vec(),
            y: y.clone(),
            differentiable: differentiable.to_vec(),
            v: velocity(1),
            dx,
            dy,
        };
        (y, pullback)
    }

    fn clone_shallow(&self) -> Arc<BoundaryKernel> {
        Arc::new(BoundaryKernel {
            boundary: self.boundary,
            n: self.n,
            d: self.d,
            target: self.target.clone(),
            r: self.r.as_ref().map(|r| Inner {
                program: r.program.clone(),
                roots: r.roots,
                output: r.output,
            }),
            q: self.q.as_ref().map(|q| Inner {
                program: q.program.clone(),
                roots: q.roots,
                output: q.output,
            }),
        })
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
    /// `ẋ = B u`, and for `⊕`, `ẏ`.
    dx: Option<Frame>,
    dy: Option<Frame>,
}

impl Pullback for BoundaryPullback {
    fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let k = &self.kernel;
        let gate = |j: usize, frame: Option<Frame>| -> Option<Frame> {
            if self.differentiable.get(j).copied().unwrap_or(false) {
                frame
            } else {
                None
            }
        };
        match (k.boundary, &k.r) {
            (Boundary::Include, None) => vec![gate(0, Some(alpha.clone()))],
            (Boundary::Include, Some(_)) => {
                vec![gate(0, Some(k.basis_transpose(&self.inputs[0], alpha)))]
            }
            (Boundary::Retract, None) => {
                vec![gate(0, Some(alpha.clone())), gate(1, Some(alpha.clone()))]
            }
            (Boundary::Retract, Some(r)) => {
                let h = k.chart_transpose(&self.y, alpha);
                let [g1, g2] = r.cotangent(&self.inputs[0], &self.inputs[1], &h);
                let into_x = g1.map(|g1| k.basis_transpose(&self.inputs[0], &g1));
                vec![gate(0, into_x), gate(1, g2)]
            }
        }
    }
}

impl SecondPullback for BoundaryPullback {
    fn second(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let k = &self.kernel;
        let target = &k.target;
        let gate = |j: usize, frame: Option<Frame>| -> Option<Frame> {
            if self.differentiable.get(j).copied().unwrap_or(false) {
                frame
            } else {
                None
            }
        };
        let (x, zero) = (&self.inputs[0], k.zero_tangent(alpha.rows()));
        match (k.boundary, &k.r, &k.q) {
            // A flat declaration's second-order maps are zero.
            (_, None, _) => vec![None; self.inputs.len()],
            (Boundary::Include, Some(r), _) => {
                // (D B(x)[ẋ])ᵀ α: the β of 𝓡 at (x, 0), moving x along ẋ, at its second input.
                let hvp = self.dx.as_ref().and_then(|dx| {
                    let (_, [_, beta]) = r.hessian(x, &zero, [Some(dx), None], alpha);
                    beta
                });
                vec![gate(0, hvp)]
            }
            (Boundary::Retract, Some(r), Some(q)) => {
                let t = &self.inputs[1];
                let h = k.chart_transpose(&self.y, alpha);
                let [g1, _] = r.cotangent(x, t, &h);
                // m = (D L(y)[ẏ])ᵀ α: 𝓠 on the diagonal, both inputs moving along ẏ.
                let m = self.dy.as_ref().and_then(|dy| {
                    let (_, [_, m]) = q.hessian(&self.y, &self.y, [Some(dy), Some(dy)], alpha);
                    m
                });
                // (ġ1, ġ2): the cotangent of 𝓡 of m, and the β of 𝓡 along (ẋ, v) seeded with h.
                let [m1, m2] = match &m {
                    Some(m) => r.cotangent(x, t, m),
                    None => [None, None],
                };
                let [b1, b2] = if self.dx.is_some() || self.v.is_some() {
                    r.hessian(x, t, [self.dx.as_ref(), self.v.as_ref()], &h).1
                } else {
                    [None, None]
                };
                let (dg1, dg2) = (sum(m1, b1, target), sum(m2, b2, target));
                // hvp_x = Bᵀ ġ1 + (D B(x)[ẋ])ᵀ g1; hvp_t = ġ2.
                let turn = match (&self.dx, &g1) {
                    (Some(dx), Some(g1)) => r.hessian(x, &zero, [Some(dx), None], g1).1[1].clone(),
                    _ => None,
                };
                let hvp_x = sum(dg1.map(|g| k.basis_transpose(x, &g)), turn, target);
                vec![gate(0, hvp_x), gate(1, dg2)]
            }
            (_, Some(_), None) => vec![None; self.inputs.len()],
        }
    }
}
