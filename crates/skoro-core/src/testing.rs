//! The checks the core's tests run, public so that a feature crate checks its own operations as
//! the core checks its primitives.

use skoro_tensor::{Device, Frame, HostData, HostValue, Precision, Target, Value};

use crate::{Error, Kernel, Key, Lowered, Lowering, Operation, Space};

#[doc(hidden)]
pub use skoro_tensor as __tensor;

/// How closely a check holds a result, per precision.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tolerance {
    /// The step of a central difference.
    pub step: f64,
    /// The relative error allowed of it, against the larger of the reference's magnitude and one.
    pub relative: f64,
}

impl Tolerance {
    /// How closely a single-precision result is held to the double-precision one.
    pub const AGREEMENT: f64 = 1e-4;

    pub fn of(precision: Precision) -> Tolerance {
        match precision {
            Precision::Double => Tolerance {
                step: 1e-6,
                relative: 1e-7,
            },
            Precision::Single => Tolerance {
                step: 1e-2,
                relative: 5e-3,
            },
        }
    }
}

/// The Flex device at `precision`.
pub fn flex(precision: Precision) -> Result<Target, String> {
    Target::new(Device::flex(), precision).map_err(|e| e.to_string())
}

/// The test adapter at single precision, or an error naming the missing adapter.
#[cfg(feature = "wgpu")]
pub fn wgpu() -> Result<Target, String> {
    let device = skoro_tensor::test_adapter(skoro_tensor::wgpu_devices())?;
    Target::new(device, Precision::Single).map_err(|e| e.to_string())
}

/// Instantiates each test body at single and at double precision on the Flex device, and, with
/// the core's `wgpu` feature on, at single precision on the test adapter: `name::f32`,
/// `name::f64` and `name::wgpu`.
///
/// ```ignore
/// skoro_core::precision_tests! {
///     fn add(target) {
///         check_operation(&Add, &inputs, &target).unwrap();
///     }
/// }
/// ```
#[macro_export]
macro_rules! precision_tests {
    ($(fn $name:ident($target:ident) $body:block)*) => {$(
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            fn f32() {
                let $target = $crate::testing::flex(
                    $crate::testing::__tensor::Precision::Single,
                ).unwrap();
                $body
            }

            #[test]
            fn f64() {
                let $target = $crate::testing::flex(
                    $crate::testing::__tensor::Precision::Double,
                ).unwrap();
                $body
            }

            $crate::__if_wgpu! {
                #[test]
                fn wgpu() {
                    let $target = $crate::testing::wgpu().unwrap();
                    $body
                }
            }
        }
    )*};
}

/// Its tokens, when the core is built with `wgpu`; nothing otherwise. The feature is read where
/// the core is built, so a crate without a feature of that name may use the test macro.
#[cfg(feature = "wgpu")]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_wgpu {
    ($($tokens:tt)*) => { $($tokens)* };
}

#[cfg(not(feature = "wgpu"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_wgpu {
    ($($tokens:tt)*) => {};
}

/// `rows` values of `space`, one per row, drawn from the key of `seed`: continuous values
/// uniform in `(low, high)` (positive values in `(low, high)` too, which must then be positive),
/// categories uniform, and integers in `[low, high)`.
pub fn random(space: &Space, rows: usize, seed: u64, low: f64, high: f64) -> HostValue {
    let storage = space.storage();
    let uniforms = Key::from_seed(seed).uniforms(rows * storage);
    let data = match space {
        Space::Categorical(n) => HostData::Discrete(
            uniforms
                .iter()
                .map(|u| (u * *n as f64).floor() as i64)
                .collect(),
        ),
        Space::Integer => HostData::Discrete(
            uniforms
                .iter()
                .map(|u| (low + u * (high - low)).floor() as i64)
                .collect(),
        ),
        _ => HostData::Continuous(uniforms.iter().map(|u| low + u * (high - low)).collect()),
    };
    match HostValue::new(Some(rows), storage, data) {
        Some(value) => value,
        None => unreachable!("the data holds rows times storage values"),
    }
}

/// The check of the tables of primitives, applied to one operation at the given inputs, each a
/// space and a value, on `target`:
///
/// - its tangent map is the derivative of its forward map, by central differences with the inputs
///   moved by their spaces' retractions and the outputs read in the chart at the output;
/// - its cotangent map is the transpose of its tangent map, input by input;
/// - its second-order map is the derivative of its cotangent map along the velocities of every
///   input, the cotangent held fixed;
/// - an input marked not differentiable receives `None`, and a missing tangent or velocity gives
///   what a zero one gives;
/// - at single precision, every result is held to the double-precision one on the host.
///
/// The retractions are those of the core's spaces, computed on the host values. An input of an
/// embedded type is refused.
pub fn check_operation(
    operation: &dyn Operation,
    inputs: &[(Space, HostValue)],
    target: &Target,
) -> Result<(), String> {
    let check = Check::new(operation, inputs, target)?;
    check.run()?;
    if target.precision() == Precision::Single {
        check.agree()?;
    }
    Ok(())
}

/// The columns of the tangent, cotangent and velocity frames a check draws.
const COLUMNS: usize = 2;

struct Check {
    spaces: Vec<Space>,
    output: Space,
    rows: usize,
    /// Every input's values, one per row.
    points: Vec<HostData>,
    target: Target,
    tolerance: Tolerance,
    kernel: Kernel,
    host: Kernel,
    name: &'static str,
}

/// Every map of a kernel at one set of inputs and frames, read back on the host.
struct Results {
    forward: Vec<f64>,
    tangent: Vec<f64>,
    cotangents: Vec<Vec<f64>>,
    seconds: Vec<Vec<f64>>,
}

fn describe(error: Error) -> String {
    error.to_string()
}

/// The kernel `operation` lowers to on `target`, at inputs of `spaces`.
pub fn lower(
    operation: &dyn Operation,
    spaces: &[Space],
    target: &Target,
) -> Result<Kernel, String> {
    match operation
        .lower(spaces, &mut Lowering::new(target))
        .map_err(describe)?
    {
        Lowered::Kernel(kernel) => Ok(kernel),
        Lowered::Composite(composite) => Err(format!(
            "{} lowers to the composite {}, which a compiled program checks",
            operation.name(),
            composite.name()
        )),
    }
}

impl Check {
    fn new(
        operation: &dyn Operation,
        inputs: &[(Space, HostValue)],
        target: &Target,
    ) -> Result<Check, String> {
        let spaces: Vec<Space> = inputs.iter().map(|(space, _)| space.clone()).collect();
        let output = operation.output(&spaces).map_err(describe)?;
        let rows = inputs
            .iter()
            .filter_map(|(_, value)| value.rows())
            .max()
            .unwrap_or(1);
        let points = inputs
            .iter()
            .map(|(space, value)| rows_of(space, value, rows))
            .collect::<Result<_, _>>()?;
        Ok(Check {
            kernel: lower(operation, &spaces, target)?,
            host: lower(operation, &spaces, &Target::host())?,
            spaces,
            output,
            rows,
            points,
            target: target.clone(),
            tolerance: Tolerance::of(target.precision()),
            name: operation.name(),
        })
    }

    fn continuous(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.spaces.len()).filter(|&j| self.spaces[j].is_continuous())
    }

    fn dimension(&self, j: usize) -> usize {
        self.spaces[j].dimension().unwrap_or(0)
    }

    fn values(&self, points: &[HostData], target: &Target) -> Result<Vec<Value>, String> {
        points
            .iter()
            .zip(&self.spaces)
            .map(|(data, space)| {
                let value = HostValue::new(Some(self.rows), space.storage(), data.clone())
                    .ok_or("an input's values do not fit its space")?;
                value
                    .load(self.rows, target)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| "an input does not fit the batch".to_string())
            })
            .collect()
    }

    /// A random frame `[rows, dimension, columns]` of each continuous input, `None` for a discrete
    /// one.
    fn frames(&self, seed: u64, columns: usize) -> Vec<Option<Vec<f64>>> {
        (0..self.spaces.len())
            .map(|j| {
                self.spaces[j]
                    .is_continuous()
                    .then(|| signed(seed + j as u64, self.rows * self.dimension(j) * columns))
            })
            .collect()
    }

    fn frame(&self, data: &[f64], dimension: usize, columns: usize, target: &Target) -> Frame {
        Frame::from_host(data, self.rows, dimension, columns, target)
    }

    fn input_frames(
        &self,
        frames: &[Option<Vec<f64>>],
        columns: usize,
        target: &Target,
    ) -> Vec<Option<Frame>> {
        frames
            .iter()
            .enumerate()
            .map(|(j, f)| {
                f.as_ref()
                    .map(|f| self.frame(f, self.dimension(j), columns, target))
            })
            .collect()
    }

    fn forward(
        &self,
        kernel: &Kernel,
        points: &[HostData],
        target: &Target,
    ) -> Result<Vec<f64>, String> {
        let value = kernel.forward().forward(&self.values(points, target)?);
        continuous(&value)
    }

    /// The inputs moved by their retractions along column `column` of `frames`, scaled by `h`.
    fn moved(
        &self,
        frames: &[Option<Vec<f64>>],
        column: usize,
        columns: usize,
        h: f64,
    ) -> Result<Vec<HostData>, String> {
        self.points
            .iter()
            .enumerate()
            .map(|(j, data)| match (data, &frames[j]) {
                (HostData::Continuous(x), Some(frame)) => {
                    let (s, d) = (self.spaces[j].storage(), self.dimension(j));
                    let mut moved = Vec::with_capacity(x.len());
                    for r in 0..self.rows {
                        let t: Vec<f64> = (0..d)
                            .map(|i| h * frame[(r * d + i) * columns + column])
                            .collect();
                        moved.extend(retract(&self.spaces[j], &x[r * s..(r + 1) * s], &t)?);
                    }
                    Ok(HostData::Continuous(moved))
                }
                _ => Ok(data.clone()),
            })
            .collect()
    }

    /// The output at `y` read in the chart at `at`, row by row.
    fn local(&self, y: &[f64], at: &[f64]) -> Result<Vec<f64>, String> {
        let s = self.output.storage();
        let mut out = Vec::new();
        for r in 0..self.rows {
            out.extend(local(
                &self.output,
                &y[r * s..(r + 1) * s],
                &at[r * s..(r + 1) * s],
            )?);
        }
        Ok(out)
    }

    fn close(
        &self,
        found: &[f64],
        expected: &[f64],
        relative: f64,
        what: &str,
    ) -> Result<(), String> {
        close(found, expected, relative).map_err(|error| {
            format!(
                "{} at {:?}: {what}: {error}",
                self.name,
                self.target.precision()
            )
        })
    }

    fn run(&self) -> Result<(), String> {
        let target = &self.target;
        let y = self.forward(&self.kernel, &self.points, target)?;
        if y.len() != self.rows * self.output.storage() {
            return Err(format!(
                "{}: the output does not have the output space's storage",
                self.name
            ));
        }
        let Some(first) = self.kernel.first() else {
            return Ok(());
        };
        let d_out = self.output.dimension().ok_or_else(|| {
            format!(
                "{}: a differentiable operation has a discrete output",
                self.name
            )
        })?;
        let values = self.values(&self.points, target)?;
        let h = self.tolerance.step;
        let rel = self.tolerance.relative;

        // Tangent and cotangent maps, input by input.
        let tangents = self.frames(1, COLUMNS);
        let alpha = signed(100, self.rows * d_out * COLUMNS);
        let alpha_frame = self.frame(&alpha, d_out, COLUMNS, target);
        let all: Vec<bool> = self.spaces.iter().map(Space::is_continuous).collect();
        let (_, pullback) = first.record(&values, &all);
        let cotangents = pullback.cotangent(&alpha_frame);
        for j in self.continuous() {
            let only: Vec<Option<Vec<f64>>> = (0..self.spaces.len())
                .map(|i| if i == j { tangents[i].clone() } else { None })
                .collect();
            let pushed = tangent_of(first, &values, &self.input_frames(&only, COLUMNS, target))?;
            let mut differences = vec![0.0; pushed.len()];
            for c in 0..COLUMNS {
                let up = self.forward(&self.kernel, &self.moved(&only, c, COLUMNS, h)?, target)?;
                let down =
                    self.forward(&self.kernel, &self.moved(&only, c, COLUMNS, -h)?, target)?;
                let (up, down) = (self.local(&up, &y)?, self.local(&down, &y)?);
                for r in 0..self.rows {
                    for i in 0..d_out {
                        differences[(r * d_out + i) * COLUMNS + c] =
                            (up[r * d_out + i] - down[r * d_out + i]) / (2.0 * h);
                    }
                }
            }
            self.close(
                &pushed,
                &differences,
                rel,
                &format!("the tangent map of input {j} against central differences"),
            )?;

            // A missing tangent is a zero one.
            let zeros: Vec<Option<Vec<f64>>> = (0..self.spaces.len())
                .map(|i| match &tangents[i] {
                    Some(t) if i != j => Some(vec![0.0; t.len()]),
                    _ => only[i].clone(),
                })
                .collect();
            let with_zeros =
                tangent_of(first, &values, &self.input_frames(&zeros, COLUMNS, target))?;
            self.close(
                &pushed,
                &with_zeros,
                Tolerance::AGREEMENT,
                "a missing tangent against a zero one",
            )?;

            // The cotangent map is the transpose: ⟨α, J t⟩ = ⟨Jᵀ α, t⟩ per row and column.
            let pulled = cotangents[j]
                .as_ref()
                .ok_or_else(|| {
                    format!(
                        "{}: input {j} is differentiable and received nothing",
                        self.name
                    )
                })?
                .to_host();
            let (t, d) = (
                tangents[j].as_deref().unwrap_or_default(),
                self.dimension(j),
            );
            self.close(
                &pairing(&pulled, t, self.rows, d),
                &pairing(&alpha, &pushed, self.rows, d_out),
                rel,
                &format!("the cotangent map of input {j} against the transpose of the tangent map"),
            )?;

            // An input not differentiable receives nothing.
            let mut gated = all.clone();
            gated[j] = false;
            let (_, pullback) = first.record(&values, &gated);
            let gated = pullback.cotangent(&alpha_frame);
            if gated[j].is_some()
                || (0..gated.len()).any(|i| all[i] && i != j && gated[i].is_none())
            {
                return Err(format!(
                    "{}: the cotangent map ignores input {j}'s gate",
                    self.name
                ));
            }
        }
        for (j, cotangent) in cotangents.iter().enumerate() {
            if !all[j] && cotangent.is_some() {
                return Err(format!(
                    "{}: discrete input {j} received a cotangent",
                    self.name
                ));
            }
        }

        let Some(second) = self.kernel.second() else {
            return Ok(());
        };
        let velocities = self.frames(200, COLUMNS);
        let alpha = signed(300, self.rows * d_out);
        let alpha_frame = self.frame(&alpha, d_out, 1, target);
        let velocity_frames = self.input_frames(&velocities, COLUMNS, target);
        let (_, velocity, pullback) = second.record_second(&values, &all, &velocity_frames);
        let expected_velocity = tangent_of(first, &values, &velocity_frames)?;
        let velocity = velocity
            .map(|v| v.to_host())
            .unwrap_or_else(|| vec![0.0; expected_velocity.len()]);
        self.close(
            &velocity,
            &expected_velocity,
            Tolerance::AGREEMENT,
            "the velocity record_second gives against the tangent map",
        )?;
        let first_cotangent = first.record(&values, &all).1.cotangent(&alpha_frame);
        let second_cotangent = pullback.cotangent(&alpha_frame);
        for j in self.continuous() {
            let (a, b) = (host(&second_cotangent[j]), host(&first_cotangent[j]));
            self.close(
                &a,
                &b,
                Tolerance::AGREEMENT,
                &format!("record_second's cotangent of input {j} against record's"),
            )?;
        }
        let hvps = pullback.second(&alpha_frame);
        for j in self.continuous() {
            let d = self.dimension(j);
            let found = host(&hvps[j])
                .into_iter()
                .chain(std::iter::repeat(0.0))
                .take(self.rows * d * COLUMNS)
                .collect::<Vec<_>>();
            let mut differences = vec![0.0; self.rows * d * COLUMNS];
            for c in 0..COLUMNS {
                let mut pulled = Vec::new();
                for sign in [1.0, -1.0] {
                    let moved =
                        self.values(&self.moved(&velocities, c, COLUMNS, sign * h)?, target)?;
                    let cotangent = first.record(&moved, &all).1.cotangent(&alpha_frame);
                    pulled.push(host(&cotangent[j]));
                }
                for r in 0..self.rows {
                    for i in 0..d {
                        differences[(r * d + i) * COLUMNS + c] =
                            (pulled[0][r * d + i] - pulled[1][r * d + i]) / (2.0 * h);
                    }
                }
            }
            self.close(&found, &differences, rel, &format!("the second-order map of input {j} against central differences of the cotangent map"))?;

            // A missing velocity is a zero one.
            let mut missing = velocity_frames.clone();
            missing[j] = None;
            let mut zero = velocity_frames.clone();
            zero[j] = Some(self.frame(&vec![0.0; self.rows * d * COLUMNS], d, COLUMNS, target));
            let (_, _, a) = second.record_second(&values, &all, &missing);
            let (_, _, b) = second.record_second(&values, &all, &zero);
            let (a, b) = (a.second(&alpha_frame), b.second(&alpha_frame));
            for i in self.continuous() {
                let size = self.rows * self.dimension(i) * COLUMNS;
                self.close(&padded(&a[i], size), &padded(&b[i], size), Tolerance::AGREEMENT, &format!("the second-order map of input {i} with input {j}'s velocity missing against zero"))?;
            }

            // An input not differentiable receives nothing.
            let mut gated = all.clone();
            gated[j] = false;
            let (_, _, pullback) = second.record_second(&values, &gated, &velocity_frames);
            if pullback.second(&alpha_frame)[j].is_some()
                || pullback.cotangent(&alpha_frame)[j].is_some()
            {
                return Err(format!(
                    "{}: the second-order map ignores input {j}'s gate",
                    self.name
                ));
            }
        }
        for (j, hvp) in hvps.iter().enumerate() {
            if !all[j] && hvp.is_some() {
                return Err(format!(
                    "{}: discrete input {j} received a second-order term",
                    self.name
                ));
            }
        }
        Ok(())
    }

    /// Every result at this target, held to the same result in double precision on the host.
    fn agree(&self) -> Result<(), String> {
        let here = self.results(&self.kernel, &self.target)?;
        let there = self.results(&self.host, &Target::host())?;
        let what = "against double precision on the host";
        self.close(
            &here.forward,
            &there.forward,
            Tolerance::AGREEMENT,
            &format!("the forward map {what}"),
        )?;
        self.close(
            &here.tangent,
            &there.tangent,
            Tolerance::AGREEMENT,
            &format!("the tangent map {what}"),
        )?;
        for (a, b) in here.cotangents.iter().zip(&there.cotangents) {
            self.close(
                a,
                b,
                Tolerance::AGREEMENT,
                &format!("the cotangent map {what}"),
            )?;
        }
        for (a, b) in here.seconds.iter().zip(&there.seconds) {
            self.close(
                a,
                b,
                Tolerance::AGREEMENT,
                &format!("the second-order map {what}"),
            )?;
        }
        Ok(())
    }

    fn results(&self, kernel: &Kernel, target: &Target) -> Result<Results, String> {
        let values = self.values(&self.points, target)?;
        let forward = continuous_or_discrete(&kernel.forward().forward(&values));
        let (mut tangent, mut cotangents, mut seconds) = (Vec::new(), Vec::new(), Vec::new());
        if let (Some(first), Some(d_out)) = (kernel.first(), self.output.dimension()) {
            let all: Vec<bool> = self.spaces.iter().map(Space::is_continuous).collect();
            let frames = self.input_frames(&self.frames(1, COLUMNS), COLUMNS, target);
            tangent = tangent_of(first, &values, &frames)?;
            let alpha = self.frame(
                &signed(100, self.rows * d_out * COLUMNS),
                d_out,
                COLUMNS,
                target,
            );
            cotangents = first
                .record(&values, &all)
                .1
                .cotangent(&alpha)
                .iter()
                .map(host)
                .collect();
            if let Some(second) = kernel.second() {
                let velocities = self.input_frames(&self.frames(200, COLUMNS), COLUMNS, target);
                let alpha = self.frame(&signed(300, self.rows * d_out), d_out, 1, target);
                let (_, _, pullback) = second.record_second(&values, &all, &velocities);
                seconds = pullback.second(&alpha).iter().map(host).collect();
            }
        }
        Ok(Results {
            forward,
            tangent,
            cotangents,
            seconds,
        })
    }
}

/// An input's values, one per row of the batch.
fn rows_of(space: &Space, value: &HostValue, rows: usize) -> Result<HostData, String> {
    if matches!(value.rows(), Some(n) if n != rows) {
        return Err("the inputs pin different batches".to_string());
    }
    if value.storage() != space.storage() {
        return Err(format!(
            "an input of {space:?} has storage {}",
            value.storage()
        ));
    }
    let copies = if value.rows().is_some() { 1 } else { rows };
    Ok(match value.data() {
        HostData::Continuous(data) => HostData::Continuous(data.repeat(copies)),
        HostData::Discrete(data) => HostData::Discrete(data.repeat(copies)),
    })
}

/// `n` values uniform in `(-1, 1)` from the key of `seed`.
fn signed(seed: u64, n: usize) -> Vec<f64> {
    Key::from_seed(seed)
        .uniforms(n)
        .into_iter()
        .map(|u| 2.0 * u - 1.0)
        .collect()
}

fn continuous(value: &Value) -> Result<Vec<f64>, String> {
    match value {
        Value::Continuous(points) => Ok(points.to_host()),
        Value::Discrete(_) => Err("a continuous output was expected".to_string()),
    }
}

fn continuous_or_discrete(value: &Value) -> Vec<f64> {
    match value {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(integers) => integers.to_host().into_iter().map(|i| i as f64).collect(),
    }
}

fn host(frame: &Option<Frame>) -> Vec<f64> {
    frame.as_ref().map(Frame::to_host).unwrap_or_default()
}

/// A second-order result, `None` read as zero.
fn padded(frame: &Option<Frame>, size: usize) -> Vec<f64> {
    frame
        .as_ref()
        .map(Frame::to_host)
        .unwrap_or_else(|| vec![0.0; size])
}

fn tangent_of(
    kernel: &dyn crate::FirstOrder,
    values: &[Value],
    frames: &[Option<Frame>],
) -> Result<Vec<f64>, String> {
    let (_, tangent) = kernel.tangent(values, frames);
    tangent
        .map(|t| t.to_host())
        .ok_or_else(|| "a tangent was given and none came out".to_string())
}

/// `⟨a, b⟩` per row and column of two frames `[rows, d, columns]`.
fn pairing(a: &[f64], b: &[f64], rows: usize, d: usize) -> Vec<f64> {
    let columns = a.len() / (rows * d).max(1);
    let mut out = vec![0.0; rows * columns];
    for r in 0..rows {
        for i in 0..d {
            for c in 0..columns {
                let at = (r * d + i) * columns + c;
                out[r * columns + c] += a[at] * b[at];
            }
        }
    }
    out
}

fn close(found: &[f64], expected: &[f64], relative: f64) -> Result<(), String> {
    if found.len() != expected.len() {
        return Err(format!(
            "{} values where {} were expected",
            found.len(),
            expected.len()
        ));
    }
    let scale = expected.iter().fold(1.0f64, |m, e| m.max(e.abs()));
    let (at, error) = found
        .iter()
        .zip(expected)
        .map(|(f, e)| (f - e).abs())
        .enumerate()
        .fold((0, 0.0f64), |best, (i, e)| {
            if e > best.1 || e.is_nan() {
                (i, e)
            } else {
                best
            }
        });
    if error <= relative * scale {
        Ok(())
    } else {
        Err(format!(
            "{} where {} was expected, at {at} of {}, an error of {error:e} against {:e} allowed",
            found[at],
            expected[at],
            found.len(),
            relative * scale
        ))
    }
}

/// `x ⊕ t` for one value of a core space, by the table of retractions.
fn retract(space: &Space, x: &[f64], t: &[f64]) -> Result<Vec<f64>, String> {
    match space {
        Space::Real | Space::Vector(_) | Space::Matrix(_, _) => {
            Ok(x.iter().zip(t).map(|(x, t)| x + t).collect())
        }
        Space::Positive(_) => Ok(x.iter().zip(t).map(|(x, t)| x * t.exp()).collect()),
        Space::Product(_) => {
            let mut out = Vec::with_capacity(x.len());
            for block in space.blocks() {
                out.extend(retract(&block.space, &x[block.storage], &t[block.tangent])?);
            }
            Ok(out)
        }
        Space::Categorical(_) | Space::Integer => Err(format!("{space:?} has no retraction")),
    }
}

/// `y ⊖ x` for values of a core space.
fn local(space: &Space, y: &[f64], x: &[f64]) -> Result<Vec<f64>, String> {
    match space {
        Space::Real | Space::Vector(_) | Space::Matrix(_, _) => {
            Ok(y.iter().zip(x).map(|(y, x)| y - x).collect())
        }
        Space::Positive(_) => Ok(y.iter().zip(x).map(|(y, x)| y.ln() - x.ln()).collect()),
        Space::Product(_) => {
            let mut out = Vec::new();
            for block in space.blocks() {
                out.extend(local(
                    &block.space,
                    &y[block.storage.clone()],
                    &x[block.storage],
                )?);
            }
            Ok(out)
        }
        Space::Categorical(_) | Space::Integer => Err(format!("{space:?} has no retraction")),
    }
}
