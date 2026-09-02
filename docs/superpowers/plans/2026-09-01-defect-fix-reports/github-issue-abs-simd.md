<!--
Ready to paste as the body of a new issue at https://github.com/tracel-ai/burn/issues/new.
Suggested title:

    abs() silently reorders elements on a shared, non-standard-layout tensor >= 32 elements (burn-ndarray SIMD path)
-->

## Describe the bug

`burn-ndarray`'s SIMD fast path for `abs()` silently **reorders tensor elements** — with no
panic, shape error, or other signal — under a specific, reachable combination of conditions:

1. the input tensor is **non-standard-layout but memory-contiguous** (e.g. column-major — the
   result of `select()` on an axis other than 0), and
2. the input has **>= 32 elements** (the SIMD length threshold), and
3. the input's backing buffer is **not uniquely held** — i.e. something else still holds a
   reference/clone to the same `ArcArray` when `abs()` is called (very easy to hit in practice:
   any caller that clones the tensor before or after calling `abs()`, or any framework that keeps
   node outputs alive for a backward pass, does this).

Under (1)+(2) alone the op is still correct, because `try_unary_simd` takes an **in-place**
sub-path that never changes the buffer's layout. The bug is specifically in the **owned**
sub-path, taken only when the buffer isn't uniquely held (condition 3) — see Root cause below.

Below 32 elements, on a standard-layout tensor, or when the buffer is uniquely held, `abs()` is
correct.

## To Reproduce

Minimal, self-contained, no workspace/patches needed — just `burn = "0.21.0"` with the `ndarray`
backend:

**`Cargo.toml`**
```toml
[package]
name = "burn-abs-repro"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { version = "=0.21.0", default-features = false, features = ["ndarray", "std", "simd"] }
```
(`"simd"` is enabled explicitly here only because this repro depends on the `burn` facade crate
with `default-features = false`; if you depend on `burn-ndarray` directly instead — the more
common setup for a backend-specific crate — `simd` is already on by default, per
`burn-ndarray`'s own `default = ["std", "simd", "multi-threads"]`.)

**`src/main.rs`**
```rust
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};

type B = NdArray<f64>;

fn main() {
    let device = Default::default();

    // A 4x16 tensor of sequential values (64 elements).
    let values: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let base = Tensor::<B, 2>::from_data(TensorData::new(values, [4, 16]), &device);

    // Gathering along axis 1 (not axis 0) routes through `ndarray::concatenate`, which
    // returns a column-major (non-standard-layout) array: 8 columns selected -> [4, 8] = 32
    // elements, right at the SIMD threshold.
    let idx: Vec<i64> = (0..16).step_by(2).collect();
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(idx, [8]), &device);
    let gathered = base.select(1, indices);

    // Keep a second reference to the same underlying buffer alive across the `.abs()` call.
    // burn-ndarray's SIMD dispatch only takes the (buggy) reallocating path when the buffer
    // isn't uniquely held -- an in-place fast path is used instead when it is, and that one
    // happens to be layout-safe. This clone is what makes the bug visible.
    let expected_tensor = gathered.clone();
    let actual_tensor = gathered.abs();

    // abs() on all-positive input is the identity function mathematically, so any mismatch
    // below is pure element reordering, not a sign bug.
    let expected: Vec<f64> = expected_tensor.into_data().to_vec().unwrap();
    let actual: Vec<f64> = actual_tensor.into_data().to_vec().unwrap();

    println!("expected: {:?}", expected);
    println!("actual:   {:?}", actual);
    assert_eq!(expected, actual, "abs() must not reorder elements");
    println!("OK - no reordering detected");
}
```

```
$ cargo run
expected: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0]
actual:   [0.0, 16.0, 32.0, 48.0, 2.0, 18.0, 34.0, 50.0, 4.0, 20.0, 36.0, 52.0, 6.0, 22.0, 38.0, 54.0, 8.0, 24.0, 40.0, 56.0, 10.0, 26.0, 42.0, 58.0, 12.0, 28.0, 44.0, 60.0, 14.0, 30.0, 46.0, 62.0]

thread 'main' panicked at src/main.rs:26:5:
assertion `left == right` failed: abs() must not reorder elements
```

`actual` is exactly the **transpose** of the logical 4x8 array, flattened — `expected`'s row-major
layout `[[0,2,4,6,8,10,12,14], [16,18,20,22,24,26,28,30], ...]` comes back read down each column
first (`[0,16,32,48], [2,18,34,50], ...`).

## Expected behavior

`abs()` should return values in the tensor's logical order regardless of its memory layout or
buffer sharing state, matching its non-SIMD/slow-path behavior.

## Root cause

Read directly from `burn-ndarray` 0.21.0 source (`crates.io` package, not `git main` — line
numbers may drift there):

1. `src/ops/simd/base.rs:17` — `pub fn should_use_simd(len: usize) -> bool { len >= 32 }`: the
   32-element threshold.
2. `src/ops/base.rs:974` — `abs` dispatches via
   `dispatch_unary_simd!(E, VecAbs, tensor, i8, i16, i32, f32, f64)`, so f32 and f64 both take the
   SIMD path once eligible.
3. `src/ops/simd/unary.rs:85` (`try_unary_simd`) — the layout guard is
   `input.as_slice_memory_order().is_none()`. This returns `Some(..)` (i.e. "eligible") for a
   **column-major** array too — contiguous in memory, but not in logical order.
4. `src/ops/simd/unary.rs:96-104` — **this is the actual trigger.** `try_unary_simd` picks
   between two sub-paths:
   ```rust
   let out = if size_of::<T>() == size_of::<Out>()
       && align_of::<T>() >= align_of::<Out>()
       && input.is_unique()
   {
       unsafe { unary_scalar_simd_inplace::<T, Out, Op>(input) }
   } else {
       unary_scalar_simd_owned::<T, Out, Op>(input)
   };
   ```
   - `unary_scalar_simd_inplace` (taken when `input.is_unique()` is true, i.e. no other clone of
     the `ArcArray` is alive) mutates the existing buffer in place through
     `as_slice_memory_order_mut()`. Because the buffer's own shape/strides metadata is untouched,
     this is layout-safe regardless of column-major-ness — **this path is correct**.
   - `unary_scalar_simd_owned` (taken whenever the buffer is *not* uniquely held) is where the bug
     lives (`src/ops/simd/unary.rs:126-133`):
     ```rust
     fn unary_scalar_simd_owned<T, Out, Op>(input: SharedArray<T>) -> SharedArray<Out> {
         let mut out = unsafe { ArrayD::uninit(input.shape()).assume_init() };
         let input = input.as_slice_memory_order().unwrap();
         let out_slice = out.as_slice_memory_order_mut().unwrap();
         unary_slice::<T, Out, Op>(input, out_slice, PhantomData);
         out.into_shared()
     }
     ```
     `out` is allocated with `ArrayD::uninit(input.shape())`, which gets **default (standard/
     row-major) strides** — it ignores `input`'s actual strides entirely. The op then reads
     `input` in *its* memory order (column-major) and writes into `out` in *its* memory order,
     which for a freshly allocated standard-layout array is the same as its *logical* order. A
     value at logical position `p`, physically stored at memory offset `m` in the column-major
     input, is read at slice-index `m` and written to slice-index `m` of `out` — but `out`'s
     slice-index `m` **is** its logical position `m`. The value silently lands at the wrong
     logical position.

   Compare to every other SIMD-dispatched binary/comparison op in the same module
   (`src/ops/simd/binary.rs:216`, `cmp.rs:147,277`, `binary_elemwise.rs:311`), which all build
   their output via `uninit_array_like(&input)` (`src/ops/simd/base.rs:63`) instead — that helper
   explicitly copies the **input's own strides** onto the freshly allocated output, which is
   layout-preserving and does not have this bug. `unary_scalar_simd_owned` is the one call site
   that uses `ArrayD::uninit(input.shape())` directly instead of `uninit_array_like`.

The layout producer, for context: `NdArrayOps::select` (`src/ops/base.rs:922`) delegates to
`ndarray::ArrayBase::select`, which for an axis other than 0 goes through `ndarray::concatenate`
and returns a column-major array. `burn-ndarray`'s own `concatenate` op (`src/ops/base.rs:370`) is
already aware of this exact hazard and works around it for itself, citing `ndarray`/burn issue
`#1053`:
```rust
// Transform column-major layout into row-major (standard) layout. (fix #1053)
let shape = array.shape().into_shape();
Self::reshape(array, shape)
```
`select` never received the same treatment — so anything downstream of `select` on axis != 0 that
calls `abs()` (or any other single-elem-in/single-elem-out unary op sharing this dispatch — see
"Related" below) while a second reference to the buffer is alive can hit this.

## Suggested fix

The most targeted fix: make `unary_scalar_simd_owned` build its output the same way every other
SIMD op in this module already does — via `uninit_array_like(&input)` instead of
`ArrayD::uninit(input.shape())` — so the output's strides mirror the input's instead of defaulting
to standard layout. That's a one-line change and brings this function's allocation strategy in
line with `binary.rs`/`cmp.rs`/`binary_elemwise.rs`.

A complementary, more defensive fix at the producer: apply the same `#1053` normalization
`concatenate` already does to `NdArrayOps::select`'s output too, so it never returns column-major
in the first place.

## Related, in the same dispatch family

On the **f32** backend specifically, `recip` shares the exact same `dispatch_unary_simd!` /
`try_unary_simd` machinery (its dispatch lists only `f32`, which is why an f64-only probe wouldn't
see it), so it has the identical reordering defect under the same three conditions. It also has a
second, unrelated issue: even on a correctly laid out, *uniquely held*, contiguous f32 tensor, the
SIMD `recip` above 32 elements is a low-precision approximation:

```
contiguous f32 recip n=31 exact=true  max_rel_err=0e0
contiguous f32 recip n=32 exact=false max_rel_err=2.7e-4
```

~2.7e-4 relative error is far looser than typical 1e-5/1e-6 tolerances. Happy to open that as a
separate issue if preferred — flagging here since it surfaced from the same investigation and the
same macro.

## Environment

- `burn` / `burn-ndarray`: 0.21.0 (current stable at time of filing; 0.22.0 is still pre-release)
- Backend: `NdArray<f64>` (also confirmed on `f32`)
- OS: Linux x86_64, CPU with AVX2 (SIMD fast path confirmed taken via `is_accelerated`)
