//! The process-wide float precision of a backend.
//!
//! Float tensors are created at their device's float policy, and Burn locks that policy the first
//! time any tensor is created on the device. A precision other than `B::FloatElem`'s therefore has
//! to be chosen before anything else runs. A test binary does it with
//! [`f64_test_binary!`](crate::f64_test_binary), which runs [`use_f64`] before `main`.

use burn::tensor::{DType, Element, set_default_dtypes};

use crate::{Backend, Tensor};

/// Make `f64` the float precision of `B`'s default device, or confirm that it already is.
///
/// # Panics
///
/// If the device's policy is already locked at another precision: a tensor was created before
/// this ran, or another precision was chosen earlier in the same process.
pub fn use_f64<B: Backend>() {
    let device = B::Device::default();
    let int = <B::IntElem as Element>::dtype();
    if set_default_dtypes::<B>(&device, DType::F64, int).is_err() {
        let dtype = Tensor::<B, 1>::zeros([1], &device).dtype();
        assert_eq!(
            dtype,
            DType::F64,
            "the float policy was locked at {dtype:?} before `use_f64` ran: a tensor was created \
             first, or two precisions share this process"
        );
    }
}

/// Run [`use_f64`] for `$B` before `main`, so that every test in the binary sees `f64` tensors.
///
/// Invoke once, at the top level of a test binary. One binary holds one precision.
#[macro_export]
macro_rules! f64_test_binary {
    ($B:ty) => {
        #[used]
        #[cfg_attr(
            any(target_os = "linux", target_os = "android", target_os = "freebsd"),
            unsafe(link_section = ".init_array")
        )]
        #[cfg_attr(
            target_vendor = "apple",
            unsafe(link_section = "__DATA,__mod_init_func")
        )]
        #[cfg_attr(windows, unsafe(link_section = ".CRT$XCU"))]
        static __SKORO_F64_POLICY: extern "C" fn() = {
            extern "C" fn init() {
                $crate::policy::use_f64::<$B>();
            }
            init
        };
    };
}
