//! macOS Thread Quality of Service (QoS) acceleration for Apple Silicon Performance core scheduling.
//!
//! Elevates inference and decision threads to `QOS_CLASS_USER_INITIATED`, guaranteeing
//! Performance core (P-core) scheduling on Apple Silicon M-series chips and avoiding
//! Efficiency core (E-core) scheduling penalties.

/// Elevates the current thread's Quality of Service to `QOS_CLASS_USER_INITIATED` on Apple Silicon.
/// On non-macOS or non-aarch64 architectures, this is an inlined zero-cost no-op.
#[inline]
pub fn elevate_thread_qos() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0);
    }
}
