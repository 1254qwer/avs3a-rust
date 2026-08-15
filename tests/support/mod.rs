/// Selects an exact RustFFT regression baseline for the active SIMD family.
///
/// RustFFT deliberately uses different kernels on x86-64 and AArch64. New
/// architectures must establish a reviewed baseline instead of silently
/// weakening the fingerprint assertion.
#[cfg(target_arch = "x86_64")]
pub(crate) fn expected_rustfft_fingerprint<T>(x86_64: T, _aarch64: T) -> T {
    x86_64
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn expected_rustfft_fingerprint<T>(_x86_64: T, aarch64: T) -> T {
    aarch64
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn expected_rustfft_fingerprint<T>(_x86_64: T, _aarch64: T) -> T {
    panic!(
        "no strict RustFFT fingerprint baseline for target architecture {}",
        std::env::consts::ARCH
    );
}
