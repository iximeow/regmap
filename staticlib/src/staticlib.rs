#![no_std]

#[panic_handler]
#[cold]
fn panic(_panic: &core::panic::PanicInfo) -> ! {
    unsafe { libc::exit(125) }
}

#[path = "../../src/regmap.rs"]
mod regmap;

// not strictly necessary, since staticlib items are `no_mangle` functions anyway.
pub use regmap::*;
