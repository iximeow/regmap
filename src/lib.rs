//! `regmap` allows users to memory-map x86_64 registers.
//!
//! in many cases this will work, but for some uses (SIMD, or bulk memory instructions like
//! `fxsave` and `rep movs`) will have unpredictable results; likely an application crash.
//!
//! the vehicle is shaped for GPR arithmetic, bitwise operation, and control flow. pleas keep your
//! hands and feet inside the vehicle at all times and everything will be fine.

mod regmap;

// re-export regmap innards for use from std-participating crates. no_std builds as a staticlib
// present the same items with a little extra scaffolding.
pub use regmap::*;
