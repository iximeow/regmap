# `regmap`

some well-known and known-to-be-good computer architectures, such as the Microchip PIC product line, or many of the AVR processor family, were fortunate enough to have architects that understood the power of a strategic alignment between the processor's register file and main memory. on these architectures, the foresight to synergize register and memory accesses reduces instruction complexity: to load or store registers a developer only has to know the instructions to operate on memory.

unfortunately, the architects at Intel who designed the 8086 did not appreciate the learnings of these architectures and did not synergize the register file with main memory. `regmap` handles this design oversight by allowing users to memory-map the processor's general-purpose registers (GPR).

think of `regmap` as "[niche-filling](https://doc.rust-lang.org/core/mem/union.MaybeUninit.html#layout), but for main memory."

## usage

call `regmap::map_registers()`, then use memory between 0 and 4096 to your delight. you probably want to run `regmap`-using binaries in `--release`; if you use `RegU64` and do not aggressively inline helper functions like `add_assign`, results will be unpredictable.

since you would never use addresses in the first page of memory otherwise, you may want to even add a stub like

```rust
#[cfg_attr(link_section = ".init_array")]
static MAP_REGS: extern fn() = {
  unsafe { regmap::map_registers(); }
};
```

so that the register mapping is always enabled, and you can't forget to actually run the function in your application. HOWEVER, if you are a library taking a dependency on `regmap`, first of all thank you! but attempting to use `.init_array` in your library is fraught with issues. you will likely want to encourage downstream users of your library to, themselves, `regmap::map_registers()`.

another interesting usage of `regmap` could be as a support mechanism for hand-written assembly using this feature. as long as `regmap::map_registers()` has been called in the process, any instruction referencing a mapped register will be correctly interpreted. a user could write optimized assembly, call `regmap::map_registers`, and proceed to use memory-mapped registers without issue.

if you intend to use `regmap` in a standalone manner (as a supporting library in a C program, for example), `cargo build --workspace --release` will produce a `./target/release/libregmap.a` that is suitable for linkage.

examples are present. the `examples/hello_world.rs` example can be run with `cargo run --example hello_world --release`. for a demonstration of using `regmap` outside a Rust program, `make joy` will build and run the example in `examples/example.s`.

## why

because it's funny

## how

the 0 page is generally reserved as no-execute no-write so accesses to it (via null pointer) don't cause "very weird" memory corruption. `regmap` repurposes this 4k (or larger, for hugepages, i guess) region to map parts of the amd64 register file. a fault is translated to a referenced address, the instruction is evaluated as if it was against the specified register, and execution continues none the wiser.

instead of rewriting the instruction stream or generating the instruction to execute, `regmap` emulates the instruction which had a memory reference. this means that `regmap` does not involve any extra allocations for handling, and is fairly self-contained in the signal handler implementation on which this feature relies.

## where

`regmap` currently makes some assumptions about ucontext layout, and is only known to correctly work on linux. in the future this restriction may be releaxed. `regmap` also cannot be used to execute SSE instructions. this restriction may also be relaxed in the future.

## bugs?

`regmap` probably doesn't have bugs. it's just an x86 disassembler and emulator hooked up to a signal handler. what could go wrong. if `regmap` incorrectly emulates an instruction, or should emulate an instruction differently in consideration of its special execution circumstances, please file an issue or email me at the email used for commits in this repo.

## TODO:
[ ] unaligned accesses to the register map should be handled. they currently panic. ideally, regmap should operate as if an apporopriately-sized register was formed from region being addressed.
[ ] non-qword operations are incorrectly emulated. `regmap` is not immediately impacted as `RegU64` forces u64 operations, but other sizes should be supported.
[ ] SSE operations are not handled. if they were handled it seems reasonable that they should work on GPR bytes. just the same.
