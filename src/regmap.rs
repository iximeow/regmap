//! `regmap` allows users to memory-map x86_64 registers.
//!
//! in many cases this will work, but for some uses (SIMD, or bulk memory instructions like
//! `fxsave` and `rep movs`) will have unpredictable results; likely an application crash.
//!
//! the vehicle is shaped for GPR arithmetic, bitwise operation, and control flow. please keep your
//! hands and feet inside the vehicle at all times and everything will be fine.

use core::ffi::c_void;
use core::ops::{BitAnd, BitOr};

use yaxpeax_x86;
use yaxpeax_x86::long_mode::{RegSpec, Operand, Instruction, register_class};
use yaxpeax_arch::{AddressBase, Decoder, LengthedInstruction};

#[repr(C)]
struct stack_t {
    ss_sp: *const c_void,
    ss_flags: i32,
    ss_size: usize,
}

#[repr(C)]
struct ucontext {
    uc_flags: u32,
    uc_link: *const ucontext,
    uc_stack: stack_t,
    uc_mcontext: sigcontext,
    // and uc_sigmask but w/e
}

/// quoth sigcontext.h:
/// `FPU environment matching the 64-bit FXSAVE layout.`
#[repr(C)]
struct _fpstate {
    cwd: u16,
    swd: u16,
    ftw: u16,
    fop: u16,
    rip: u64,
    rdp: u64,
    mxcsr: u32,
    mxcr_mask: u32,
    _st: [_fpxreg; 8],
    _xmm: [_xmmreg; 16],
    // __glibc_reserved1 that i don't feel like copying
}

#[repr(C)]
struct _fpxreg {
    items: [u16; 8]
}

#[repr(C)]
struct _xmmreg {
    element: [u32; 4]
}

#[repr(C)]
struct sigcontext {
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rdi: u64,
    rsi: u64,
    rbp: u64,
    rbx: u64,
    rdx: u64,
    rax: u64,
    rcx: u64,
    rsp: u64,
    rip: u64,
    eflags: u64,
    cs: u16,
    gs: u16,
    fs: u16,
    __pad0: u16,
    err: u64,
    trapno: u64,
    oldmask: u64,
    cr2: u64,
    fpstate: *const _fpstate,
    // more fields, __reserved1: [u64; 8]
}

#[repr(C)]
struct sigfault_t {
    _addr: *const c_void,
    _trapno: i32,
    // some other fields, maybe, but we don't care about em here
}

// siginfo_t but with _sifields known to be sigfault_t
#[repr(C)]
struct siginfo_sigfault_t {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _sifields: sigfault_t,
}

#[derive(Copy, Clone)]
pub struct RegU64(u64);

impl RegU64 {
    const fn new(addr: u64) -> Self {
        RegU64(addr)
    }

    #[inline(always)]
    pub fn load(&self) -> u64 {
        unsafe {
            core::ptr::read(self.0 as *const u64)
        }
    }
    #[inline(always)]
    pub fn store(&self, v: u64) {
        unsafe {
            core::ptr::write(self.0 as *mut u64, v)
        }
    }

    #[inline(always)]
    pub fn add(&self, other: u64) -> u64 {
        u64::wrapping_add(self.load(), other)
    }

    #[inline(always)]
    pub fn sub(&self, other: u64) -> u64 {
        u64::wrapping_sub(self.load(), other)
    }

    #[inline(always)]
    pub fn add_assign(&self, other: u64) {
        self.store(self.add(other));
    }

    #[inline(always)]
    pub fn sub_assign(&self, other: u64) {
        self.store(self.sub(other));
    }

    #[inline(always)]
    pub fn bitand(&self, other: u64) -> u64 {
        self.load() & other
    }

    #[inline(always)]
    pub fn bitor(&self, other: u64) -> u64 {
        self.load() | other
    }

    #[inline(always)]
    pub fn bitxor(&self, other: u64) -> u64 {
        self.load() ^ other
    }

    #[inline(always)]
    pub fn bitand_assign(&self, other: u64) {
        self.store(self.bitand(other))
    }

    #[inline(always)]
    pub fn bitor_assign(&self, other: u64) {
        self.store(self.bitor(other))
    }

    #[inline(always)]
    pub fn bitxor_assign(&self, other: u64) {
        self.store(self.bitxor(other))
    }
}

trait UcontextExt {
    fn read_reg(&self, reg: RegSpec) -> u64;
    fn write_reg(&mut self, reg: RegSpec, value: u64);
    fn do_binop<F: FnOnce(&ucontext, u64, u64) -> (u64, u64)>(&mut self, instr: &Instruction, addr: u64, f: F);
}

impl UcontextExt for ucontext {
    fn read_reg(&self, reg: RegSpec) -> u64 {
        if reg == RegSpec::RIP {
            self.uc_mcontext.rip
        } else if reg.class() == register_class::RFLAGS {
            self.uc_mcontext.eflags
        } else if reg.class() == register_class::Q {
            match reg.num() {
                0 => self.uc_mcontext.rax,
                1 => self.uc_mcontext.rcx,
                2 => self.uc_mcontext.rdx,
                3 => self.uc_mcontext.rbx,
                4 => self.uc_mcontext.rsp,
                5 => self.uc_mcontext.rbp,
                6 => self.uc_mcontext.rsi,
                7 => self.uc_mcontext.rdi,
                8 => self.uc_mcontext.r8,
                9 => self.uc_mcontext.r9,
                10 => self.uc_mcontext.r10,
                11 => self.uc_mcontext.r11,
                12 => self.uc_mcontext.r12,
                13 => self.uc_mcontext.r13,
                14 => self.uc_mcontext.r14,
                15 => self.uc_mcontext.r15,
                _ => { unreachable!("sins"); }
            }
        } else {
            panic!("crimes");
        }
    }

    fn write_reg(&mut self, reg: RegSpec, value: u64) {
        if reg == RegSpec::RIP {
            self.uc_mcontext.rip = value;
        } else if reg.class() == register_class::RFLAGS {
            self.uc_mcontext.eflags = value;
        } else if reg.class() == register_class::Q {
            match reg.num() {
                0 => { self.uc_mcontext.rax = value; },
                1 => { self.uc_mcontext.rcx = value; },
                2 => { self.uc_mcontext.rdx = value; },
                3 => { self.uc_mcontext.rbx = value; },
                4 => { self.uc_mcontext.rsp = value; },
                5 => { self.uc_mcontext.rbp = value; },
                6 => { self.uc_mcontext.rsi = value; },
                7 => { self.uc_mcontext.rdi = value; },
                8 => { self.uc_mcontext.r8 = value; },
                9 => { self.uc_mcontext.r9 = value; },
                10 => { self.uc_mcontext.r10 = value; },
                11 => { self.uc_mcontext.r11 = value; },
                12 => { self.uc_mcontext.r12 = value; },
                13 => { self.uc_mcontext.r13 = value; },
                14 => { self.uc_mcontext.r14 = value; },
                15 => { self.uc_mcontext.r15 = value; },
                _ => { unreachable!("sins"); }
            }
        } else {
            panic!("crimes");
        }
    }

    fn do_binop<F: FnOnce(&ucontext, u64, u64) -> (u64, u64)>(&mut self, instr: &Instruction, addr: u64, f: F) {
        // binop: two operands. the left operand is probably a written one. if it's not, `f` must
        // return the original value, because we're sure as heck gonna assign something to that
        // register.
        let lreg = match instr.operand(0) {
            Operand::Register(l) => l,
            _ => reg_for(addr),
        };
        let l = self.read_reg(lreg);

        // the right operand might be a register, might be memory, might be an immediate. if it's
        // memory, it's a memory mapped register (HOPEFULLY :)). so there's only possibly no right
        // register if the operand is an immediate. in that case we just have a value, no rreg.
        //
        // we won't use rreg anyway.
        let (_rreg, r) = match instr.operand(1) {
            Operand::ImmediateU64(u) => (None, u),
            Operand::ImmediateI64(u) => (None, u as u64),
            Operand::ImmediateU32(u) => (None, u as u64),
            Operand::ImmediateI32(u) => (None, u as i64 as u64),
            Operand::ImmediateU16(u) => (None, u as u64),
            Operand::ImmediateI16(u) => (None, u as i64 as u64),
            Operand::ImmediateU8(u) => (None, u as u64),
            Operand::ImmediateI8(u) => (None, u as i64 as u64),
            Operand::Register(r) => {
                (Some(r), self.read_reg(r))
            }
            // some memory operand. the register is given by the fault address.
            _ => {
                let r = reg_for(addr);
                (Some(r), self.read_reg(r))
            }
        };

        let (flags, v) = f(self, l, r);

//        panic!("writing {:x} to {}", v, lreg);
        self.write_reg(lreg, v);
        self.write_reg(RegSpec::rflags(), flags);
    }
}

fn reg_for(offset: u64) -> RegSpec {
    if offset == 128 {
        RegSpec::rip()
    } else if offset == 136 {
        RegSpec::rflags()
    } else {
        // todo: error checking
        RegSpec::q((offset / 8) as u8)
    }
}

const CF: u64 = 0x0001;
const PF: u64 = 0x0004;
const AF: u64 = 0x0010;
const ZF: u64 = 0x0040;
const SF: u64 = 0x0080;
const OF: u64 = 0x0800;
const BITWISE_FLAGS: u64 = CF | PF | ZF | SF | OF;
const ARITHMETIC_FLAGS: u64 = AF | BITWISE_FLAGS;
const BITWISE_MASK: u64 = !BITWISE_FLAGS;
const ARITHMETIC_MASK: u64 = !ARITHMETIC_FLAGS;

fn do_arithmetic_flags(x: u64, y: u64, f: fn(u64, u64) -> (u64, bool)) -> (u64, u64) {
    let (v, of) = f(x, y);
    let mut flags = 0u64;
    let pf = (v as u8).count_ones();
    if pf % 2 == 0 {
        flags |= PF;
    }
    if v > 0x8000_0000_0000_0000 {
        flags |= SF;
    }
    if v == 0 {
        flags |= ZF;
    }
    if of {
        flags |= CF;
    }
    let (xs, ys, vs) = (
        x & 0x8000_0000_0000_0000 != 0,
        y & 0x8000_0000_0000_0000 != 0,
        x & 0x8000_0000_0000_0000 != 0,
    );

    if xs == ys && xs != vs {
        flags |= OF;
    }

    if f(x & 0b1111, y & 0b1111).0 >= 0b1_0000 {
        flags |= AF;
    }

    (v, flags)
}

fn do_arithmetic_flags_with_carry(x: u64, y: u64, f: fn(u64, u64) -> (u64, bool), carry: bool) -> (u64, u64) {
    let c = if carry { 1 } else { 0 };
    let (v, of1) = f(x, y);
    let (v, of2) = f(v, c);
    let of = of1 || of2;

    let mut flags = 0u64;
    let pf = (v as u8).count_ones();
    if pf % 2 == 0 {
        flags |= PF;
    }
    if v > 0x8000_0000_0000_0000 {
        flags |= SF;
    }
    if v == 0 {
        flags |= ZF;
    }
    if of {
        flags |= CF;
    }
    let (xs, ys, vs) = (
        x & 0x8000_0000_0000_0000 != 0,
        y & 0x8000_0000_0000_0000 != 0,
        x & 0x8000_0000_0000_0000 != 0,
    );

    if xs == ys && xs != vs {
        flags |= OF;
    }

    if f(f(x & 0b1111, y & 0b1111).0, c).0 >= 0b1_0000 {
        flags |= AF;
    }

    (v, flags)
}

fn do_bitwise_flags(x: u64, y: u64, f: fn(u64, u64) -> u64) -> (u64, u64) {
    let v = f(x, y);

    let mut flags = 0u64;
    let pf = (v as u8).count_ones();
    if pf % 2 == 0 {
        flags |= PF;
    }
    if v > 0x8000_0000_0000_0000 {
        flags |= SF;
    }
    if v == 0 {
        flags |= ZF;
    }

    (v, flags)
}

// why ignore `signum`? because this is registerd for two signals, and we do the same thing for
// both.
extern "C" fn regmap_impl(_signum: i32, siginfo_ptr: *mut siginfo_sigfault_t, ucontext_ptr: *mut c_void) {
    let siginfo_ptr = siginfo_ptr as *mut siginfo_sigfault_t;
    let ucontext_ptr = ucontext_ptr as *mut ucontext;
    let ucontext = unsafe { ucontext_ptr.as_mut().expect("ucontext is set") } ;
    // find out where the fault PC was, we need to emulate that instruction
    let fault_rip = ucontext.uc_mcontext.rip;
    // find out the fault *address*. this simplifies emulation, since the MMU already assembled the
    // register address.
    let addr = unsafe { siginfo_ptr.as_mut().expect("siginfo valid")._sifields._addr } as usize as u64;

    // why is this perfectly fine, you may ask?
    //
    // `handle_signal` will handle SIGSEGV and SIGBUS, exclusviely. if these arise in normal
    // execution, we know the instruction at rip *did* decode, because it was run, and then
    // faulted. x86 instruction are no longer than 16 bytes, so we know that the first instruction
    // in the 16-byte sequence from `fault_rip` is valid.
    //
    // however, if someone sends a SIGSEGV or SIGBUS exactly when PC would advance to an
    // instruction that goes past the end of some memory range, this could, itself, fault in trying
    // to read inaccessible data.
    //
    // if there is a SIGSEGV inside `handle_signal` itself, this is the most probable cause.
    let rip_slice = unsafe { core::slice::from_raw_parts(fault_rip as *const u8, 16) };
    let instr = yaxpeax_x86::long_mode::InstDecoder::default().decode(rip_slice.iter().cloned()).unwrap();

    // advance past the faulting instruction. x86 instructions execute with `rip` being the start
    // of the following address, so rip must advance before any emulation occurs.
    ucontext.uc_mcontext.rip = ucontext.uc_mcontext.rip.wrapping_add((0 + instr.len()).to_linear() as u64);

    // lock prefixes on regmap addresses can be ignored: the memory access wants a register,
    // register operations are atomic, and the thread is paused while we handle the signal.
    // functionally, every `regmap`'d operation is "locked".
    use yaxpeax_x86::long_mode::Opcode;

    if instr.operand_count() == 0 {
        // no-operand instruction is nop, wait, something like that that doesn't access memory
        // so this was a spurious segfault and we can continue on.
        return;
    };

    // x86 has at most one memory operand, find it and emulate the instruction with the appropriate
    // `regmap`'d register swapped in.
    match instr.opcode() {
        Opcode::MOV => {
            ucontext.do_binop(&instr, addr, |ctx, _x, y| {
                (ctx.read_reg(RegSpec::rflags()), y)
            });
        }
        Opcode::ADD => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (v, flags) = do_arithmetic_flags(x, y, u64::overflowing_add);
                (((ctx.read_reg(RegSpec::rflags()) & ARITHMETIC_MASK) | flags), v)
            });
        }
        Opcode::SUB => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (v, flags) = do_arithmetic_flags(x, y, u64::overflowing_sub);
                (((ctx.read_reg(RegSpec::rflags()) & ARITHMETIC_MASK) | flags), v)
            });
        }
        Opcode::ADC => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (v, flags) = do_arithmetic_flags_with_carry(x, y, u64::overflowing_add, (ctx.read_reg(RegSpec::rflags()) & CF) != 0);
                (((ctx.read_reg(RegSpec::rflags()) & ARITHMETIC_MASK) | flags), v)
            });
        }
        Opcode::SBB => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (v, flags) = do_arithmetic_flags_with_carry(x, y, u64::overflowing_sub, (ctx.read_reg(RegSpec::rflags()) & CF) != 0);
                (((ctx.read_reg(RegSpec::rflags()) & ARITHMETIC_MASK) | flags), v)
            });
        }
        Opcode::OR => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (v, flags) = do_bitwise_flags(x, y, BitOr::bitor);
                (((ctx.read_reg(RegSpec::rflags()) & BITWISE_MASK) | flags), v)
            });
        }
        Opcode::AND => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (v, flags) = do_bitwise_flags(x, y, BitOr::bitor);
                (((ctx.read_reg(RegSpec::rflags()) & BITWISE_MASK) | flags), v)
            });
        }
        Opcode::CMP => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (_v, flags) = do_arithmetic_flags(x, y, u64::overflowing_sub);
                (((ctx.read_reg(RegSpec::rflags()) & ARITHMETIC_MASK) | flags), x)
            });
        }
        Opcode::TEST => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let (_v, flags) = do_bitwise_flags(x, y, BitAnd::bitand);
                (((ctx.read_reg(RegSpec::rflags()) & BITWISE_MASK) | flags), x)
            });
        }
        Opcode::NOT => {
            // no, `not <reg>` cannot segfault, but we COULD get a sigsegv at the worst time from
            // someone throwing sigsegv's around. look. we're here to emulate instructions. i don't
            // care who asks for it.
            let lreg = match instr.operand(0) {
                Operand::Register(l) => l,
                _ => reg_for(addr),
            };
            let l = ucontext.read_reg(lreg);
            let l = !l;
            ucontext.write_reg(lreg, l);
        }
        Opcode::MOVSX_b => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let l = x & 0xffff_ffff_ffff_0000;
                let r = (y & 0xff) as i8 as i16 as u64;
                (ctx.read_reg(RegSpec::rflags()), l | r)
            });
        }
        Opcode::MOVSX_w => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let l = x & 0xffff_ffff_0000_0000;
                let r = (y & 0xffff) as i16 as i32 as u64;
                (ctx.read_reg(RegSpec::rflags()), l | r)
            });
        }
        Opcode::MOVZX_b => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let l = x & 0xffff_ffff_ffff_0000;
                let r = y & 0xff;
                (ctx.read_reg(RegSpec::rflags()), l | r)
            });
        }
        Opcode::MOVZX_w => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let l = x & 0xffff_ffff_0000_0000;
                let r = y & 0xffff;
                (ctx.read_reg(RegSpec::rflags()), l | r)
            });
        }
        Opcode::JMP => {
            // if this is a `jmp <rel>`, update rip and bail out before we'd try using a register.
            // since jmp has encodings without a memory operand, it's possible this segfault wasn't
            // due to a real memory-mapped address but some kinda spurious segv sent our way.
            let lreg = match instr.operand(0) {
                Operand::ImmediateI8(i) => {
                    let rip = ucontext.read_reg(RegSpec::rip()).wrapping_add(i as u64);
                    ucontext.write_reg(RegSpec::rip(), rip);
                    return;
                }
                Operand::ImmediateI32(i) => {
                    let rip = ucontext.read_reg(RegSpec::rip()).wrapping_add(i as u64);
                    ucontext.write_reg(RegSpec::rip(), rip);
                    return;
                }
                Operand::Register(l) => l,
                _ => reg_for(addr),
            };
            let l = ucontext.read_reg(lreg);
            ucontext.write_reg(RegSpec::rip(), l);
        }
        Opcode::CALL => {
            let mut rsp = ucontext.read_reg(RegSpec::rsp());
            rsp -= 8;
            unsafe { *(rsp as *mut u64) = ucontext.read_reg(RegSpec::rip()) };
            ucontext.write_reg(RegSpec::rsp(), rsp);

            let lreg = match instr.operand(0) {
                Operand::ImmediateI32(i) => {
                    let rip = ucontext.read_reg(RegSpec::rip()).wrapping_add(i as u64);
                    ucontext.write_reg(RegSpec::rip(), rip);
                    return;
                }
                Operand::Register(l) => l,
                _ => reg_for(addr),
            };
            let l = ucontext.read_reg(lreg);
            ucontext.write_reg(RegSpec::rip(), l);
        }
        Opcode::RETURN => {
            let mut rsp = ucontext.read_reg(RegSpec::rsp());
            let ra = unsafe { *(rsp as *const u64) };
            rsp += 8;
            ucontext.write_reg(RegSpec::rsp(), rsp);
            ucontext.write_reg(RegSpec::rip(), ra);
        }
        Opcode::CMOVA => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::above(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVB => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::below(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVG => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::greater(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVGE => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::greater_equal(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVL => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::greater_equal(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVLE => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::greater(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVNA => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::above(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVNB => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::below(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVNO => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::overflow(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVNP => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::parity(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVNS => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::signed(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVNZ => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if !predicate::zero(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVO => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::overflow(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVP => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::parity(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVS => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::signed(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        Opcode::CMOVZ => {
            ucontext.do_binop(&instr, addr, |ctx, x, y| {
                let rflags = ctx.read_reg(RegSpec::rflags());
                if predicate::zero(rflags) {
                    (ctx.read_reg(RegSpec::rflags()), y)
                } else {
                    (ctx.read_reg(RegSpec::rflags()), x)
                }
            });
        }
        _ => {
            panic!("unhandled instruction: {}", instr);
        }
    }
}

mod predicate {
    pub(crate) fn above(flags: u64) -> bool {
        flags & (crate::regmap::CF | crate::regmap::ZF) == 0
    }
    pub(crate) fn below(flags: u64) -> bool {
        flags & crate::regmap::CF != 0
    }
    pub(crate) fn greater(flags: u64) -> bool {
        let bits = flags & (crate::regmap::SF | crate::regmap::OF);
        bits & crate::regmap::ZF == 0 && (bits == 0 || bits == crate::regmap::SF | crate::regmap::OF)
    }
    pub(crate) fn greater_equal(flags: u64) -> bool {
        let bits = flags & (crate::regmap::SF | crate::regmap::OF);
        bits == 0 || bits == crate::regmap::SF | crate::regmap::OF
    }
    pub(crate) fn overflow(flags: u64) -> bool {
        flags & crate::regmap::OF != 0
    }
    pub(crate) fn parity(flags: u64) -> bool {
        flags & crate::regmap::PF != 0
    }
    pub(crate) fn signed(flags: u64) -> bool {
        flags & crate::regmap::SF != 0
    }
    pub(crate) fn zero(flags: u64) -> bool {
        flags & crate::regmap::ZF != 0
    }
}

pub mod registers {
    use crate::regmap::RegU64;
    /// a memory reference for the `rax` register.
    pub static RAX: RegU64 = RegU64::new(0);
    /// a memory reference for the `rcx` register.
    pub static RCX: RegU64 = RegU64::new(8);
    /// a memory reference for the `rdx` register.
    pub static RDX: RegU64 = RegU64::new(16);
    /// a memory reference for the `rbx` register.
    pub static RBX: RegU64 = RegU64::new(24);
    /// a memory reference for the `rsp` register.
    pub static RSP: RegU64 = RegU64::new(32);
    /// a memory reference for the `rbp` register.
    pub static RBP: RegU64 = RegU64::new(40);
    /// a memory reference for the `rsi` register.
    pub static RSI: RegU64 = RegU64::new(48);
    /// a memory reference for the `rdi` register.
    pub static RDI: RegU64 = RegU64::new(56);
    /// a memory reference for the `r8` register.
    pub static R8: RegU64 = RegU64::new(64);
    /// a memory reference for the `r9` register.
    pub static R9: RegU64 = RegU64::new(72);
    /// a memory reference for the `r10` register.
    pub static R10: RegU64 = RegU64::new(80);
    /// a memory reference for the `r11` register.
    pub static R11: RegU64 = RegU64::new(88);
    /// a memory reference for the `r12` register.
    pub static R12: RegU64 = RegU64::new(96);
    /// a memory reference for the `r13` register.
    pub static R13: RegU64 = RegU64::new(104);
    /// a memory reference for the `r14` register.
    pub static R14: RegU64 = RegU64::new(112);
    /// a memory reference for the `r15` register.
    pub static R15: RegU64 = RegU64::new(120);
    /// a memory reference for the `rip` register.
    ///
    /// yes, assigning to `rip`` will move the program counter, immediately. be careful to not violate
    /// ABI guarantees, such as that `rsp` is 16-byte aligned at function entry.
    pub static RIP: RegU64 = RegU64::new(128);
    /// a memory reference feor the `rflags` register.
    pub static RFLAGS: RegU64 = RegU64::new(136);
}

/// enable memory-mapped registers for this process.
///
/// calling this repeatedly won't be an error, but you shouldn't need to. if you for some reason
/// trample `SIGSEGV` or `SIGBUS` handlers, you could call this again to reinitialize register
/// memory-mapping.
#[no_mangle]
pub unsafe extern "C" fn map_registers() {
    fn __sigmask(sig: u32) -> u64 {
        1u64 << ((sig as u64 - 1) % ((8 * core::mem::size_of::<u32>()) as u64))
    }

    let mut sa = core::mem::MaybeUninit::<libc::sigaction>::uninit();
    libc::sigaction(libc::SIGSEGV, core::ptr::null_mut(), sa.as_mut_ptr());
    let mut sa = sa.assume_init();
    libc::sigemptyset(&mut sa.sa_mask as *mut libc::sigset_t);
    sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART | libc::SA_ONSTACK;
    sa.sa_sigaction = regmap_impl as usize;

    libc::sigaction(libc::SIGSEGV, &sa as *const libc::sigaction, core::ptr::null_mut());
    libc::sigaction(libc::SIGBUS, &sa as *const libc::sigaction, core::ptr::null_mut());
    let altstack = libc::stack_t {
        ss_sp: ALTSTACK.as_mut_ptr() as *mut c_void,
        ss_flags: 0,
        ss_size: ALTSTACK.len(),
    };
    libc::sigaltstack(&altstack as *const libc::stack_t, core::ptr::null_mut());
}

static mut ALTSTACK: [u8; 8192] = [0; 8192];
