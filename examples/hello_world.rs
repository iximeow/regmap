use regmap::registers::*;

fn main() {
    unsafe { regmap::map_registers(); }

    println!("rsp: {:x}", RSP.load());
    RSP.sub_assign(8);
    RIP.store(lol as u64);
}

fn lol() {
    println!("very good");
    RSP.sub_assign(8);
    RDI.store(1);
    RSI.store(1234);
    RIP.store(lol_args as u64);
}

fn lol_args(a1: u64, a2: u64) {
    println!("a1: {}, a2: {}", a1, a2);
    println!("goodbye!");
    std::process::exit(0);
}
