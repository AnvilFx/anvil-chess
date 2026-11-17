fn main() {
    // `anvil-engine bench [depth]` prints the bench signature and exits.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        let depth = args.get(2).and_then(|d| d.parse().ok());
        anvil_engine::bench::run(depth.unwrap_or(anvil_engine::bench::BENCH_DEPTH));
        return;
    }
    anvil_engine::uci::run();
}
