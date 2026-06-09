// Anvil engine binary: a UCI server. Plug into any UCI GUI (Cutechess, Arena)
// or drive it by hand: `uci`, `position startpos moves e2e4`, `go movetime 1000`.

fn main() {
    anvil_engine::uci::run();
}
