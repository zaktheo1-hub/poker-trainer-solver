# Poker Trainer solver

Offline batch solver that builds the postflop solution library for the Poker Trainer app. It wraps [postflop-solver](https://github.com/b-inary/postflop-solver) and is licensed AGPL-3.0-or-later, as that library requires.

```bash
cargo build --release
./target/release/poker-trainer-solver memory test/spots-test.json
./target/release/poker-trainer-solver solve test/spots-test.json out
```

- `memory` prints the RAM each spot needs, without solving.
- `solve` solves each spot from the flop and writes `out/<id>/`. A spot that already has a `meta.json` is skipped, so a batch can be stopped and resumed.

The output format is described at the top of `src/main.rs`. Flop and turn strategies are exported; rivers are solved on demand by the app.
