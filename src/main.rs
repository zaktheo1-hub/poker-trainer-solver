//! Batch solver for the Poker Trainer postflop library.
//!
//!   poker-trainer-solver solve <spots.json> <out_dir> [--max-iterations N]
//!   poker-trainer-solver memory <spots.json>
//!
//! Each spot is solved from the flop with postflop-solver. The flop and turn
//! strategies are exported; river strategies are not (the app solves rivers on
//! demand). A spot whose output folder already has a meta.json is skipped, so a
//! batch can be stopped and resumed.
//!
//! Output per spot, in <out_dir>/<id>/:
//!   meta.json            ranges, tree settings, solve statistics, list of turn cards
//!   flop.bin.gz          every flop decision node
//!   turn-<card>.bin.gz   every turn decision node for that turn card
//!
//! A .bin file is: u32 little-endian header length, a JSON header, then the body.
//! The header lists the nodes; for each node the body holds, in order:
//!   strategy  u8  [actions x hands]   frequency, 0 to 255
//!   loss      u8  [actions x hands]   EV lost against the best action, in steps of
//!                                     LOSS_STEP_PCT percent of the pot at the node (capped)
//!   best_ev   i16 [hands]             EV of the best action, in units of EV_UNIT chips
//! Hands are in the order listed in meta.json for the acting player.

use flate2::{write::GzEncoder, Compression};
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

/// One step of the stored EV loss, as a percentage of the pot.
const LOSS_STEP_PCT: f32 = 0.25;
/// Stored best-EV unit, in chips. With 100 chips per big blind this is 0.02bb.
const EV_UNIT: f32 = 2.0;

#[derive(Deserialize, Clone)]
struct Sizes {
    bet: String,
    raise: String,
}

#[derive(Deserialize, Clone)]
struct Spot {
    id: String,
    /// Out-of-position and in-position ranges in postflop-solver notation.
    oop_range: String,
    ip_range: String,
    /// e.g. "Ks7h2d"
    flop: String,
    /// Chips; the app uses 100 chips per big blind.
    pot: i32,
    stack: i32,
    flop_sizes: Sizes,
    turn_sizes: Sizes,
    river_sizes: Sizes,
    /// Target exploitability as a percentage of the pot.
    #[serde(default = "default_target")]
    target_pct: f32,
}

fn default_target() -> f32 {
    0.5
}

#[derive(Serialize)]
struct NodeHeader {
    /// Actions from the start of the street's file to this node, e.g. "x-b180".
    line: String,
    player: usize,
    actions: Vec<String>,
    /// Pot at this node in chips, including bets not yet called.
    pot: i32,
    /// Byte offset of this node's data in the body.
    offset: usize,
}

#[derive(Serialize)]
struct FileHeader {
    version: u32,
    loss_step_pct: f32,
    ev_unit: f32,
    hands: [usize; 2],
    nodes: Vec<NodeHeader>,
}

#[derive(Serialize)]
struct Meta {
    version: u32,
    id: String,
    flop: String,
    pot: i32,
    stack: i32,
    oop_hands: Vec<String>,
    ip_hands: Vec<String>,
    oop_weights: Vec<f32>,
    ip_weights: Vec<f32>,
    /// Flop action lines that end with the turn being dealt.
    turn_lines: Vec<String>,
    turn_cards: Vec<String>,
    exploitability_pct: f32,
    solve_seconds: f32,
    memory_gb: f32,
    solver: &'static str,
}

fn action_label(a: &Action) -> String {
    match a {
        Action::Fold => "f".into(),
        Action::Check => "x".into(),
        Action::Call => "c".into(),
        Action::Bet(n) => format!("b{n}"),
        Action::Raise(n) => format!("r{n}"),
        Action::AllIn(n) => format!("a{n}"),
        Action::Chance(c) => format!("d{}", card_to_string(*c).unwrap_or_default()),
        Action::None => "?".into(),
    }
}

fn build_game(spot: &Spot) -> Result<PostFlopGame, String> {
    let sizes = |s: &Sizes| BetSizeOptions::try_from((s.bet.as_str(), s.raise.as_str()));
    let (f, t, r) = (sizes(&spot.flop_sizes)?, sizes(&spot.turn_sizes)?, sizes(&spot.river_sizes)?);
    let card_config = CardConfig {
        range: [spot.oop_range.parse()?, spot.ip_range.parse()?],
        flop: flop_from_str(&spot.flop)?,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: spot.pot,
        effective_stack: spot.stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [f.clone(), f],
        turn_bet_sizes: [t.clone(), t],
        river_bet_sizes: [r.clone(), r],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };
    PostFlopGame::with_config(card_config, ActionTree::new(tree_config)?)
}

struct Exporter {
    nodes: Vec<NodeHeader>,
    body: Vec<u8>,
    /// Histories (action indices) that reach the next street's chance node, with their labels.
    next_street: Vec<(Vec<usize>, String)>,
    starting_pot: i32,
}

impl Exporter {
    fn new(starting_pot: i32) -> Self {
        Self { nodes: Vec::new(), body: Vec::new(), next_street: Vec::new(), starting_pot }
    }

    /// Records every decision node on the current street, depth first.
    fn walk(&mut self, game: &mut PostFlopGame, history: &mut Vec<usize>, line: &mut Vec<String>) {
        if game.is_terminal_node() {
            return;
        }
        if game.is_chance_node() {
            self.next_street.push((history.clone(), line.join("-")));
            return;
        }

        let player = game.current_player();
        let actions = game.available_actions();
        let hands = game.private_cards(player).len();
        game.cache_normalized_weights();
        let strategy = game.strategy();
        let ev = game.expected_values_detail(player);
        let bets = game.total_bet_amount();
        let pot = self.starting_pot + bets[0] + bets[1];
        let step = pot as f32 * LOSS_STEP_PCT / 100.0;

        self.nodes.push(NodeHeader {
            line: line.join("-"),
            player,
            actions: actions.iter().map(action_label).collect(),
            pot,
            offset: self.body.len(),
        });

        for &s in &strategy {
            self.body.push((s.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        let mut best = vec![f32::MIN; hands];
        for a in 0..actions.len() {
            for h in 0..hands {
                best[h] = best[h].max(ev[a * hands + h]);
            }
        }
        for a in 0..actions.len() {
            for h in 0..hands {
                let loss = (best[h] - ev[a * hands + h]) / step;
                self.body.push(loss.round().clamp(0.0, 255.0) as u8);
            }
        }
        for &b in &best {
            let v = (b / EV_UNIT).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            self.body.extend_from_slice(&v.to_le_bytes());
        }

        for (i, action) in actions.iter().enumerate() {
            history.push(i);
            line.push(action_label(action));
            game.apply_history(history);
            self.walk(game, history, line);
            history.pop();
            line.pop();
        }
        game.apply_history(history);
    }

    fn write(self, path: &Path, hands: [usize; 2]) -> std::io::Result<u64> {
        let header = serde_json::to_vec(&FileHeader {
            version: 1,
            loss_step_pct: LOSS_STEP_PCT,
            ev_unit: EV_UNIT,
            hands,
            nodes: self.nodes,
        })?;
        let mut gz = GzEncoder::new(Vec::new(), Compression::best());
        gz.write_all(&(header.len() as u32).to_le_bytes())?;
        gz.write_all(&header)?;
        gz.write_all(&self.body)?;
        let bytes = gz.finish()?;
        fs::write(path, &bytes)?;
        Ok(bytes.len() as u64)
    }
}

fn solve_spot(spot: &Spot, out_dir: &Path, max_iterations: u32) -> Result<(), String> {
    let dir = out_dir.join(&spot.id);
    if dir.join("meta.json").exists() {
        println!("{}: already solved, skipped", spot.id);
        return Ok(());
    }
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let mut game = build_game(spot)?;
    let (_, compressed) = game.memory_usage();
    let memory_gb = compressed as f32 / 1073741824.0;
    game.allocate_memory(true);

    let started = Instant::now();
    let target = spot.pot as f32 * spot.target_pct / 100.0;
    let exploitability = solve(&mut game, max_iterations, target, false);
    let solve_seconds = started.elapsed().as_secs_f32();

    let hands = [game.private_cards(0).len(), game.private_cards(1).len()];
    let mut total_bytes = 0u64;

    // Flop
    game.back_to_root();
    let mut flop = Exporter::new(spot.pot);
    flop.walk(&mut game, &mut Vec::new(), &mut Vec::new());
    let turn_entries = std::mem::take(&mut flop.next_street);
    total_bytes += flop.write(&dir.join("flop.bin.gz"), hands).map_err(|e| e.to_string())?;

    // Turn: one file per turn card
    let mut turn_cards = Vec::new();
    if let Some((first, _)) = turn_entries.first() {
        game.apply_history(first);
        let possible = game.possible_cards();
        for card in 0..52u8 {
            if possible & (1u64 << card) == 0 {
                continue;
            }
            let name = card_to_string(card)?;
            let mut turn = Exporter::new(spot.pot);
            for (history, label) in &turn_entries {
                let mut h = history.clone();
                h.push(card as usize);
                game.apply_history(&h);
                turn.walk(&mut game, &mut h, &mut vec![label.clone()]);
            }
            total_bytes += turn
                .write(&dir.join(format!("turn-{name}.bin.gz")), hands)
                .map_err(|e| e.to_string())?;
            turn_cards.push(name);
        }
    }

    game.back_to_root();
    let meta = Meta {
        version: 1,
        id: spot.id.clone(),
        flop: spot.flop.clone(),
        pot: spot.pot,
        stack: spot.stack,
        oop_hands: holes_to_strings(game.private_cards(0))?,
        ip_hands: holes_to_strings(game.private_cards(1))?,
        oop_weights: game.weights(0).to_vec(),
        ip_weights: game.weights(1).to_vec(),
        turn_lines: turn_entries.into_iter().map(|(_, l)| l).collect(),
        turn_cards,
        exploitability_pct: 100.0 * exploitability / spot.pot as f32,
        solve_seconds,
        memory_gb,
        solver: "postflop-solver 9d1509f",
    };
    let meta_json = serde_json::to_vec(&meta).map_err(|e| e.to_string())?;
    total_bytes += meta_json.len() as u64;
    // Written last: its presence marks the spot as complete.
    fs::write(dir.join("meta.json"), meta_json).map_err(|e| e.to_string())?;

    println!(
        "{}: {:.1}s solve, {:.1}s total, {:.2}% pot, {:.2}GB RAM, {:.2}MB on disk",
        spot.id,
        solve_seconds,
        started.elapsed().as_secs_f32(),
        meta.exploitability_pct,
        memory_gb,
        total_bytes as f64 / 1048576.0
    );
    Ok(())
}

fn load_spots(path: &str) -> Result<Vec<Spot>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: poker-trainer-solver solve <spots.json> <out_dir> [--max-iterations N]\n       poker-trainer-solver memory <spots.json>";
    let result = match args.get(1).map(String::as_str) {
        Some("solve") if args.len() >= 4 => {
            let max_iterations = args
                .iter()
                .position(|a| a == "--max-iterations")
                .and_then(|i| args.get(i + 1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(500);
            load_spots(&args[2]).and_then(|spots| {
                let out = PathBuf::from(&args[3]);
                let started = Instant::now();
                let mut failed = 0;
                for spot in &spots {
                    if let Err(e) = solve_spot(spot, &out, max_iterations) {
                        eprintln!("{}: FAILED: {e}", spot.id);
                        failed += 1;
                    }
                }
                println!("{} spots, {failed} failed, {:.0}s", spots.len(), started.elapsed().as_secs_f32());
                if failed > 0 { Err(format!("{failed} spots failed")) } else { Ok(()) }
            })
        }
        Some("memory") if args.len() >= 3 => load_spots(&args[2]).and_then(|spots| {
            for spot in &spots {
                let game = build_game(spot)?;
                let (_, compressed) = game.memory_usage();
                println!("{}: {:.2}GB", spot.id, compressed as f64 / 1073741824.0);
            }
            Ok(())
        }),
        _ => Err(usage.to_string()),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
