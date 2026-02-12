//! Syzygy endgame tablebase probing for KiyEngine.
//!
//! Uses shakmaty-syzygy to probe WDL (Win/Draw/Loss) and DTZ (Distance To Zeroing)
//! tables when the board has ≤ max_pieces pieces.
//!
//! Conversion: chess::Board → FEN string → shakmaty::Chess (only at low piece counts).

use chess::Board;
use shakmaty::fen::Fen;
use shakmaty::CastlingMode;
use shakmaty_syzygy::{Dtz, Tablebase, Wdl};
use std::path::Path;

/// Wrapper around shakmaty-syzygy Tablebase with conversion from chess crate types.
pub struct SyzygyTB {
    tb: Tablebase<shakmaty::Chess>,
    max_pieces: usize,
}

/// Result of a WDL probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbWdl {
    Win,
    CursedWin, // Win but blocked by 50-move rule
    Draw,
    BlessedLoss, // Loss but saved by 50-move rule
    Loss,
}

/// Result of a DTZ probe.
#[derive(Debug, Clone, Copy)]
pub struct TbResult {
    pub wdl: TbWdl,
    pub dtz: i32,
}

impl TbWdl {
    /// Rank for comparison: higher is better for the side to move.
    #[inline]
    pub fn rank(self) -> i32 {
        match self {
            TbWdl::Win => 4,
            TbWdl::CursedWin => 3,
            TbWdl::Draw => 2,
            TbWdl::BlessedLoss => 1,
            TbWdl::Loss => 0,
        }
    }

    /// Negate WDL (swap perspective: Win↔Loss, CursedWin↔BlessedLoss).
    #[inline]
    pub fn negate(self) -> TbWdl {
        match self {
            TbWdl::Win => TbWdl::Loss,
            TbWdl::CursedWin => TbWdl::BlessedLoss,
            TbWdl::Draw => TbWdl::Draw,
            TbWdl::BlessedLoss => TbWdl::CursedWin,
            TbWdl::Loss => TbWdl::Win,
        }
    }
}

impl SyzygyTB {
    /// Create a new Syzygy tablebase instance and load tables from a directory.
    pub fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut tb = Tablebase::new();
        let count = tb.add_directory(path.as_ref())?;
        let max_pieces = tb.max_pieces();
        println!(
            "  Syzygy TB loaded: {} tables, max {} pieces",
            count, max_pieces
        );
        Ok(Self { tb, max_pieces })
    }

    /// Maximum number of pieces supported by loaded tables.
    #[inline]
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    /// Check if a position can be probed (piece count within table limits).
    #[inline]
    pub fn can_probe(&self, board: &Board) -> bool {
        let piece_count = board.combined().0.count_ones() as usize;
        piece_count <= self.max_pieces
    }

    /// Convert chess::Board to shakmaty::Chess via FEN.
    fn to_shakmaty(board: &Board) -> Option<shakmaty::Chess> {
        let fen_str = format!("{}", board);
        let fen: Fen = fen_str.parse().ok()?;
        fen.into_position(CastlingMode::Standard).ok()
    }

    /// Probe WDL (Win/Draw/Loss) for the side to move.
    /// Returns None if position cannot be probed.
    pub fn probe_wdl(&self, board: &Board) -> Option<TbWdl> {
        if !self.can_probe(board) {
            return None;
        }
        let pos = Self::to_shakmaty(board)?;
        match self.tb.probe_wdl_after_zeroing(&pos) {
            Ok(wdl) => Some(match wdl {
                Wdl::Win => TbWdl::Win,
                Wdl::CursedWin => TbWdl::CursedWin,
                Wdl::Draw => TbWdl::Draw,
                Wdl::BlessedLoss => TbWdl::BlessedLoss,
                Wdl::Loss => TbWdl::Loss,
            }),
            Err(_) => None,
        }
    }

    /// Probe DTZ (Distance To Zeroing move) for the side to move.
    /// Returns None if position cannot be probed.
    pub fn probe_dtz(&self, board: &Board) -> Option<TbResult> {
        if !self.can_probe(board) {
            return None;
        }
        let pos = Self::to_shakmaty(board)?;

        let wdl = match self.tb.probe_wdl_after_zeroing(&pos) {
            Ok(w) => match w {
                Wdl::Win => TbWdl::Win,
                Wdl::CursedWin => TbWdl::CursedWin,
                Wdl::Draw => TbWdl::Draw,
                Wdl::BlessedLoss => TbWdl::BlessedLoss,
                Wdl::Loss => TbWdl::Loss,
            },
            Err(_) => return None,
        };

        let dtz = match self.tb.probe_dtz(&pos) {
            Ok(maybe_dtz) => {
                let d: Dtz = maybe_dtz.ignore_rounding();
                d.0
            }
            Err(_) => 0,
        };

        Some(TbResult { wdl, dtz })
    }
}

/// Convert TbWdl to a centipawn score for the search.
/// Uses large values that are distinguishable from mate scores.
pub fn wdl_to_score(wdl: TbWdl, ply: usize) -> i32 {
    match wdl {
        TbWdl::Win => 20000 - ply as i32, // Large win, prefer shorter path
        TbWdl::CursedWin => 50,           // Slight advantage (can't force win under 50-move)
        TbWdl::Draw => 0,
        TbWdl::BlessedLoss => -50, // Slight disadvantage (can't lose under 50-move)
        TbWdl::Loss => -20000 + ply as i32, // Large loss, prefer longer path
    }
}
