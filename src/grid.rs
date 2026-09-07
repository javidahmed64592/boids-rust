//! Boids algorithm — spatial partitioning.
//!
//! A uniform grid that buckets boid indices by position each frame, so
//! neighbor search only has to check a handful of nearby cells instead of
//! every boid in the flock. Correctness depends on `cell_size >= radius`
//! for any radius you intend to query with — see `query_candidates`.

use crate::{Boid, Vec2};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellCoord {
    x: i32,
    y: i32,
}

pub struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<CellCoord, Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        SpatialGrid {
            cell_size,
            cells: HashMap::new(),
        }
    }

    /// Clear and rebuild the grid from scratch against the current boid
    /// positions. Call this once per `step()`, before any neighbor
    /// queries for that frame — the grid only reflects whatever
    /// positions were present the last time this was called.
    pub fn rebuild(&mut self, boids: &[Boid]) {
        self.cells.clear();

        for (i, boid) in boids.iter().enumerate() {
            let cell = self.cell_coord(boid.position);
            self.cells.entry(cell).or_default().push(i);
        }
    }

    /// Map a world position to the cell coordinate it falls in.
    fn cell_coord(&self, position: Vec2) -> CellCoord {
        CellCoord {
            x: (position.x / self.cell_size).floor() as i32,
            y: (position.y / self.cell_size).floor() as i32,
        }
    }

    /// Return the indices (into the `boids` slice passed to `rebuild`) of
    /// every boid sharing `boid`'s cell or one of its 8 neighboring cells.
    /// This is a *candidate* set, not a final answer: cell-adjacency
    /// only guarantees candidates are within `radius` when
    /// `cell_size >= radius` (see module docs) — callers still need to
    /// filter by actual distance, and still need to exclude `self_index`.
    pub fn query_candidates(&self, boid: &Boid, self_index: usize) -> Vec<usize> {
        let cell = self.cell_coord(boid.position);
        let mut candidates = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                let neighbor_cell = CellCoord {
                    x: cell.x + dx,
                    y: cell.y + dy,
                };
                if let Some(indices) = self.cells.get(&neighbor_cell) {
                    candidates.extend(indices.iter().copied());
                }
            }
        }

        candidates.retain(|&i| i != self_index);
        candidates
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_boids() -> Vec<Boid> {
        vec![
            Boid::new(Vec2::new(0.0, 0.0), Vec2::zero()), // index 0
            Boid::new(Vec2::new(1.0, 0.0), Vec2::zero()), // index 1, same cell as 0
            Boid::new(Vec2::new(25.0, 25.0), Vec2::zero()), // index 2, far away
            Boid::new(Vec2::new(10.0, 0.0), Vec2::zero()), // index 3, adjacent cell to 0
        ]
    }

    #[test]
    fn rebuild_then_query_finds_same_cell_neighbor() {
        // cell_size = 10.0: boids 0 and 1 land in the same cell.
        let mut grid = SpatialGrid::new(10.0);
        let boids = test_boids();
        grid.rebuild(&boids);
        let candidates = grid.query_candidates(&boids[0], 0);
        assert!(candidates.contains(&1));
    }

    #[test]
    fn query_excludes_self_index() {
        let mut grid = SpatialGrid::new(10.0);
        let boids = test_boids();
        grid.rebuild(&boids);
        let candidates = grid.query_candidates(&boids[0], 0);
        assert!(!candidates.contains(&0));
    }

    #[test]
    fn query_includes_adjacent_cell_boid() {
        // boid 3 at x=10.0 should appear as a candidate for boid 0 at
        // x=0.0 when cell_size = 10.0 (adjacent cell, not same cell).
        let mut grid = SpatialGrid::new(10.0);
        let boids = test_boids();
        grid.rebuild(&boids);
        let candidates = grid.query_candidates(&boids[0], 0);
        assert!(candidates.contains(&3));
    }

    #[test]
    fn query_excludes_far_away_boid() {
        // boid 2 is more than one cell away in both axes from boid 0.
        let mut grid = SpatialGrid::new(10.0);
        let boids = test_boids();
        grid.rebuild(&boids);
        let candidates = grid.query_candidates(&boids[0], 0);
        assert!(!candidates.contains(&2));
    }

    #[test]
    fn rebuild_clears_previous_state() {
        // Rebuilding with a different (e.g. smaller) set of boids should
        // not leave stale indices from the previous rebuild behind.
        let mut grid = SpatialGrid::new(10.0);
        let boids = test_boids();
        grid.rebuild(&boids);
        let new_boids = vec![Boid::new(Vec2::new(0.0, 0.0), Vec2::zero())];
        grid.rebuild(&new_boids);
        let candidates = grid.query_candidates(&new_boids[0], 0);
        assert!(candidates.is_empty());
    }
}
