//! Boids algorithm — simulation layer.

use crate::{Boid, Vec2, Weights, alignment, cohesion, combine, find_neighbors, separation};
use rand::{RngExt, rng};

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------

/// How out-of-bounds boids are handled at the edge of the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMode {
    /// Positions wrap around to the opposite edge (torus world).
    Wrap,
    /// The relevant velocity component is flipped on contact with an edge.
    Bounce,
}

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    /// Radius used for alignment and cohesion neighbor search.
    pub perception_radius: f32,
    /// Radius used for separation neighbor search.
    pub separation_radius: f32,
    pub weights: Weights,
    pub max_speed: f32,
    pub max_force_scale: f32,
    /// World size.
    pub width: f32,
    pub height: f32,
    pub boundary_mode: BoundaryMode,
}

#[derive(Debug, Clone, Copy)]
pub struct BoidConfig {
    pub count: usize,
    pub initial_speed: f32,
    pub pos_x_range: (f32, f32),
    pub pos_y_range: (f32, f32),
}

pub fn spawn_boids(config: BoidConfig) -> Vec<Boid> {
    let mut rng = rng();
    (0..config.count)
        .map(|_| {
            let position = Vec2::new(
                rng.random_range(config.pos_x_range.0..config.pos_x_range.1),
                rng.random_range(config.pos_y_range.0..config.pos_y_range.1),
            );
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            let velocity = Vec2::new(angle.cos(), angle.sin()).scale(config.initial_speed);
            Boid::new(position, velocity)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------

pub struct Simulation {
    pub boids: Vec<Boid>,
    pub config: SimConfig,
}

impl Simulation {
    pub fn new(boids: Vec<Boid>, config: SimConfig) -> Self {
        Simulation { boids, config }
    }

    /// Advance the simulation by one timestep.
    ///
    /// Must be two-phase: compute every boid's new (velocity, position)
    /// from the current snapshot of `self.boids` first, then apply all
    /// updates at once. Computing and mutating in the same pass would let
    /// later boids in the loop see already-updated neighbors instead of
    /// the previous frame's state.
    pub fn step(&mut self, dt: f32) {
        let new_boids: Vec<Boid> = self
            .boids
            .iter()
            .map(|boid| {
                let acceleration = self
                    .compute_acceleration(boid)
                    .clamp_length(self.config.max_speed / self.config.max_force_scale);
                let new_velocity =
                    (boid.velocity + acceleration.scale(dt)).clamp_length(self.config.max_speed);
                let new_position = boid.position + new_velocity.scale(dt);
                let (final_velocity, final_position) =
                    self.apply_boundary(new_velocity, new_position);
                Boid {
                    position: final_position,
                    velocity: final_velocity,
                }
            })
            .collect();

        self.boids = new_boids;
    }

    /// Compute the steering acceleration for a single boid against the
    /// rest of the (unmodified) flock, using `self.config`.
    fn compute_acceleration(&self, boid: &Boid) -> Vec2 {
        let broad_radius = self
            .config
            .perception_radius
            .max(self.config.separation_radius);
        let broad_neighbors = find_neighbors(boid, &self.boids, broad_radius);

        let (perception_neighbors, separation_neighbors) =
            if self.config.perception_radius >= self.config.separation_radius {
                let separation_neighbors = find_neighbors(
                    boid,
                    broad_neighbors.iter().copied(),
                    self.config.separation_radius,
                );
                (broad_neighbors, separation_neighbors)
            } else {
                let perception_neighbors = find_neighbors(
                    boid,
                    broad_neighbors.iter().copied(),
                    self.config.perception_radius,
                );
                (perception_neighbors, broad_neighbors)
            };

        let separation = separation(boid, &separation_neighbors);
        let alignment = alignment(boid, &perception_neighbors, self.config.max_speed);
        let cohesion = cohesion(boid, &perception_neighbors);
        combine(separation, alignment, cohesion, self.config.weights)
    }

    /// Apply `self.config.boundary_mode` to a single (velocity, position)
    /// pair, returning the corrected pair.
    fn apply_boundary(&self, velocity: Vec2, position: Vec2) -> (Vec2, Vec2) {
        match self.config.boundary_mode {
            BoundaryMode::Wrap => {
                let new_position = Vec2 {
                    x: (position.x + self.config.width) % self.config.width,
                    y: (position.y + self.config.height) % self.config.height,
                };
                (velocity, new_position)
            }
            BoundaryMode::Bounce => {
                let mut new_velocity = velocity;
                let mut new_position = position;

                if new_position.x < 0.0 || new_position.x > self.config.width {
                    new_velocity.x = -new_velocity.x;
                    new_position.x = new_position.x.clamp(0.0, self.config.width);
                }
                if new_position.y < 0.0 || new_position.y > self.config.height {
                    new_velocity.y = -new_velocity.y;
                    new_position.y = new_position.y.clamp(0.0, self.config.height);
                }

                (new_velocity, new_position)
            }
        }
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
            Boid {
                position: Vec2 { x: 0.0, y: 0.0 },
                velocity: Vec2 { x: 1.0, y: 0.0 },
            },
            Boid {
                position: Vec2 { x: 1.0, y: 1.0 },
                velocity: Vec2 { x: 0.0, y: 1.0 },
            },
            Boid {
                position: Vec2 { x: 2.0, y: 2.0 },
                velocity: Vec2 { x: -1.0, y: 0.0 },
            },
            Boid {
                position: Vec2 { x: 3.0, y: 3.0 },
                velocity: Vec2 { x: 0.0, y: -1.0 },
            },
        ]
    }

    fn test_config() -> SimConfig {
        SimConfig {
            perception_radius: 5.0,
            separation_radius: 1.0,
            weights: Weights {
                separation: 1.0,
                alignment: 1.0,
                cohesion: 1.0,
            },
            max_speed: 2.0,
            max_force_scale: 5.0,
            width: 100.0,
            height: 100.0,
            boundary_mode: BoundaryMode::Wrap,
        }
    }

    fn test_simulation(boids: Vec<Boid>) -> Simulation {
        Simulation::new(boids, test_config())
    }

    #[test]
    fn step_preserves_boid_count() {
        let mut sim = test_simulation(test_boids());
        let initial_boid_count = sim.boids.len();
        sim.step(1.0);
        assert_eq!(sim.boids.len(), initial_boid_count);
    }

    #[test]
    fn step_never_exceeds_max_speed() {
        let mut sim = test_simulation(test_boids());
        sim.step(1.0);
        for boid in &sim.boids {
            let speed = (boid.velocity.x.powi(2) + boid.velocity.y.powi(2)).sqrt();
            assert!(speed <= sim.config.max_speed);
        }
    }

    #[test]
    fn step_is_deterministic() {
        let mut sim1 = test_simulation(test_boids());
        let mut sim2 = test_simulation(test_boids());
        sim1.step(1.0);
        sim2.step(1.0);
        for (boid1, boid2) in sim1.boids.iter().zip(sim2.boids.iter()) {
            assert_eq!(boid1.position, boid2.position);
            assert_eq!(boid1.velocity, boid2.velocity);
        }
    }

    #[test]
    fn wrap_boundary_wraps_position_past_edge() {
        let mut sim = test_simulation(vec![Boid {
            position: Vec2 { x: 99.0, y: 50.0 },
            velocity: Vec2 { x: 20.0, y: 0.0 },
        }]);
        sim.step(1.0);
        let new_position = sim.boids[0].position;
        assert!(new_position.x < sim.config.width);
    }

    #[test]
    fn bounce_boundary_flips_velocity_component_at_edge() {
        let mut sim = test_simulation(vec![Boid {
            position: Vec2 { x: 0.0, y: 50.0 },
            velocity: Vec2 { x: -2.0, y: 0.0 },
        }]);
        sim.config.boundary_mode = BoundaryMode::Bounce;
        sim.step(1.0);
        let boid = &sim.boids[0];
        assert!(boid.velocity.x > 0.0);
    }
}
