//! Boids algorithm — pure math layer.

// ---------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn zero() -> Self {
        Vec2 { x: 0.0, y: 0.0 }
    }

    /// Vector addition.
    pub fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Vector subtraction (self - other).
    pub fn sub(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    /// Scalar multiplication.
    pub fn scale(self, s: f32) -> Vec2 {
        Vec2 {
            x: self.x * s,
            y: self.y * s,
        }
    }

    /// Squared length — prefer this over `length()` when you only need
    /// to compare or weight by distance, to avoid an unnecessary sqrt.
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length.
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Distance between two points.
    pub fn distance(self, other: Vec2) -> f32 {
        self.sub(other).length()
    }

    /// Squared distance between two points.
    pub fn distance_squared(self, other: Vec2) -> f32 {
        self.sub(other).length_squared()
    }

    /// Unit vector in the same direction. Decide what happens for the
    /// zero vector (return zero? panic? something else?) — and test it.
    pub fn normalize(self) -> Vec2 {
        let len = self.length();
        if len == 0.0 {
            Vec2::zero()
        } else {
            self.scale(1.0 / len)
        }
    }

    /// Clamp the vector's length to `max`, preserving direction.
    /// Vectors already at or under `max` are returned unchanged.
    pub fn clamp_length(self, max: f32) -> Vec2 {
        if self.length() > max {
            self.normalize().scale(max)
        } else {
            self
        }
    }
}

// ---------------------------------------------------------------------
// Boid
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boid {
    pub position: Vec2,
    pub velocity: Vec2,
}

impl Boid {
    pub fn new(position: Vec2, velocity: Vec2) -> Self {
        Boid { position, velocity }
    }
}

// ---------------------------------------------------------------------
// Neighbor finding
// ---------------------------------------------------------------------

/// Returns references to every boid in `boids` within `radius` of
/// `boid`, excluding `boid` itself.
pub fn find_neighbors<'a>(boid: &Boid, boids: &'a [Boid], radius: f32) -> Vec<&'a Boid> {
    boids
        .iter()
        .filter(|&other| {
            let distance = boid.position.distance(other.position);
            distance < radius && boid.position != other.position
        })
        .collect()
}

// ---------------------------------------------------------------------
// The three rules
// ---------------------------------------------------------------------

/// Steer away from nearby neighbors, weighted more strongly the closer
/// they are. Returns the zero vector if `neighbors` is empty.
pub fn separation(boid: &Boid, neighbors: &[&Boid]) -> Vec2 {
    if neighbors.is_empty() {
        Vec2::zero()
    } else {
        let mut steer = Vec2::zero();
        for &neighbor in neighbors {
            let diff = boid.position.sub(neighbor.position);
            let distance = diff.length_squared();
            if distance > 0.0 {
                steer = steer.add(diff.scale(1.0 / distance));
            }
        }
        steer
    }
}

/// Steer towards the average heading of `neighbors`.
/// Returns the zero vector if `neighbors` is empty.
pub fn alignment(boid: &Boid, neighbors: &[&Boid]) -> Vec2 {
    if neighbors.is_empty() {
        Vec2::zero()
    } else {
        let mut steer = Vec2::zero();
        for &neighbor in neighbors {
            steer = steer.add(neighbor.velocity);
        }
        steer = steer.scale(1.0 / neighbors.len() as f32);
        steer.sub(boid.velocity)
    }
}

/// Steer towards the centroid of `neighbors`.
/// Returns the zero vector if `neighbors` is empty.
pub fn cohesion(boid: &Boid, neighbors: &[&Boid]) -> Vec2 {
    if neighbors.is_empty() {
        Vec2::zero()
    } else {
        let mut steer = Vec2::zero();
        for &neighbor in neighbors {
            steer = steer.add(neighbor.position);
        }
        steer = steer.scale(1.0 / neighbors.len() as f32);
        steer = steer.sub(boid.position);
        steer.sub(boid.velocity)
    }
}

// ---------------------------------------------------------------------
// Combining + integrating
// ---------------------------------------------------------------------

/// Weights for combining the three rules into one acceleration vector.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub separation: f32,
    pub alignment: f32,
    pub cohesion: f32,
}

/// Combine the three rule outputs into a single acceleration vector.
pub fn combine(separation: Vec2, alignment: Vec2, cohesion: Vec2, weights: Weights) -> Vec2 {
    separation
        .scale(weights.separation)
        .add(alignment.scale(weights.alignment))
        .add(cohesion.scale(weights.cohesion))
}

/// Apply `acceleration` to `boid` over timestep `dt`, clamping the
/// resulting velocity to `max_speed`. Returns (new_velocity, new_position).
pub fn steer(boid: &Boid, acceleration: Vec2, dt: f32, max_speed: f32) -> (Vec2, Vec2) {
    let new_velocity = boid
        .velocity
        .add(acceleration.scale(dt))
        .clamp_length(max_speed);
    let new_position = boid.position.add(new_velocity.scale(dt));
    (new_velocity, new_position)
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Vec2 basics ---

    #[test]
    fn add_combines_components() {
        assert_eq!(
            Vec2::new(1.0, 2.0).add(Vec2::new(3.0, 4.0)),
            Vec2::new(4.0, 6.0)
        );
    }

    #[test]
    fn sub_combines_components() {
        assert_eq!(
            Vec2::new(5.0, 7.0).sub(Vec2::new(3.0, 4.0)),
            Vec2::new(2.0, 3.0)
        );
    }

    #[test]
    fn scale_multiplies_components() {
        assert_eq!(Vec2::new(1.0, 2.0).scale(3.0), Vec2::new(3.0, 6.0));
    }

    #[test]
    fn length_squared_matches_length() {
        assert_eq!(Vec2::new(3.0, 4.0).length_squared(), 25.0);
    }

    #[test]
    fn normalize_zero_vector_behaves_as_decided() {
        assert_eq!(Vec2::zero().normalize(), Vec2::zero());
    }

    #[test]
    fn clamp_length_under_max_is_unchanged() {
        assert_eq!(Vec2::new(3.0, 4.0).clamp_length(5.0), Vec2::new(3.0, 4.0));
    }

    #[test]
    fn clamp_length_over_max_is_scaled_down() {
        assert_eq!(Vec2::new(3.0, 4.0).clamp_length(2.5), Vec2::new(1.5, 2.0));
    }

    // --- find_neighbors ---

    #[test]
    fn find_neighbors_excludes_self() {
        let radius = 1.0;
        let radius_sqrt: f32 = (radius as f32).sqrt();
        let boids = vec![
            Boid::new(Vec2::new(0.0, 0.0), Vec2::zero()),
            Boid::new(
                Vec2::new(radius_sqrt / 2.0, radius_sqrt / 2.0),
                Vec2::zero(),
            ),
            Boid::new(Vec2::new(radius_sqrt, radius_sqrt), Vec2::zero()),
        ];
        let boid = &boids[0];
        let neighbors = find_neighbors(boid, &boids, radius);
        assert!(!neighbors.contains(&boid));
    }

    #[test]
    fn find_neighbors_excludes_boids_outside_radius() {
        let radius = 1.0;
        let radius_sqrt: f32 = (radius as f32).sqrt();

        let boid_outside = Boid::new(
            Vec2::new(radius_sqrt * 2.0, radius_sqrt * 2.0),
            Vec2::zero(),
        );
        let boids = vec![
            Boid::new(Vec2::new(0.0, 0.0), Vec2::zero()),
            Boid::new(
                Vec2::new(radius_sqrt / 2.0, radius_sqrt / 2.0),
                Vec2::zero(),
            ),
            Boid::new(Vec2::new(radius_sqrt, radius_sqrt), Vec2::zero()),
            boid_outside,
        ];
        let boid = &boids[0];
        let neighbors = find_neighbors(boid, &boids, radius);
        assert!(!neighbors.contains(&&boid_outside));
    }

    // --- separation ---

    #[test]
    fn separation_with_no_neighbors_is_zero() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors: Vec<&Boid> = vec![];
        assert_eq!(separation(&boid, &neighbors), Vec2::zero());
    }

    #[test]
    fn separation_pushes_away_from_a_single_close_neighbor() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors = [&Boid::new(Vec2::new(1.0, 0.0), Vec2::zero())];
        assert_eq!(separation(&boid, &neighbors), Vec2::new(-1.0, 0.0));
    }

    // --- alignment ---

    #[test]
    fn alignment_with_no_neighbors_is_zero() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors: Vec<&Boid> = vec![];
        assert_eq!(alignment(&boid, &neighbors), Vec2::zero());
    }

    #[test]
    fn alignment_matches_identical_neighbor_velocities() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors = [&Boid::new(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0))];
        assert_eq!(alignment(&boid, &neighbors), Vec2::new(1.0, 0.0));
    }

    // --- cohesion ---

    #[test]
    fn cohesion_with_no_neighbors_is_zero() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors: Vec<&Boid> = vec![];
        assert_eq!(cohesion(&boid, &neighbors), Vec2::zero());
    }

    #[test]
    fn cohesion_toward_symmetric_ring_is_near_zero() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors = [
            &Boid::new(Vec2::new(1.0, 0.0), Vec2::zero()),
            &Boid::new(Vec2::new(-1.0, 0.0), Vec2::zero()),
            &Boid::new(Vec2::new(0.0, 1.0), Vec2::zero()),
            &Boid::new(Vec2::new(0.0, -1.0), Vec2::zero()),
        ];
        let result = cohesion(&boid, &neighbors);
        assert!(result.x.abs() < 1e-6 && result.y.abs() < 1e-6);
    }

    // --- steer ---
    #[test]
    fn test_steer_loop() {
        let boid = Boid::new(Vec2::new(0.0, 0.0), Vec2::zero());
        let neighbors = [
            &Boid::new(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0)),
            &Boid::new(Vec2::new(-1.0, 0.0), Vec2::new(-1.0, 0.0)),
            &Boid::new(Vec2::new(0.0, 1.0), Vec2::new(0.0, 1.0)),
        ];

        let dt = 0.1;
        let max_speed = 1.0;

        let separation = separation(&boid, &neighbors);
        let alignment = alignment(&boid, &neighbors);
        let cohesion = cohesion(&boid, &neighbors);

        let acceleration = combine(
            separation,
            alignment,
            cohesion,
            Weights {
                separation: 1.0,
                alignment: 1.0,
                cohesion: 1.0,
            },
        );
        let (new_velocity, new_position) = steer(&boid, acceleration, dt, max_speed);

        // Check that the new velocity is clamped to max_speed
        assert!(new_velocity.length() > 0.0);
        assert!(new_velocity.length() <= max_speed);

        // Check that the new position is updated correctly
        assert_eq!(new_position, boid.position.add(new_velocity.scale(dt)));
    }
}
