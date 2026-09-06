use boids_rust::Weights;
use boids_rust::sim::{BoidConfig, BoundaryMode, SimConfig, Simulation, spawn_boids};
use macroquad::prelude::*;

#[macroquad::main("Boids")]
async fn main() {
    let configuration = SimConfig {
        perception_radius: 50.0,
        separation_radius: 20.0,
        weights: Weights {
            separation: 100.0,
            alignment: 3.0,
            cohesion: 2.0,
        },
        max_speed: 200.0,
        max_force_scale: 7.0,
        width: screen_width(),
        height: screen_height(),
        boundary_mode: BoundaryMode::Bounce,
    };

    let boid_config = BoidConfig {
        count: 300,
        initial_speed: 50.0,
        pos_x_range: (screen_width() * 0.1, screen_width() * 0.9),
        pos_y_range: (screen_height() * 0.1, screen_height() * 0.9),
    };

    let mut sim = Simulation::new(spawn_boids(boid_config), configuration);

    loop {
        clear_background(BLACK);

        for boid in &sim.boids {
            draw_circle(boid.position.x, boid.position.y, 3.0, WHITE);
        }

        sim.config.width = screen_width();
        sim.config.height = screen_height();
        sim.step(get_frame_time());

        next_frame().await
    }
}
