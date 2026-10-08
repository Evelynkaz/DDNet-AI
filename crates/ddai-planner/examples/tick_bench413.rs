//! Throwaway (task 4.13): instructions per physics tick of a 2-tee `PhysicsWorld` on Copy Love Box. Not committed.
use std::path::PathBuf;
use std::sync::Arc;

use ddai_planner::physics_adapter::PhysicsWorld;
use ddai_planner::plan_world::PlanWorld;
use ddai_planner::types::empty_input;
use ddai_planner::vmath::Vec2;

fn main() {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join("aiddnet/data/maps/copy-love-box");
    let f = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "map"))
        .unwrap();
    let m = ddai_map::load_map(&std::fs::read(f).unwrap()).unwrap();
    let map = Arc::new(m.data);
    let tees: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let mut pw = PhysicsWorld::new(map, 1);
    for i in 0..tees {
        pw.add_tee(
            i as i32,
            Vec2 {
                x: (103.5 - 2.4 * i as f64) * 32.0,
                y: 84.5 * 32.0,
            },
        );
    }
    let saved = pw.save_state();
    let mut rng = 12345u64;
    let mut sink = 0.0;
    for rep in 0..20000 {
        pw.restore_state(&saved);
        for t in 0..30 {
            for i in 0..tees {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let mut inp = empty_input();
                inp.direction = ((rng >> 33) % 3) as i32 - 1;
                inp.jump = ((rng >> 40) % 8 == 0) as i32;
                inp.hook = ((rng >> 44) % 6 == 0) as i32;
                inp.target_x = ((rng >> 20) % 200) as f64 - 100.0;
                inp.target_y = ((rng >> 28) % 200) as f64 - 100.0;
                pw.set_input(i as i32, inp);
            }
            pw.step();
            if let Some(t) = pw.get_tee(0) {
                sink += t.pos.x;
            }
            let _ = (rep, t);
        }
    }
    println!("{sink}");
}
