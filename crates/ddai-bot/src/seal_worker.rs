//! The seal search on its own thread. `sealedIn` is up to 4 x 90 physics steps (~0.5-1.5 ms on this
//! VM) and the bot's own overhead budget is 0.5 ms p99 (D-042), so in the live runner the search does
//! not run on the decision path: the bot sends a copy of the snapshot's world to this worker and uses
//! the answer when it arrives, normally at the next snapshot (40 ms later). Until then the tee counts
//! as "not sealed" (or keeps its previous answer) — at worst a hopelessly stuck tee stays targetable for
//! one more snapshot. The sans-IO [`crate::bot::Bot`] used by tests keeps the synchronous search
//! ([`crate::target::TargetPicker`] without a worker), so every scenario is deterministic.
//!
//! One request per client id is in flight at a time; a worker that cannot keep up simply answers
//! later (requests queue on an unbounded channel, but only one per id exists, so at most 128).

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ddai_physics::map::MapData;
use ddai_physics::world::World;

use crate::planning::PlanScratch;

struct Request {
    id: i32,
    tick: i32,
    generation: u64,
    world: Box<World<f32>>,
}

/// One finished search.
#[derive(Debug, Clone, Copy)]
pub struct SealResult {
    pub id: i32,
    /// The tick of the world the search ran on.
    pub tick: i32,
    pub generation: u64,
    pub sealed: bool,
    pub took: Duration,
}

pub struct SealWorker {
    tx: Option<Sender<Request>>,
    rx: Receiver<SealResult>,
    handle: Option<JoinHandle<()>>,
}

impl SealWorker {
    pub fn spawn(map: Arc<MapData>) -> std::io::Result<SealWorker> {
        let (tx, req_rx) = mpsc::channel::<Request>();
        let (res_tx, rx) = mpsc::channel::<SealResult>();
        let handle = std::thread::Builder::new()
            .name("ddai-seal".to_string())
            .stack_size(64 << 20)
            .spawn(move || {
                let mut plan = PlanScratch::new(map);
                while let Ok(req) = req_rx.recv() {
                    let t0 = Instant::now();
                    let sealed = plan.sealed(&req.world, req.id);
                    let res = SealResult {
                        id: req.id,
                        tick: req.tick,
                        generation: req.generation,
                        sealed,
                        took: t0.elapsed(),
                    };
                    if res_tx.send(res).is_err() {
                        return;
                    }
                }
            })?;
        Ok(SealWorker {
            tx: Some(tx),
            rx,
            handle: Some(handle),
        })
    }

    /// Queues a search on a copy of `world` (the one allocation of this path).
    pub fn request(&self, id: i32, tick: i32, generation: u64, world: &World<f32>) -> bool {
        self.tx.as_ref().is_some_and(|tx| {
            tx.send(Request {
                id,
                tick,
                generation,
                world: Box::new(world.clone()),
            })
            .is_ok()
        })
    }

    /// Finished searches, if any.
    pub fn try_recv(&self) -> Option<SealResult> {
        self.rx.try_recv().ok()
    }
}

impl Drop for SealWorker {
    fn drop(&mut self) {
        // Closing the request channel ends the thread after its current search.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_comes_back_with_its_tag_and_a_dropped_worker_joins() {
        let map = Arc::new(crate::mapgrid::test_maps::room(30, 30, &[]));
        let mut world = World::<f32>::from_map(&map, 1);
        let _ = world.init(std::iter::empty::<&str>());
        let worker = SealWorker::spawn(Arc::clone(&map)).unwrap();
        // No such tee in the world: "not sealed", but the round trip and the tags are what is tested.
        assert!(worker.request(7, 1234, 3, &world));
        let mut got = None;
        for _ in 0..200 {
            if let Some(r) = worker.try_recv() {
                got = Some(r);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let r = got.expect("an answer");
        assert_eq!((r.id, r.tick, r.generation, r.sealed), (7, 1234, 3, false));
        drop(worker); // must not hang
    }
}
