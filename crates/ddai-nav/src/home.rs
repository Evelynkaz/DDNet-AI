//! `home` (`!home`, `bot.ts:1586-1601`, `2561-2568`): a tile to walk back to when there is nobody to
//! fight; forgotten when the map changes; while it is set the wayblock is not held.

/// Walk home after this many idle ticks (`GO_HOME_AFTER_TICKS`, 4 s).
pub const GO_HOME_AFTER_TICKS: i64 = 4 * 50;

/// `this.home` and the map it was set on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Home {
    pub tx: i32,
    pub ty: i32,
    /// The map name it was set on (`"?"` until the first map is known).
    pub map: String,
}

impl Home {
    pub fn new(tx: i32, ty: i32, map: &str) -> Home {
        Home {
            tx,
            ty,
            map: map.to_string(),
        }
    }

    /// A map change: `Some(true)` when this home belongs to another map and must be forgotten.
    pub fn on_map_change(&mut self, new_map: &str) -> bool {
        if self.map == "?" {
            self.map = new_map.to_string();
            return false;
        }
        new_map != self.map
    }

    /// Should the bot walk home now? Idle for more than [`GO_HOME_AFTER_TICKS`] and more than 2 tiles away.
    pub fn due(&self, tick: i64, idle_since: i64, here: (i32, i32)) -> bool {
        idle_since >= 0
            && tick - idle_since > GO_HOME_AFTER_TICKS
            && ((here.0 - self.tx).abs() > 2 || (here.1 - self.ty).abs() > 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_is_forgotten_on_another_map_and_adopted_by_the_first_one() {
        let mut h = Home::new(10, 20, "Copy Love Box");
        assert!(!h.on_map_change("Copy Love Box"), "a reload of the same map keeps it");
        assert!(h.on_map_change("BlmapChill"), "another map forgets it");
        let mut unknown = Home::new(1, 2, "?");
        assert!(
            !unknown.on_map_change("BlmapChill"),
            "set before any map was known: adopted"
        );
        assert_eq!(unknown.map, "BlmapChill");
        assert!(unknown.on_map_change("Copy Love Box"));
    }

    #[test]
    fn it_is_due_only_after_four_idle_seconds_and_more_than_two_tiles_away() {
        let h = Home::new(50, 50, "m");
        assert!(!h.due(1000, -1, (0, 0)), "not idle at all");
        assert!(!h.due(1000, 900, (0, 0)), "idle only 2 s");
        assert!(!h.due(1000, 800, (0, 0)), "exactly 200 ticks is not more than 200");
        assert!(h.due(1000, 799, (0, 0)), "idle 201 ticks, far away");
        assert!(!h.due(1000, 0, (52, 48)), "within 2 tiles of home");
        assert!(h.due(1000, 0, (53, 50)), "3 tiles away");
    }
}
