//! Choosing `threads` for the batched trainer from the cores that are actually free (task 7.2c).
//!
//! The engine ends every substep with a rendezvous of all its threads, so on a shared host the
//! right thread count is **the number of cores the other processes leave idle**, not the number
//! of CPUs (task 7.2b: on a saturated 8-vCPU machine 8 threads were no faster than 1). This
//! module measures it the only portable-enough way without a privileged tool: two samples of the
//! aggregate counters of `/proc/stat` a short window apart, `free = host CPUs * idle share`. The
//! counters and the CPU count both come from `/proc/stat`, i.e. they describe the *host* (the
//! guest, in a VM) as a whole; the CPUs this process may actually use (`cpu_count`: affinity,
//! cgroup quota) only cap the advice, they are never mixed into the idle share. Linux only
//! (`None` elsewhere); the caller falls back to a fixed count.
//!
//! [`advise`] turns the number into a recommendation, `ddnet-ai fly cores` prints it.

use std::time::Duration;

/// Aggregate CPU time counters (ticks) of one `/proc/stat` sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSample {
    /// `idle + iowait`.
    pub idle: u64,
    /// `user + nice + system + idle + iowait + irq + softirq + steal` (the guest columns are
    /// already inside `user`/`nice`).
    pub total: u64,
}

/// Parses the aggregate `cpu ` line of `/proc/stat`. `None` when it is missing or malformed.
pub fn parse_proc_stat(text: &str) -> Option<CpuSample> {
    let line = text.lines().find(|l| l.starts_with("cpu "))?;
    let f: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .take(8)
        .map(|x| x.parse().ok())
        .collect::<Option<_>>()?;
    if f.len() < 5 {
        return None;
    }
    // user nice system idle iowait irq softirq steal (older kernels give fewer columns).
    Some(CpuSample {
        idle: f[3] + f[4],
        total: f.iter().sum(),
    })
}

/// Number of CPUs `/proc/stat` lists (its `cpuN` lines).
pub fn host_cpu_count(text: &str) -> usize {
    text.lines()
        .filter(|l| {
            l.strip_prefix("cpu")
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
        })
        .count()
}

/// Cores' worth of idle time between two samples (`cpus * Δidle / Δtotal`); `None` if no time
/// has passed or a counter went backwards.
pub fn free_cores_between(a: &CpuSample, b: &CpuSample, cpus: usize) -> Option<f64> {
    let total = b.total.checked_sub(a.total).filter(|&t| t > 0)?;
    let idle = b.idle.checked_sub(a.idle)?;
    Some(cpus as f64 * (idle as f64 / total as f64).min(1.0))
}

/// Number of CPUs this process may use (affinity / cgroup quota); not the host's CPU count.
pub fn cpu_count() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Samples `/proc/stat` twice, `window` apart, and returns the host's idle cores in between
/// (host CPUs from the same file times the idle share). Run it *before* the training starts (the
/// trainer's own threads would count as busy).
pub fn measure_free_cores(window: Duration) -> Option<f64> {
    let read = || -> Option<(CpuSample, usize)> {
        let text = std::fs::read_to_string("/proc/stat").ok()?;
        Some((parse_proc_stat(&text)?, host_cpu_count(&text)))
    };
    let (a, host_cpus) = read()?;
    std::thread::sleep(window);
    free_cores_between(&a, &read()?.0, host_cpus.max(1))
}

/// A recommendation for the batched trainer's parallelism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreAdvice {
    pub cpus: usize,
    pub free_cores: f64,
    /// `train.threads` / `--threads`.
    pub threads: usize,
    /// `[fly] batched_subengines` (`1` = one engine over all threads).
    pub subengines: usize,
}

/// Free cores below which the host counts as loaded: a single engine's per-substep rendezvous
/// (`threads` workers need `threads` cores at the same moment) stalls on the slowest thread, and
/// sub-batch engines of two threads each do better (README, "Задача 7.2c", E-013).
pub const LOADED_BELOW_FREE_CORES: f64 = 6.0;

/// The recommendation, from the measurements of task 7.2c on the 8-vCPU shared host:
/// - **quiet** (at least [`LOADED_BELOW_FREE_CORES`] free cores): one engine, `threads` = the
///   free cores rounded to the nearest whole core (at most `max_threads` and the CPU count);
/// - **loaded**: `threads` = the free cores rounded *up* to an even number (2 at least, within
///   the same limits), cut into `threads / 2` sub-batch engines of two threads each
///   ([`BatchedEngine::with_subengines`](super::BatchedEngine::with_subengines)): at load 16-29
///   (1-3 free cores) M, B = 64 trained 890 decisions/s that way against 660 for one engine on
///   3 threads and 600 on 8, for about the same CPU time per step as the single engine; more
///   groups (one thread each) are a little faster still but cost twice the CPU.
pub fn advise(free_cores: f64, cpus: usize, max_threads: usize) -> CoreAdvice {
    let cap = cpus.min(max_threads).max(1);
    let free = if free_cores.is_finite() {
        free_cores.max(0.0)
    } else {
        0.0
    };
    if free >= LOADED_BELOW_FREE_CORES {
        return CoreAdvice {
            cpus,
            free_cores,
            threads: (free.round() as usize).clamp(1, cap),
            subengines: 1,
        };
    }
    let even = (free.ceil() as usize).max(2).next_multiple_of(2);
    let threads = if cap >= 2 { even.min(cap & !1) } else { 1 };
    CoreAdvice {
        cpus,
        free_cores,
        threads,
        subengines: (threads / 2).max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAT_A: &str = "cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 10 0 5 80 5 0 0 0 0 0\nintr 1\n";
    const STAT_B: &str = "cpu  160 0 70 860 50 0 0 0 0 0\ncpu0 10 0 5 80 5 0 0 0 0 0\nintr 2\n";

    #[test]
    fn parses_the_aggregate_line_and_ignores_per_cpu_lines() {
        let s = parse_proc_stat(STAT_A).unwrap();
        assert_eq!((s.idle, s.total), (850, 1000));
        assert!(parse_proc_stat("cpu0 1 2 3 4 5\n").is_none());
        assert!(parse_proc_stat("cpu  1 2 x 4 5\n").is_none());
        assert!(parse_proc_stat("").is_none());
        // older kernels: only five columns
        assert_eq!(
            parse_proc_stat("cpu 1 0 1 7 1\n"),
            Some(CpuSample { idle: 8, total: 10 })
        );
    }

    #[test]
    fn host_cpus_are_the_cpu_n_lines_not_the_aggregate_or_other_keys() {
        assert_eq!(host_cpu_count(STAT_A), 1);
        assert_eq!(
            host_cpu_count("cpu  1 2 3\ncpu0 1\ncpu1 1\ncpu12 1\ncpufreq 3\nintr 1\n"),
            3
        );
        assert_eq!(host_cpu_count(""), 0);
    }

    #[test]
    fn free_cores_are_the_idle_share_of_the_window_times_the_cpus() {
        let (a, b) = (parse_proc_stat(STAT_A).unwrap(), parse_proc_stat(STAT_B).unwrap());
        // Δtotal 140, Δidle 60 -> 60/140 of 8 cpus
        let f = free_cores_between(&a, &b, 8).unwrap();
        assert!((f - 8.0 * 60.0 / 140.0).abs() < 1e-12, "{f}");
        assert!(free_cores_between(&b, &a, 8).is_none(), "counters went backwards");
        assert!(free_cores_between(&a, &a, 8).is_none(), "no time passed");
    }

    #[test]
    fn advice_follows_the_free_cores() {
        // quiet: one engine, the free cores (clamped)
        let quiet = advise(7.9, 8, 6);
        assert_eq!((quiet.threads, quiet.subengines), (6, 1));
        assert_eq!(advise(6.4, 8, 8).threads, 6);
        assert_eq!(advise(7.9, 4, 6).threads, 4);
        // loaded: even thread count (rounded up), two threads per sub-engine
        for (free, threads, k) in [
            (0.2, 2, 1),
            (1.0, 2, 1),
            (2.3, 4, 2),
            (3.0, 4, 2),
            (3.1, 4, 2),
            (5.2, 6, 3),
        ] {
            let a = advise(free, 8, 6);
            assert_eq!((a.threads, a.subengines), (threads, k), "free {free}");
        }
        // limits and junk
        assert_eq!((advise(5.2, 8, 3).threads, advise(5.2, 8, 3).subengines), (2, 1));
        assert_eq!((advise(3.0, 1, 6).threads, advise(3.0, 1, 6).subengines), (1, 1));
        let nan = advise(f64::NAN, 8, 6);
        assert_eq!((nan.threads, nan.subengines), (2, 1));
    }

    #[test]
    fn measuring_the_real_host_gives_a_sane_number_or_nothing() {
        if let Some(f) = measure_free_cores(Duration::from_millis(50)) {
            let host = host_cpu_count(&std::fs::read_to_string("/proc/stat").unwrap());
            assert!((0.0..=host as f64 + 1e-9).contains(&f), "{f} of {host}");
        }
    }
}
