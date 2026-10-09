//! Facts about the machine the bot runs on.

/// The 1-, 5- and 15-minute load averages, where the OS keeps them (Linux: `/proc/loadavg`). `None` elsewhere, e.g. on Windows,
/// which has no load average: callers skip the load-dependent advice there.
#[must_use]
pub fn load_average() -> Option<[f64; 3]> {
    #[cfg(target_os = "linux")]
    {
        parse_loadavg(&std::fs::read_to_string("/proc/loadavg").ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The first three fields of `/proc/loadavg` exactly as the kernel prints them (`"0.52 1.25 3.50"`), where the OS has them.
#[must_use]
pub fn load_average_text() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/loadavg").ok()?;
        let fields: Vec<&str> = text.split_whitespace().take(3).collect();
        (fields.len() == 3).then(|| fields.join(" "))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The first three fields of `/proc/loadavg`'s text.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_loadavg(text: &str) -> Option<[f64; 3]> {
    let mut it = text.split_whitespace();
    let one = it.next()?.parse().ok()?;
    let five = it.next()?.parse().ok()?;
    let fifteen = it.next()?.parse().ok()?;
    Some([one, five, fifteen])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loadavg_text_is_parsed() {
        assert_eq!(parse_loadavg("0.52 1.25 3.5 2/1234 56789\n"), Some([0.52, 1.25, 3.5]));
        assert_eq!(parse_loadavg("garbage"), None);
        assert_eq!(parse_loadavg("1 2"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_has_a_load_average() {
        assert!(load_average().is_some());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn other_systems_skip_it() {
        assert_eq!(load_average(), None);
    }
}
