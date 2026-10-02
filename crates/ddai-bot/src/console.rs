//! The console (task 4.3): lines from stdin become [`crate::command::BotCommand`]s on the command bus, and
//! the answers are printed. **A line that is not a command (no `!` / `?` in front) is answered right here
//! and goes nowhere** — the TS bot said it in the game chat; this one cannot (`command::parse_line`,
//! `tests/commands.rs`). The same bus is what the web control will use.
//!
//! Replies may hold nicknames the operator typed or the roster has: they go to the operator's terminal
//! only, never to the log (`tracing`).

use std::io::{BufRead, IsTerminal};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::command::{BusError, CommandSender};

/// How long the console waits for the bot to answer a command (the bot answers between two snapshots).
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);

/// Where the console's text goes (the runner prints the navigation's answers through it too).
pub type Printer = Arc<dyn Fn(&str) + Send + Sync>;

/// Prints to stdout.
pub fn stdout_printer() -> Printer {
    Arc::new(|text: &str| println!("{text}"))
}

/// Whether stdin is a terminal (`--console` is on by default then).
pub fn stdin_is_terminal() -> bool {
    std::io::stdin().is_terminal()
}

/// Reads `input` line by line on a thread until it ends, the bot is gone or `!quit` is answered.
pub fn spawn<R: BufRead + Send + 'static>(
    sender: CommandSender,
    input: R,
    print: Printer,
) -> std::io::Result<thread::JoinHandle<()>> {
    thread::Builder::new().name("ddai-console".into()).spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else { break };
            match sender.send_line(&line, ANSWER_TIMEOUT) {
                Ok(reply) => {
                    if !reply.text.is_empty() {
                        print(&reply.text);
                    }
                    if reply.quit {
                        break;
                    }
                }
                Err(BusError::Gone) => {
                    print("the bot has stopped");
                    break;
                }
                Err(BusError::Timeout) => print("no answer yet: the bot is busy; try again"),
            }
        }
    })
}
