//! Task 3.21 (E-036): the second-generation opponent-input predictor, aimed at the **live window** (2 ticks).
//!
//! What changed against `m1` (the 3.15 model, kept as it was in the crate root):
//!
//! * **Horizon 4** and a window one-hot `0..=4`: the live window is 2 ticks (92%), 3 otherwise; ticks past 4 are not asked for.
//! * **More features** ([`feature`]): the hook's age, our own weapon cooldown and where we aim relative to him, the geometry rays around **us** as well as around him.
//! * **Known ticks** ([`feature::KnownTick`]): the server's pre-inputs (3.20) give the opponent's real input for some ticks of the window; the network reads
//!   them as inputs and is trained to predict only the ticks that are not known.
//! * **A fire head** with its own decoding threshold ([`predictor::Decode`]): a swing is rare (about 2% of ticks, 3.5% of 2-tick windows) and the
//!   network is trained with a weighted loss and ranks it well; the live decision uses a threshold tuned on held-out data, not the 0.5 of a symmetric loss.
//! * **Real opponents** ([`corpus`]): live clips (the competitor's bot) join the arena games, with masks for what a snapshot does not show.
//!
//! Live features equal training features: [`predictor::Predictor`] builds its input from a `World<f32>` with the same functions as [`corpus::Corpus`].

pub mod corpus;
pub mod data;
pub mod feature;
pub mod predictor;
pub mod train;
