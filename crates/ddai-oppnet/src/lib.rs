//! Task 3.15 (E-028): a learned predictor of the opponent's inputs for the hybrid's input-lag window.
//!
//! * [`frame`]: what a live snapshot shows of a tee, the applied-input record, the geometry rays.
//! * [`feature`]: the input vector (history of frames, rays, our in-flight inputs, the window length) and the targets (direction, jump,
//!   hook, fire press, aim change for the next [`feature::HORIZON`] ticks).
//! * [`net`]: the MLP with its hand-written backward and a fixed summation order.
//! * [`data`], [`blob`]: the dataset format and its file container (shared with the model bundle, [`bundle`]).
//! * [`train`]: samples, loss, the trainer and the offline metrics against "hold the last input".
//! * [`predictor`]: the network behind `ddai_planner::hybrid::window::WindowModel`.
//! * [`live`] (task 3.17): the predictor in the live bot -- history from every snapshot, the online guard against hold, the compact log and its analysis.

pub mod blob;
pub mod bundle;
pub mod data;
pub mod feature;
pub mod frame;
pub mod live;
pub mod net;
pub mod predictor;
pub mod train;

pub use bundle::OppBundle;
pub use predictor::OppPredictor;
