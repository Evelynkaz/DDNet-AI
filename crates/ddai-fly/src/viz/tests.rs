use super::*;
use crate::brain::ActionSelection;
use crate::brain::tests::{make_brain, sample_observation};
use ddai_brain::Brain;
use ddai_flyg::NeuronRole;

fn decide_n(brain: &mut crate::brain::FlyBrain, n: usize) {
    for i in 0..n {
        let _ = brain.decide(&sample_observation(300.0 + 10.0 * i as f32));
    }
}

#[test]
fn types_are_grouped_by_role_and_neuropil_prefix() {
    let c = |n: &str, sc: &str, r| classify(n, sc, r);
    assert_eq!(
        c("LC10a", "visual_projection", NeuronRole::InputVisual),
        (Family::Vpn, "LC10a".into())
    );
    for hs in ["HSE", "HSN", "HSS", "VS", "H2"] {
        assert_eq!(
            c(hs, "visual_projection", NeuronRole::InputVisual),
            (Family::Vpn, "HS/VS".into())
        );
    }
    assert_eq!(
        c("AN05B099", "ascending_neuron", NeuronRole::InputAscending),
        (Family::An, "AN".into())
    );
    assert_eq!(
        c("AOTU019", "cb_intrinsic", NeuronRole::Hidden),
        (Family::Central, "AOTU".into())
    );
    assert_eq!(
        c("CB4072", "cb_intrinsic", NeuronRole::Hidden),
        (Family::Central, "CB".into())
    );
    assert_eq!(
        c("DNg05_a", "descending_neuron", NeuronRole::Hidden),
        (Family::Central, "DN скрытые".into())
    );
    assert_eq!(
        c("DNge079", "descending_neuron", NeuronRole::Output),
        (Family::Dn, "DNge".into())
    );
    assert_eq!(
        c("DNa02", "descending_neuron", NeuronRole::Output),
        (Family::Dn, "DNa".into())
    );
    assert_eq!(
        c("MDN", "descending_neuron", NeuronRole::Output),
        (Family::Dn, "MDN".into())
    );
    assert_eq!(
        c("X1", "something_new", NeuronRole::Hidden),
        (Family::Central, "прочие".into())
    );
}

#[test]
fn the_layout_covers_every_type_once_with_weights_summing_to_one() {
    let brain = make_brain(1, ActionSelection::Argmax);
    let layout = brain.viz_layout();
    assert!(!layout.groups().is_empty());
    let flyg = brain.model().flyg();
    // Every type in the graph is in exactly one group.
    let mut seen = vec![0u32; flyg.types.len()];
    for &t in &layout.group_type {
        seen[t as usize] += 1;
    }
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    for g in 0..layout.groups().len() {
        let (a, b) = (layout.group_start[g] as usize, layout.group_start[g + 1] as usize);
        let sum: f32 = layout.group_weight[a..b].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "group {g}: {sum}");
    }
    // The families come in pipeline order.
    let fams: Vec<Family> = layout.groups().iter().map(|g| g.family).collect();
    assert!(fams.windows(2).all(|w| w[0] <= w[1]), "{fams:?}");
    assert_eq!(
        layout.frame_len(),
        HEADER_LEN + layout.groups().len() + layout.num_dn() + 7 * 48 * 4 + 3
    );
}

#[test]
fn the_header_is_the_documented_layout() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(1);
    decide_n(&mut brain, 1);
    let layout = brain.viz_layout().clone();
    let b = brain.viz_frame_with(0x0A0B_0C0D, None).unwrap().to_vec();
    assert_eq!(&b[0..4], b"DFLY");
    assert_eq!(b[4], 1);
    assert_eq!(
        b[5] & (flag::CHOSEN_VALID | flag::CHOSEN),
        0,
        "a fly that plays alone has no chosen flag"
    );
    assert_eq!(&b[8..12], &0x0A0B_0C0Du32.to_le_bytes());
    assert_eq!(&b[12..16], &1u32.to_le_bytes());
    assert_eq!(u16::from_le_bytes([b[34], b[35]]) as usize, layout.groups().len());
    assert_eq!(u16::from_le_bytes([b[36], b[37]]) as usize, layout.num_dn());
    assert_eq!(u16::from_le_bytes([b[38], b[39]]), 48);
    assert_eq!((b[40], b[41], b[42], b[43]), (4, 7, 3, 0));
    assert_eq!(&b[26..34], &[0u8; 8], "no proposer: the totals are zero");
}

#[test]
fn quantisation_clamps_and_survives_nan() {
    assert_eq!(q_unsigned(-1.0, 10.0), 0);
    assert_eq!(q_unsigned(99.0, 10.0), 255);
    assert_eq!(q_unsigned(f32::NAN, 10.0), 0);
    assert_eq!(q_unsigned(5.0, 10.0), 128);
    assert_eq!(q_signed(-99.0, 10.0), -127);
    assert_eq!(q_signed(99.0, 10.0), 127);
    assert_eq!(q_signed(f32::NAN, 10.0), 0);
    assert_eq!(logit_byte(1000.0), 127);
    assert_eq!(logit_byte(-1000.0), -128);
    assert_eq!(logit_byte(f32::NAN), 0);
    // A saturated probability still gives a finite logit.
    assert!(logit_of(1.0).is_finite() && logit_of(0.0).is_finite());
}

#[test]
fn decoding_rejects_bad_frames() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(1);
    decide_n(&mut brain, 1);
    let good = brain.viz_frame_with(1, None).unwrap().to_vec();
    assert!(decode_frame(&good, 10.0, 10.0).is_ok());
    assert!(decode_frame(&[], 10.0, 10.0).is_err());
    assert!(decode_frame(&good[..HEADER_LEN - 1], 10.0, 10.0).is_err());
    assert!(
        decode_frame(&good[..good.len() - 1], 10.0, 10.0).is_err(),
        "truncated body"
    );
    let mut long = good.clone();
    long.push(0);
    assert!(decode_frame(&long, 10.0, 10.0).is_err(), "trailing byte");
    let mut bad = good.clone();
    bad[0] = b'X';
    assert!(decode_frame(&bad, 10.0, 10.0).unwrap_err().contains("magic"));
    let mut bad = good;
    bad[4] = 9;
    assert!(decode_frame(&bad, 10.0, 10.0).unwrap_err().contains("version"));
}

#[test]
fn the_emitter_drops_all_but_every_nth_decision_and_starts_with_a_frame() {
    let mut e = VizEmitter::new(8, 3);
    let due: Vec<bool> = (0..9).map(|_| e.due()).collect();
    assert_eq!(due, [true, false, false, true, false, false, true, false, false]);
    e.set_every(1);
    assert!((0..5).all(|_| e.due()));
    e.set_every(0); // clamped to 1
    assert_eq!(e.every(), 1);
}

#[test]
fn the_brain_gives_one_frame_per_new_decision_decimated_and_none_before_the_first() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_viz_every(3);
    assert!(brain.viz_frame_with(0, None).is_none(), "nothing decided yet");
    let mut frames = Vec::new();
    for i in 0..9u32 {
        let _ = brain.decide(&sample_observation(300.0 + i as f32));
        let f = brain.viz_frame_with(i, None).map(<[u8]>::to_vec);
        // A second pull for the same decision is empty, whatever the decimation says.
        assert!(brain.viz_frame_with(i, None).is_none());
        frames.push(f);
    }
    let got: Vec<usize> = frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.is_some())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(got, [0, 3, 6]);
    // The frame carries its decision's tick and sequence number.
    let f = decode_frame(frames[3].as_ref().unwrap(), 10.0, 10.0).unwrap();
    assert_eq!((f.tick, f.seq), (3, 4));
}

#[test]
fn the_meta_is_valid_json_with_the_layout_and_escapes_names() {
    let mut brain = make_brain(1, ActionSelection::Argmax);
    brain.set_identity("run\"x\\y/final".to_string(), "ab".repeat(32));
    let meta = brain.viz_meta_json("proposer");
    let v: serde_json::Value = serde_json::from_str(&meta).expect("valid JSON");
    assert_eq!(v["v"], 1);
    assert_eq!(v["role"], "proposer");
    assert_eq!(v["bundle"]["name"], "run\"x\\y/final");
    assert_eq!(v["bundle"]["sha256"], "ab".repeat(32));
    assert_eq!(v["rays"], 48);
    assert_eq!(v["bins"], 4);
    assert_eq!(v["channels"].as_array().unwrap().len(), 7);
    assert_eq!(v["scalars"].as_array().unwrap().len(), 3);
    assert_eq!(
        v["frame_bytes"].as_u64().unwrap() as usize,
        brain.viz_layout().frame_len()
    );
    assert_eq!(v["groups"].as_array().unwrap().len(), brain.viz_layout().groups().len());
    assert_eq!(v["dn"].as_array().unwrap().len(), brain.viz_layout().num_dn());
    let heads = v["heads"].as_array().unwrap();
    assert!(heads.iter().any(|h| h["action"] == "jump"));
    // No identity: `null`, and the path of the file never appears (only what was set).
    let plain = make_brain(1, ActionSelection::Argmax).viz_meta_json("fly");
    let v: serde_json::Value = serde_json::from_str(&plain).unwrap();
    assert!(v["bundle"].is_null());
    assert_eq!(json_str("a\u{1}\n"), "\"a\\u0001\\n\"");
}
