#!/usr/bin/env bash
# E-031 (task 8.7): writes configs/train/e031-ro-<arm>.toml, the BC configs of the readout arms. Same data, loss weights, optimiser, seed,
# windows and opening boost as E-029 (8.6); the network is frozen (`[fly] readout_only`), only the hook head trains.
#   usage: gen_configs.sh <arm> <init bundle> [extra dagger store ...]
# `arm` names the run (~/aiddnet/data/runs/E-031/ro-<arm>); extra stores are DAgger corpora of this task (round 100).
set -eu
ARM=$1; INIT=$2; shift 2
cd "$(dirname "$0")/../.."
EXTRA=""
for d in "$@"; do EXTRA="$EXTRA    \"$d\",\n"; done
OUT=configs/train/e031-ro-$ARM.toml
cat > "$OUT" <<EOF
# E-031 (task 8.7), readout arm "$ARM": BC of the hook head's readout with the network frozen. The data, loss weights, optimiser, seed, windows and the
# opening boost are those of E-029 (configs/train/e029-bc-legacy.toml); the hook head alone trains ($(basename "$INIT")).
#   ddnet-ai train run --config $OUT
name = "e031-ro-$ARM"
flyg = "~/aiddnet/data/connectome/compiled/fly-S-v1.flyg"
brain_config = "configs/fly/S-brain-dg.toml"
init_bundle = "$INIT"
arenas_dir = "configs/arenas"
map_dir = "~/aiddnet/data/maps"
scenarios_dir = "configs/scenarios"
run_dir = "~/aiddnet/data/runs/E-031/ro-$ARM"
teacher_base = [
    "~/aiddnet/data/datasets/teacher/round0-v1",
    "~/aiddnet/data/datasets/teacher/round1-v2",
    "~/aiddnet/data/datasets/teacher/e022-postfreeze",
    "~/aiddnet/data/runs/E-023/e023-ppo-base-s2/dagger",
    "~/aiddnet/data/runs/E-023/e023-ppo-cur4-s2/dagger",
    "~/aiddnet/data/runs/E-023/e023-ppo-cur6-s2/dagger",
    "~/aiddnet/data/runs/E-023/e023-ppo-cur7-s2/dagger",
$(printf "%b" "$EXTRA")]
teacher_dagger = "~/aiddnet/data/datasets/teacher/e031-ro-$ARM-dagger"
bc_steps = \${BC_STEPS:-800}
eval_every = 0

[model]
kind = "fly"

[fly]
lr_a = 0.0
lr_b = 0.0
lr_theta = 0.0
lr_encoder = 0.0
lr_decoder = 1e-2
lr_hook_wide = \${LR_WIDE:-3e-3}
readout_only = true
l2_a = 0.0
alpha_init = 4.0
calibration_windows = 300

[train]
seed = 2
batch_windows = 24
window_len = 32
burn_in = 6
warmup_steps = 50
lr_final_frac = 0.1
threads = 3
log_every = 50
eval_windows = 600

[train.loss]
pos_weight = [3.0, 1.0, 6.0]

[train.own_hook]
mode = "mask_hook_head"

[teacher_data]
val_mod = 10
round_weights = [[7, 6.0], [8, 6.0], [100, 6.0]]
scenario_weight = 2.0
opening_boost = 8.0
opening_ticks = 32
opening_rounds = [7, 8, 100]
aim_mask = "throw_or_fire"
EOF
# the shell expanded nothing above except $ARM etc.; resolve the two env defaults
sed -i "s/\\\${BC_STEPS:-800}/${BC_STEPS:-800}/; s/\\\${LR_WIDE:-3e-3}/${LR_WIDE:-3e-3}/" "$OUT"
echo "wrote $OUT"
