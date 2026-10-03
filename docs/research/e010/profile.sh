#!/bin/bash
# usage: profile.sh TAG  -- backs up crates/, applies the sampler patch (+ the hook marker), builds, restores crates/, profiles
set -e
cd ~/aiddnet/wt/task-3.6
source ~/.cargo/env; export RUSTC_WRAPPER=sccache SCCACHE_DIR=~/aiddnet/data/cache/sccache SCCACHE_CACHE_SIZE=20G CARGO_BUILD_JOBS=3
BK=~/aiddnet/data/scratch/e010/crates-backup
rm -rf $BK; cp -a crates $BK
patch -p1 -s --force < ~/aiddnet/data/runs/E-010/profile-instrumentation.patch || true
python3 - <<'PY'
import re
p='crates/ddai-physics/src/collision.rs'
s=open(p).read()
if 'COL_HOOK' not in s:
    i=s.index('pub fn intersect_line_tele_hook')
    j=s.index('{\n',i)+2
    s=s[:j]+'        let _g = crate::sampler::enter(crate::sampler::COL_HOOK);\n'+s[j:]
    open(p,'w').write(s)
p='crates/ddai-physics/src/sampler.rs'
s=open(p).read()
s=s.replace('"-","-","-","-",','"thaw_escapable", "hook_allowed", "pickups bbox", "-",')
s=s.replace('pub const NAMES','pub const THAW: u8 = 44;\npub const HOOKALLOW: u8 = 45;\npub const PICKBB: u8 = 46;\npub const NAMES',1)
open(p,'w').write(s)
import re
def ins(path, sig, const):
    t=open(path).read()
    i=t.index(sig)
    j=t.index('{\n',i)+2
    t=t[:j]+'        let _gx = ddai_physics::sampler::enter(ddai_physics::sampler::'+const+');\n'+t[j:]
    open(path,'w').write(t)
ins('crates/ddai-planner/src/planner.rs','fn thaw_escapable(','THAW')
ins('crates/ddai-planner/src/planner.rs','pub(crate) fn hook_allowed(','HOOKALLOW')
t=open('crates/ddai-physics/src/world.rs').read()
t=t.replace('let characters_bbox = alive_characters_bbox(self);','let _gbb = crate::sampler::enter(crate::sampler::PICKBB); let characters_bbox = alive_characters_bbox(self); drop(_gbb);',1)
open('crates/ddai-physics/src/world.rs','w').write(t)
PY
cargo build --release --locked -p ddnet-ai 2>&1 | grep -E "^error|Finished" -A8 || true
cp target/release/ddnet-ai ~/aiddnet/data/scratch/e010/ddnet-ai-prof-$1
rm -rf crates; cp -a $BK crates
find . \( -name '*.orig' -o -name '*.rej' \) -not -path './target/*' -delete
echo "load=$(cut -d' ' -f1 /proc/loadavg)"
DDAI_SAMPLE=1 ~/aiddnet/data/scratch/e010/ddnet-ai-prof-$1 arena run --config ~/aiddnet/data/runs/E-010/prof-vs-planner.toml --out ~/aiddnet/data/runs/E-010/prof-$1 --threads 1 --arenas-dir configs/arenas 2>&1 | grep "^SAMPLE" > ~/aiddnet/data/runs/E-010/prof-$1.txt
python3 ~/aiddnet/data/runs/E-010/sumprof.py ~/aiddnet/data/runs/E-010/prof-$1.txt
