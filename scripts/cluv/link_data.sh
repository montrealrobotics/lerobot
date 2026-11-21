#!/bin/bash
# Put robocasa's untracked assets into this checkout from $DEXROBOT_DATA. Idempotent and safe to
# run from several jobs at once; common.sh calls it at the start of every job.
#   objects/objaverse, textures, generative_textures: wholly untracked, so symlinked.
#   fixtures/: tracked XML next to untracked meshes, so the meshes are copied in (once).
set -euo pipefail

DATA=${DEXROBOT_DATA:?}
ASSETS=third_party/robocasa/robocasa/models/assets
[ -d "$ASSETS" ] || { echo "no $ASSETS -- run from the project root after cluv sync" >&2; exit 2; }

exec 9>.link_data.lock
flock 9

for d in objects/objaverse textures generative_textures; do
    src=$DATA/robocasa_assets/$d
    dst=$ASSETS/$d
    [ -e "$src" ] || { echo "missing $src" >&2; exit 2; }
    if [ -L "$dst" ]; then
        [ "$(readlink "$dst")" = "$src" ] || ln -sfn "$src" "$dst"
    elif [ -e "$dst" ]; then
        :   # a real copy is already here; leave it alone
    else
        ln -s "$src" "$dst"
    fi
done

STAMP=.fixtures_from_dexrobot_data
if [ ! -f "$STAMP" ]; then
    echo "copying robocasa fixture meshes from $DATA/robocasa_fixtures"
    rsync -a --ignore-existing "$DATA/robocasa_fixtures/" third_party/robocasa/
    date > "$STAMP"
fi
