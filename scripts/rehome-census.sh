#!/usr/bin/env bash
#
# Create a fresh TileDB-SOMA-ML clone, with some "re-homed" history:
# 1. Reproduce the Git history of `api/python/cellxgene_census/src/cellxgene_census/experimental/ml` in the CELLxGENE Census repo:
#    - This was developed between May 2023 and July 2024
#    - A few files are omitted, that are not relevant to "PyTorch loaders" work (namely the "huggingface" subdirectory)
# 2. Insert one commit moving these files to `apis/python/src/tiledbsoma/ml` (which they were copied to, in the TileDB-SOMA repo, in July 2024)
# 3. Replay the `bkmartinjr/experimentdatapipe` branch of TileDB-SOMA (developed between July 2024 and September 2024) on top of this
#    - Just the commits that touch `apis/python/src/tiledbsoma/ml` directory, or a few other relevant paths (e.g. `other_packages, where they were moved, later in that branch's development)

set -ex

pip install git-filter-repo

# Create a Census clone, filter to files/commits relevant to PyTorch loaders:
git clone -o origin https://github.com/chanzuckerberg/cellxgene-census census-ml && cd census-ml
ml=api/python/cellxgene_census/src/cellxgene_census/experimental/ml
git filter-repo \
  --path $ml/__init__.py \
  --path $ml/pytorch.py \
  --path $ml/encoders.py \
  --path $ml/util
cd ..

# Create a TileDB-SOMA clone, filter to files/commits relevant to PyTorch loaders:
git clone -o origin -b bkmartinjr/experimentdatapipe git@github.com:single-cell-data/TileDB-SOMA.git soma-pytorch && cd soma-pytorch
git branch -m main
renames=()
for p in CHANGELOG.md README.md notebooks pyproject.toml src tests; do
  renames+=(--path-rename "other_packages/python/tiledbsoma_ml/$p:$p")
done
git filter-repo --force \
  --path other_packages \
  --path apis/python/src/tiledbsoma/ml \
  --path .github/workflows/python-tiledbsoma-ml.yml \
  "${renames[@]}"
cd ..

# Initialize TileDB-SOMA-ML clone, fetch filtered Census and TileDB-SOMA branches from the adjacent directories above:
git clone https://github.com/ryan-williams/TileDB-SOMA-ML soma-ml && cd soma-ml
git remote add c ../census-ml && git fetch c
git remote add t ../soma-pytorch && git fetch t
git reset --hard c/main

# From the filtered Census HEAD, `git mv` the files to where the TileDB-SOMA branch ported them
tdbs=apis/python/src/tiledbsoma
mkdir -p $tdbs
git mv $ml $tdbs/

# Cherry-pick the root commit of the TileDB-SOMA port
root="$(git rev-list --max-parents=0 t/main)"
git cherry-pick $root
# Ensure all files match the TileDB-SOMA root commit
git status --porcelain | grep '^UU' | cut -c4- | xargs git checkout --theirs --
# Verify there are no diffs vs TileDB-SOMA root commit
git diff --exit-code $root

# Rebase `$root..t/main` (the rest of the filtered TileDB-SOMA commits) onto cherry-picked HEAD
git reset --hard t/main
git rebase --onto "HEAD@{1}" $root
