#!/usr/bin/env bash
# Bump patch version on a release branch, commit, and tag (e.g. 0.2.2 -> 0.2.3).
#
# Usage: scripts/release_patch.sh
set -euo pipefail

# Verify we're on a release branch
BRANCH="$(git branch --show-current)"
if [[ ! "$BRANCH" =~ ^release-[0-9]+-[0-9]+$ ]]; then
    echo "Error: must be on a release-X-Y branch (currently on '$BRANCH')" >&2
    exit 1
fi

# Extract major.minor from branch name (release-0-2 -> 0.2)
BRANCH_VERSION="${BRANCH#release-}"
BRANCH_VERSION="${BRANCH_VERSION//-/.}"

# Read current version from pyproject.toml
CURRENT="$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')"
CURRENT_MAJOR_MINOR="${CURRENT%.*}"
CURRENT_PATCH="${CURRENT##*.}"

# Verify branch and version match
if [ "$CURRENT_MAJOR_MINOR" != "$BRANCH_VERSION" ]; then
    echo "Error: branch is release for $BRANCH_VERSION but version in pyproject.toml is $CURRENT" >&2
    exit 1
fi

# Bump patch
NEW_PATCH=$((CURRENT_PATCH + 1))
NEW_VERSION="${BRANCH_VERSION}.${NEW_PATCH}"
TAG="v${NEW_VERSION}"

# Verify tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag $TAG already exists" >&2
    exit 1
fi

echo "Bumping version: $CURRENT -> $NEW_VERSION"

# Update pyproject.toml
sed -i.bak "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" pyproject.toml
rm -f pyproject.toml.bak

# Update all Cargo.toml files
for cargo_toml in \
    crates/bloqade-lanes-bytecode-core/Cargo.toml \
    crates/bloqade-lanes-bytecode-python/Cargo.toml \
    crates/bloqade-lanes-bytecode-cli/Cargo.toml; do
    sed -i.bak "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" "$cargo_toml"
    rm -f "${cargo_toml}.bak"
done

# Regenerate uv.lock
uv lock --upgrade

# Commit and tag
git add pyproject.toml uv.lock \
    crates/bloqade-lanes-bytecode-core/Cargo.toml \
    crates/bloqade-lanes-bytecode-python/Cargo.toml \
    crates/bloqade-lanes-bytecode-cli/Cargo.toml
git commit -m "Bump version to $NEW_VERSION"
git tag "$TAG"

echo ""
echo "Version bumped to $NEW_VERSION and tagged $TAG"
echo "Push with:"
echo "  git push origin $BRANCH --tags"
