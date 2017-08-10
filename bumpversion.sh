#!/bin/bash

# Follows the Semantic Versioning convention

# Check that a VERSION file exists in the current directory
[[ -f VERSION.txt ]] || { echo "Cannot find VERSION file.
Please create one in the current directory." && exit 1; }

VERSION_ALL=($(cat VERSION.txt | tr '.' ' '))
VERSION_STR=$(printf '%s.' "${VERSION_ALL[@]}")

VERSION_MAJOR=${VERSION_ALL[0]}
VERSION_MINOR=${VERSION_ALL[1]}
VERSION_PATCH=${VERSION_ALL[2]}
echo "-- Current version: $VERSION_STR"

# Suggest minor version bump
VERSION_MINOR=$((VERSION_MINOR+1))
VERSION_PATCH=$((0))
VERSION_BUMP="$VERSION_MAJOR.$VERSION_MINOR.$VERSION_PATCH"

# Let use overwrite the default bump
read -p ">> Set new version [$VERSION_BUMP]: " INPUT_VERSION
[[ "$INPUT_VERSION" = "" ]] || VERSION_BUMP=$INPUT_VERSION

echo "-- Setting new version to v-$VERSION_BUMP"
echo $VERSION_BUMP > VERSION.txt

# Suggest CHANGELOG.md entry
echo "-- Adding new changelog entry
Please fill in the corresponding sections to document your changes as follows:"

echo "## v[$VERSION_BUMP] - $(date +'%Y-%m-%d')" >> tmpfile

echo "### Added:" | tee -a tmpfile
while true; do
  read -p ">>> - " INPUT_ADDED
  [[ "$INPUT_ADDED" = "" ]] && break;
  echo "- $INPUT_ADDED" >> tmpfile
done

echo "### Changed:" | tee -a tmpfile
while true; do
  read -p ">>> - " INPUT_ADDED
  [[ "$INPUT_ADDED" = "" ]] && break;
  echo "- $INPUT_ADDED" >> tmpfile
done

echo "### Removed:" | tee -a tmpfile
while true; do
  read -p ">>> - " INPUT_ADDED
  [[ "$INPUT_ADDED" = "" ]] && break;
  echo "- $INPUT_ADDED" >> tmpfile
done

echo "### Fixed:" | tee -a tmpfile
while true; do
  read -p ">>> - " INPUT_ADDED
  [[ "$INPUT_ADDED" = "" ]] && break;
  echo "- $INPUT_ADDED" >> tmpfile
done

# Add entry
entry="$(cat tmpfile)" && rm tmpfile
awk -ventry="$entry" '/^#.*Unreleased/{print;print entry;next}1' CHANGELOG.md > CHANGELOG.md.tmp
mv CHANGELOG.md.tmp CHANGELOG.md

# Push tags to git
git add CHANGELOG.md VERSION.txt
git commit -m "Version bump to $VERSION_BUMP"
git tag -a -m "Tagging version $VERSION_BUMP" "v$VERSION_BUMP"
git push origin --tags
