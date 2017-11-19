
#######################################################################
#           Copyright ARM Ltd. 2010 - 2017.                           #
#  Distributed under the Boost Software License, Version 1.0.         #
#     (See accompanying file LICENSE.txt or copy at                   #
#           http://www.boost.org/LICENSE_1_0.txt)                     #
#######################################################################

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
awk -ventry="$entry" '/^#.*Unreleased/{print;print entry;next}1' \
  CHANGELOG.md > CHANGELOG.md.tmp
mv CHANGELOG.md.tmp CHANGELOG.md

# Create git commit and add git tags
git add CHANGELOG.md VERSION.txt
git commit -q -m "Version bump to $VERSION_BUMP"
git tag -a -m "Tagging version $VERSION_BUMP" "v$VERSION_BUMP"
echo "-- New version bump commit ready to be pushed:
-------------------------------------------------------------------------------
$(git show --oneline --stat)
-------------------------------------------------------------------------------"
echo "** To drop the commit, consider running 'git reset (--hard) HEAD^'"
echo "** Note: --hard drops ALL the changes in the current directory."

# Prompt user before executing the push
read -p ">> Do you want to push the new commit upstream? Would run command:
>> $ git push origin --tags (Y|N): " INPUT_ANS
[[ "$INPUT_ANS" == "Y" ]] && git push origin --tags \
			  || { echo "-- Aborting..." && exit 1; }
