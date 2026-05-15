#!/bin/bash

# set -e

# If no tag is provided, do nothing and exit gracefully
if [ -z "$1" ]; then
  echo "No tag provided. Skipping tag re-creation."
  exit 0
fi

TAG="$1"

# Check if the local tag exists before deleting
if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  echo "Deleting existing local tag: $TAG"
  git tag -d "$TAG"
else
  echo "Local tag $TAG does not exist. Skipping deletion."
fi

# Check if the remote tag exists before deleting
if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "Deleting existing remote tag: $TAG"
  git push origin --delete "$TAG"
else
  echo "Remote tag $TAG does not exist. Skipping deletion."
fi

# Create and push the new tag
echo "Creating and pushing new tag: $TAG"
git tag "$TAG"
git push origin "$TAG"