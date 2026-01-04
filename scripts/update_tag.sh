#!/bin/bash

set -e

# Check if a tag name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <tag>"
  exit 1
fi

TAG=$1

# Delete the local tag
git tag -d $TAG

# Delete the remote tag
git push origin --delete $TAG

# # Create the tag again
git tag $TAG

# # Push the tag to the remote repository
git push origin $TAG



# git ls-remote --tags origin | awk '{print $2}' | sed 's|refs/tags/||' | xargs -I {} git push origin :refs/tags/{}