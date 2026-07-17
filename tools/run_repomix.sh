#!/bin/bash

# Get the directory of the script to make it runnable from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Target folder defaults to the repository root if not specified
TARGET_FOLDER="."
if [ -n "$1" ]; then
  TARGET_FOLDER="$1"
fi

# Resolve absolute path of the target folder
ABS_TARGET_FOLDER="$(cd "$TARGET_FOLDER" 2>/dev/null && pwd)"

if [ -z "$ABS_TARGET_FOLDER" ]; then
  echo "Error: Directory '$TARGET_FOLDER' does not exist."
  exit 1
fi

# Get the path relative to the repository root
REL_TARGET_FOLDER=$(realpath --relative-to="$REPO_ROOT" "$ABS_TARGET_FOLDER")

# Determine output filename based on target folder
if [ "$REL_TARGET_FOLDER" = "." ]; then
  OUTPUT_FILE="repomix/repomix-output.xml"
  echo "Packing the entire repository into '$OUTPUT_FILE'..."
  # Run repomix from the repo root
  (cd "$REPO_ROOT" && npx repomix --config repomix/repomix.config.js -o "$OUTPUT_FILE")
else
  # Clean folder name for file output
  SAFE_NAME=$(echo "$REL_TARGET_FOLDER" | tr '/' '_')
  OUTPUT_FILE="repomix/repomix-output-${SAFE_NAME}.xml"
  echo "Packing '$REL_TARGET_FOLDER' into '$OUTPUT_FILE'..."
  # Run repomix on the target folder, specifying output file relative to repo root
  (cd "$REPO_ROOT" && npx repomix "$REL_TARGET_FOLDER" --config repomix/repomix.config.js -o "$OUTPUT_FILE")
fi
