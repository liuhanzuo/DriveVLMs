#!/bin/bash

# Set your source and destination directories
SOURCE_DIR="/root/autodl-tmp/DriveVLMs/data/models/phi-4-multimodal-finetuned-merged"
DEST_DIR="/root/autodl-tmp/DriveVLMs/data/models/phi-4-multimodal-finetuned-merged-ssd-pruned"

# File to exclude (relative to SOURCE_DIR)
EXCLUDED_FILE="modeling_phi4mm.py"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop over files in source directory
for filepath in "$SOURCE_DIR"/*; do
    filename=$(basename "$filepath")

    # Skip the excluded file
    if [[ "$filename" == "$EXCLUDED_FILE" ]]; then
        continue
    fi

    # Create symbolic link in destination directory
    ln -s "$filepath" "$DEST_DIR/$filename"
done
