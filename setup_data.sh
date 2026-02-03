#!/bin/bash
set -e

# Create data directory
mkdir -p data
cd data

# Check if data already exists
if [ -d "lra_release/listops-1000" ]; then
    echo "ListOps data found in data/lra_release/listops-1000"
    exit 0
fi

# Download
if [ ! -f "lra_release.gz" ]; then
    echo "Downloading Long Range Arena dataset (this may take a while)..."
    if command -v wget >/dev/null 2>&1; then
        # wget is standard on many Linux HPCs
        wget -O lra_release.gz https://storage.googleapis.com/long-range-arena/lra_release.gz
    elif command -v curl >/dev/null 2>&1; then
        # curl fallback
        curl -L -o lra_release.gz https://storage.googleapis.com/long-range-arena/lra_release.gz
    else
        echo "Error: Neither wget nor curl found. Please download manually."
        exit 1
    fi
fi

# Extract
echo "Extracting LRA dataset..."
# The file is a gzipped tarball (despite the .gz extension often confusingly used for .tar.gz on GS)
tar -xzf lra_release.gz

echo "Setup complete. Data is in data/lra_release"
