#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Creating COCO_data directory..."
mkdir -p COCO_data
cd COCO_data

# Grab the full absolute path of the new directory
CURRENT_DIR=$(pwd)

echo "Downloading COCO 2017 Train/Val Annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "-> Downloaded to: ${CURRENT_DIR}/annotations_trainval2017.zip"

echo "Unzipping Annotations..."
unzip -q annotations_trainval2017.zip

echo "Cleaning up annotations zip file..."
rm annotations_trainval2017.zip

echo "Downloading COCO 2017 Validation Images (1GB)..."
wget http://images.cocodataset.org/zips/val2017.zip
echo "-> Downloaded to: ${CURRENT_DIR}/val2017.zip"

echo "Unzipping Validation Images..."
unzip -q val2017.zip

echo "Cleaning up images zip file..."
rm val2017.zip

echo "Done! The dataset is ready in: ${CURRENT_DIR}/"