#!/bin/bash

set -e

echo "--- WARNING: This script will download the COCO 2017 dataset, which is approximately 25GB in size. ---"
echo "--- Please ensure you have sufficient disk space (~40GB). ---"

echo "--- COCO_data Directory ---"
mkdir -p COCO_data
cd COCO_data

echo "--- Handling Annotations ---"
if [ ! -d "annotations" ]; then
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
else
    echo "Annotations already exist, skipping..."
fi

# 3. Download and unzip Training Images (~18GB)
echo "--- Handling Training Images (Warning: This will take a while) ---"
if [ ! -d "train2017" ]; then
    wget -c http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
else
    echo "Training images already exist, skipping..."
fi

# 4. Download and unzip Validation Images (~1GB)
echo "--- Handling Validation Images ---"
if [ ! -d "val2017" ]; then
    wget -c http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
else
    echo "Validation images already exist, skipping..."
fi
