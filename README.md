# ImageBasedSatelliteTracking

Used for the Colibri Observatory to track moving satellites.

## Description

This repository contains a Python script designed for the Colibri Observatory to track moving satellites. The script processes FITS images, detects motion between frames, and provides visual feedback for telescope adjustments.

## Features

- Load and preprocess FITS images.
- Detect motion between two frames.
- Visualize detected movement.
- Provide guidance for telescope adjustments based on detected movement.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nickbarrie/ImageBasedSatelliteTracking.git
```

2. Install the required dependencies:
```bash
pip install numpy opencv-python astropy matplotlib
```

## Usage

1. Ensure you have FITS files available in a directory.
2. Run the script with the directory containing the FITS files:
```bash
python satelliteMotionDetection.py /path/to/fits/files
```

## Example
  ![Example Image](Example.png)
