# Tomography Volume Cropper

Python script for processing X-ray tomography data from Diamond Light Source beamline K11. The script automatically detects and removes the steel pin holder regions from the top and bottom of reconstructed volumes, and normalizes the data to 8-bit.

## What it does

1. Loads reconstructed data from Savu pipeline output
2. Detects volume boundaries by analyzing slice averages
3. Crops out the steel pin regions
4. Normalizes data to 0.5-99.5 percentile range and converts to uint8
5. Saves processed volumes and diagnostic plots

## Usage

1. Update experiment numbers in `main()`:
```python
experiment_numbers = [
    '52548',
    '52549',
    # Add your experiment numbers here
]
```

2. Run the script:
```bash
python crop_volumes.py
```

## Output

For each experiment:
```
/output_base/experiment_number/
├── *_threshold_analysis.png
├── *_average_pixel_values.png
├── *_xz_slice.png
└── trimmed_data/
    └── volume.tiff
```

A processing summary is saved to `processing_summary.txt`.

## Requirements

```
numpy
pillow
matplotlib
tqdm
tifffile
```
