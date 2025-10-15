# Results Folder

This folder stores all saved inspection images from the CT600 Vision Inspection System.

## Folder Structure

```
results/
├── ct600/
│   ├── ct600_20250115_143022.png
│   ├── ct600_20250115_144533.png
│   └── ...
├── ct601/
│   ├── ct601_20250115_150012.png
│   └── ...
├── ct602/
│   └── ...
└── README.md
```

## Organization

- **Machine Number Folders**: Images are organized by machine number (e.g., ct600, ct601, ct602)
- **Filename Format**: `{machine_number}_{timestamp}.png`
  - Example: `ct600_20250115_143022.png`
  - Format: `{machine}_YYYYMMDD_HHMMSS.png`

## File Details

Each saved image contains:
- Original captured image with AI judgment overlay
- Measurement annotations
- Judgment result (Good/Acceptable/No Good)
- Timestamp of inspection
- Microns per pixel calibration info

## Excel Record

All saved images are also logged in:
- `../processed/measurement_results.xlsx`

This Excel file contains:
- Image Name
- Machine Number
- Saved Path
- Checked Date and Time

## Docker Volume

When running in Docker, this folder is mounted as a volume:
- Host path: `./backend/results`
- Container path: `/app/results`

This ensures all saved images persist even when the container is restarted or removed.

## API Access

Saved images can be accessed via API:
```
GET /results/{machine_number}/{filename}
```

Example:
```
GET http://localhost:5000/results/ct600/ct600_20250115_143022.png
```

## Cleanup

To clean up old results:
1. Manually delete folders/files as needed
2. Or use a scheduled cleanup script (not included)

**Note**: The Excel record is separate and won't be automatically updated if files are manually deleted.

## Permissions

Ensure the application has write permissions to this directory when running locally or in Docker.

