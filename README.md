# CT600 Vision Inspection System

Production-ready vision inspection system with AI-powered edge detection.

## Quick Start

```bash
# Start both applications
start_all.bat

# Or use Docker
docker-compose up -d
```

**Access:** http://localhost:5001/vision-inspection

## Architecture

- **Backend (Port 5000):** Pure API - YOLO vision processing
- **Frontend (Port 5001):** Material Design UI

## Requirements

- Python 3.10+
- Docker (optional)
- Camera (optional)

## Setup

### Option 1: Direct Run
```bash
# Backend
cd backend
pip install -r requirements_fastapi.txt
python app_fastapi.py

# Frontend  
cd frontend
pip install -r requirements.txt
python app.py
```

### Option 2: Docker
```bash
docker-compose up --build
```

## Features

- ✅ Live camera feed
- ✅ Image upload/capture
- ✅ AI edge detection (YOLO)
- ✅ Measurement in microns
- ✅ Quality judgment
- ✅ Result export (Excel)

## API Endpoints

- `GET /health` - Health check
- `POST /capture` - Capture image
- `POST /` - Process image
- `POST /manual-submit` - Manual mode
- `POST /save-image` - Save result

## Structure

```
├── backend/          # API service
│   ├── app_fastapi.py
│   ├── requirements_fastapi.txt
│   └── Dockerfile
├── frontend/         # UI application
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   ├── static/
│   └── Dockerfile
└── docker-compose.yml
```

## License

Proprietary - Internal Use Only

