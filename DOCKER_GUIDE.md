# Docker Guide - CT600 Vision Inspection System

This guide explains how to run the application using Docker and Docker Compose.

## Prerequisites

- Docker installed and running
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start with Docker Compose (Recommended)

The easiest way to run both services together:

```bash
# Build and start both services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

**Access the application:**
- Frontend: http://localhost:5001/vision-inspection
- Backend API: http://localhost:5000

## Running Individual Docker Images

### Build Individual Images

```bash
# Build backend image
cd backend
docker build -t ct600-backend:latest .

# Build frontend image
cd ../frontend
docker build -t ct600-frontend:latest .
```

### Run Individual Containers

**Backend:**
```bash
docker run -d \
  --name ct600-backend \
  -p 5000:5000 \
  -v "$(pwd)/backend/uploads:/app/uploads" \
  -v "$(pwd)/backend/processed:/app/processed" \
  -v "$(pwd)/backend/models:/app/models" \
  -v "$(pwd)/backend/results:/app/results" \
  ct600-backend:latest
```

**Frontend:**
```bash
docker run -d \
  --name ct600-frontend \
  -p 5001:5001 \
  --link ct600-backend:backend \
  -v "$(pwd)/frontend/static:/app/static" \
  -v "$(pwd)/frontend/templates:/app/templates" \
  -v "$(pwd)/frontend/conf:/app/conf" \
  ct600-frontend:latest
```

**Note:** On Windows PowerShell, use `${PWD}` instead of `$(pwd)`:
```powershell
docker run -d `
  --name ct600-backend `
  -p 5000:5000 `
  -v "${PWD}\backend\uploads:/app/uploads" `
  -v "${PWD}\backend\processed:/app/processed" `
  -v "${PWD}\backend\models:/app/models" `
  -v "${PWD}\backend\results:/app/results" `
  ct600-backend:latest
```

## Common Docker Commands

### Docker Compose Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Rebuild images before starting
docker-compose up --build

# View logs
docker-compose logs

# View logs for specific service
docker-compose logs backend
docker-compose logs frontend

# Follow logs (real-time)
docker-compose logs -f

# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers and volumes
docker-compose down -v

# Restart a specific service
docker-compose restart backend

# View running containers
docker-compose ps

# Execute command in running container
docker-compose exec backend bash
docker-compose exec frontend bash
```

### Individual Container Commands

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# View container logs
docker logs ct600-backend
docker logs ct600-frontend

# Follow logs
docker logs -f ct600-backend

# Stop container
docker stop ct600-backend

# Start stopped container
docker start ct600-backend

# Remove container
docker rm ct600-backend

# Remove image
docker rmi ct600-backend:latest

# Execute command in container
docker exec -it ct600-backend bash
docker exec -it ct600-frontend bash
```

## Service Details

### Backend Service
- **Port:** 5000
- **Container Name:** ct600-backend
- **Health Check:** http://localhost:5000/health
- **Volumes:**
  - `./backend/uploads` → `/app/uploads`
  - `./backend/processed` → `/app/processed`
  - `./backend/models` → `/app/models`
  - `./backend/results` → `/app/results`

### Frontend Service
- **Port:** 5001
- **Container Name:** ct600-frontend
- **Health Check:** http://localhost:5001/health
- **Depends On:** backend
- **Volumes:**
  - `./frontend/static` → `/app/static`
  - `./frontend/templates` → `/app/templates`
  - `./frontend/conf` → `/app/conf`

## Troubleshooting

### Check if containers are running
```bash
docker-compose ps
# or
docker ps
```

### Check container logs for errors
```bash
docker-compose logs backend
docker-compose logs frontend
```

### Rebuild after code changes
```bash
docker-compose up --build
```

### Check if ports are already in use
```bash
# Windows PowerShell
netstat -ano | findstr :5000
netstat -ano | findstr :5001

# Linux/Mac
lsof -i :5000
lsof -i :5001
```

### Remove all containers and start fresh
```bash
docker-compose down -v
docker-compose up --build
```

### Access container shell for debugging
```bash
docker-compose exec backend bash
docker-compose exec frontend bash
```

## Network Configuration

Both services run on a shared Docker network (`ct600-network`) which allows them to communicate with each other. The frontend can reach the backend using the service name `backend` on port 5000.

## Environment Variables

Both containers use:
- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered

You can add additional environment variables in `docker-compose.yml` or pass them with `-e` flag when using `docker run`.

