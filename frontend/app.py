import json
import secrets
import random
import logging
import sys
from datetime import datetime
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import uvicorn
from configreader.configreader import return_config
from icecream import ic
from icecream import install
from jose import jwe

# Get the directory where this file lives
BASE_DIR = Path(__file__).resolve().parent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure icecream
ic.configureOutput(
    includeContext=True,
    prefix='🍦 ',
    outputFunction=lambda *args: print(*args, file=sys.stderr)
)
install()

@lru_cache()
def read_allconfig():
    """Cached config reader"""
    return return_config(0)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifecycle manager for the FastAPI application"""
    # Startup
    try:
        logger.info("Starting up application...")
        
        # Load configuration
        app.state.config = read_allconfig()
        
        # Configure icecream mode
        icecreammode = app.state.config.get('ICECREAMMODE', 'FALSE')
        if icecreammode == 'FALSE':
            ic.disable()
        
        app.state.ic = ic
        
        logger.info("Application startup complete")
        ic(app.state.config)
        
    except Exception as e:
        logger.error(f"Failed to initialize configurations: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down application...")

# Dependency functions
def get_config(request: Request) -> dict:
    """Get configuration from app state"""
    return request.app.state.config

def get_ic(request: Request):
    """Get icecream instance from app state"""
    return request.app.state.ic

# Create FastAPI app with lifespan
app = FastAPI(
    title="For Perry",
    description="SSO app",
    version="1.0.0",
    lifespan=lifespan,
    redoc_url=None,
    default_response_class=ORJSONResponse,
    swagger_ui_parameters={"syntaxHighlight": False}
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path for templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time to response headers"""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(
        f'Endpoint: {request.url.path}, Method: {request.method}, '
        f'Process Time: {process_time:.6f}s'
    )
    return response

@app.get("/", response_class=HTMLResponse)
def index(config: dict = Depends(get_config), ic = Depends(get_ic)):
    """Index route with dependency injection"""
    logger.debug(f"Loaded config keys: {list(config.keys())}")

    tokens = config.get('AUTHTOKENS')
    if isinstance(tokens, list) and len(tokens) > 0:
        AUTHTOK = tokens[random.randint(0, len(tokens) - 1)]
    else:
        AUTHTOK = config.get('AUTHTOKEN', '')

    sso_proxy = config.get('SSO_PROXY', '')
    endpoint = config.get('ENDPOINT', '')
    
    if not sso_proxy or not endpoint:
        raise HTTPException(status_code=500, detail="Missing SSO_PROXY or ENDPOINT")
    
    headerjwt = sso_proxy + endpoint + 'callback&Authorization=' + AUTHTOK
    
    logger.warning(headerjwt)
    ic(f"Redirecting to: {headerjwt}")
    
    return RedirectResponse(headerjwt)
    

@app.get("/callback", response_class=HTMLResponse)
async def callback(
    request: Request, 
    Authorization: str | None = None,
    config: dict = Depends(get_config),
    ic = Depends(get_ic)
):
    """Callback route - receives JWE encrypted token from SSO proxy"""
    title = 'For Perry'

    if Authorization is None:
        logger.error("No Authorization token provided")
        return templates.TemplateResponse(
            "error401.html", 
            {"request": request, "page": "For Perry", "title": title}
        )

    logger.info(f'Authorization token received (first 50 chars): {Authorization[:50]}...')
    
    try:
        jwe_key = config.get('JWEKEY')
        if not jwe_key:
            logger.error("JWEKEY not found in configuration")
            return templates.TemplateResponse(
                "error401.html", 
                {"request": request, "page": "For Perry", "title": title}
            )
        
        payload = jwe.decrypt(Authorization, jwe_key).decode("utf-8")
        
        if not payload:
            logger.error("JWE decryption returned empty payload")
            return templates.TemplateResponse(
                "error401.html", 
                {"request": request, "page": "For Perry", "title": title}
            )
        
        data = json.loads(payload)
        username = data.get('username', 'SSO User')
        payroll = data.get('payroll', 'N/A')
        deptcode = data.get('deptcode', '')
        
        logger.info(f"Successful SSO login: {username} ({payroll})")
        ic(f"User logged in: {username}, Payroll: {payroll}, Dept: {deptcode}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from decrypted payload: {e}")
        return templates.TemplateResponse(
            "error401.html", 
            {"request": request, "page": "For Perry", "title": title}
        )
    except Exception as e:
        logger.error(f"Failed to decrypt JWE token: {e}")
        return templates.TemplateResponse(
            "error401.html", 
            {"request": request, "page": "For Perry", "title": title}
        )

    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "For Perry",
        "username": username,
        "payroll": payroll,
        "deptcode": deptcode,
        "token": Authorization,
        "endpoint": config.get('ENDPOINT', '/'),
        "sidemenu": config.get('sidemenu', []),
        "year": datetime.now().year
    })

@app.get("/logout/{payroll}", response_class=HTMLResponse)
async def logoutscreen(
    request: Request,
    payroll: str,
    config: dict = Depends(get_config),
    ic = Depends(get_ic)
):
    """Logout route - clears user session"""
    logger.info(f"User {payroll} logging out")
    ic(f"Logout requested for: {payroll}")
    
    return templates.TemplateResponse(
        "logout.html",
        {
            "request": request,
            "endpoint": config.get('ENDPOINT', '/'),
            "year": datetime.now().year
        }
    )

@app.get("/vision-inspection", response_class=HTMLResponse)
async def vision_inspection(
    request: Request,
    config: dict = Depends(get_config),
    ic = Depends(get_ic)
):
    """Vision inspection page"""
    # Get user info from query params or session if available
    username = request.query_params.get('username', 'User')
    payroll = request.query_params.get('payroll', 'N/A')
    
    logger.info(f"Vision inspection page accessed by: {username}")
    
    return templates.TemplateResponse("vision_inspection.html", {
        "request": request,
        "page": "CT600 Vision Inspection",
        "username": username,
        "payroll": payroll,
        "token": "",
        "endpoint": config.get('ENDPOINT', '/'),
        "sidemenu": config.get('sidemenu', {}),
        "year": datetime.now().year,
        "backend_api": "http://localhost:5000"
    })

@app.get("/health")
def healthcheck():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Mount static files with absolute path at module level
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
logger.info(f"Static files mounted from: {BASE_DIR / 'static'}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5001,
        reload=False,
        access_log=True
    )