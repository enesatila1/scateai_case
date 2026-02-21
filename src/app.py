from .config import app_config
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from .router.public_router import router as public_router
from .router.music_router import router as music_router


app = FastAPI(
    title="Scate AI - Music Generation API",
    description="AI-powered music generation and voice cloning API. Generate original songs and create covers with AI voice cloning.",
    version="1.0.0",
    contact={
        "name": "Scate AI Support",
        "url": "https://scate.ai/contact",
        "email": "support@scate.ai",
    },
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(public_router)
app.include_router(music_router)


if __name__ == "__main__":
    print("🚀 Server başlatılıyor...")
    print("📝 Loglar hem console'da hem de app.log dosyasında görünecek")
    print("🔧 Debug: Current working directory:", __import__("os").getcwd())
    print("🔧 Debug: __file__:", __file__)
    print("=" * 50)

    uvicorn.run(
        "app:app",  # ---! Import string olarak geçir - reload için gerekli
        host="0.0.0.0",
        port=app_config.port,
        log_level="info",
        reload=True,
    )
