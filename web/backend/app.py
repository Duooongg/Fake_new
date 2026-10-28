# web/backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from pathlib import Path

from web.backend.routes.predict import router as predict_router
from web.backend.routes.feedback import router as feedback_router

import mimetypes
mimetypes.add_type("application/javascript", ".js")

# =========================
# ROOT DIR
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

def create_app() -> FastAPI:
    app = FastAPI(
        title="Fake News Detection API",
        description="API nhận diện tin giả sử dụng SVM + CNN + Rule",
        version="1.0.0"
    )

    # =========================
    # CORS
    # =========================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================
    # STATIC FILES (SAFE)
    # =========================
    static_dir = BASE_DIR / "static"
    scripts_dir = BASE_DIR / "scripts"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        print(f"⚠️ static folder not found: {static_dir}")

    if scripts_dir.exists():
        app.mount("/scripts", StaticFiles(directory=scripts_dir), name="scripts")
    else:
        print(f"⚠️ scripts folder not found: {scripts_dir}")

    # =========================
    # API ROUTES
    # =========================
    app.include_router(predict_router, prefix="/api", tags=["Prediction"])
    app.include_router(feedback_router, prefix="/api", tags=["Feedback"])

    # =========================
    # FRONTEND (OPTIONAL)
    # =========================
    templates_dir = BASE_DIR / "templates"

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = templates_dir / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return PlainTextResponse("Frontend not found. API is running.")

    @app.get("/result", response_class=HTMLResponse)
    def result():
        html_path = templates_dir / "result.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return PlainTextResponse("Result page not found.")

    return app


app = create_app()
