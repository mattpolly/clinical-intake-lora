"""FastAPI surface for the research-only text intake demo."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import RESEARCH_DISCLAIMER
from .service import DemoService, DemoServiceError, DemoSettings


class TurnRequest(BaseModel):
    patient_text: str = Field(description="Synthetic standardized-patient text only; never PHI.")


def create_app(service: DemoService | None = None) -> FastAPI:
    app = FastAPI(title="Clinical Interview Research Demo API", version="0.1.0")
    app.state.demo_service = service or DemoService(DemoSettings.from_env())

    def run(operation):
        try:
            return operation()
        except DemoServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail={"state": exc.state, "message": str(exc), "research_only": True, "disclaimer": RESEARCH_DISCLAIMER}) from exc

    @app.post("/demo/start")
    def start_demo():
        return run(app.state.demo_service.start)

    @app.get("/demo/status/{session_id}")
    def demo_status(session_id: str):
        return run(lambda: app.state.demo_service.get_status(session_id))

    @app.post("/demo/turn/{session_id}")
    def demo_turn(session_id: str, request: TurnRequest):
        return run(lambda: app.state.demo_service.turn(session_id, request.patient_text))

    @app.post("/demo/end/{session_id}")
    def end_demo(session_id: str):
        return run(lambda: app.state.demo_service.end(session_id))

    return app
