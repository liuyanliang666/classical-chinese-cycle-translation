from __future__ import annotations

from typing import Any

from ccnlp.api_service import GeneratorRegistry


def create_app(registry: GeneratorRegistry | None = None) -> Any:
    from fastapi import Body, FastAPI, HTTPException

    active_registry = registry or GeneratorRegistry.from_env()
    app = FastAPI(title="Classical Chinese NLP API")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/generate")
    def generate(payload: dict[str, Any] = Body(...)) -> dict[str, str]:
        text = str(payload.get("text", "")).strip()
        task = str(payload.get("task", "")).strip()
        try:
            style_strength = float(payload.get("style_strength", 1.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="style_strength must be a number") from exc

        try:
            result = active_registry.generate(
                text,
                task,
                style_strength=style_strength,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {"output": result.output_text, "note": result.note}

    return app


app = create_app()
