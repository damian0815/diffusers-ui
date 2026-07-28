import json
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from backend.training.config import TIConfig
from backend.gallery.db import AsyncSessionLocal
from backend.training.models import TIProject

router = APIRouter(prefix="/api/v1/training", tags=["training"])

class TIStartRequest(BaseModel):
    config_version: int = 1
    gallery_id: str
    placeholder_token: str = "<sks>"
    initializer_token: str = "dog"
    num_vectors: int = 1
    learnable_property: str = "object"
    class_prompt: str = ""
    repeats: int = 100
    model_key: str
    resolution: int = 512
    center_crop: bool = False
    train_batch_size: int = 1
    max_train_steps: int = 500
    learning_rate: float = 5e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    mixed_precision: str = "bf16"
    seed: int = 42
    validation_prompt: str = ""
    validation_steps: int = 100
    num_validation_images: int = 4
    validation_seed: int = 42
    save_steps: int = 100
    output_name: str = "learned_embeds"

@router.post("/start")
async def start_training(request: Request, body: TIStartRequest):
    trainer = request.app.state.ti_trainer
    config = TIConfig(**body.model_dump())
    try:
        await trainer.start(config)
        return {"status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/pause")
async def pause_training(request: Request):
    trainer = request.app.state.ti_trainer
    trainer.pause()
    return {"status": "paused"}

@router.put("/resume")
async def resume_training(request: Request):
    trainer = request.app.state.ti_trainer
    trainer.resume()
    return {"status": "resumed"}

@router.put("/stop")
async def stop_training(request: Request):
    trainer = request.app.state.ti_trainer
    trainer.stop()
    return {"status": "stopping"}

@router.post("/samples")
async def generate_samples(request: Request):
    trainer = request.app.state.ti_trainer
    if not trainer.is_running:
        raise HTTPException(status_code=400, detail="Training is not running")
    trainer.request_samples()
    return {"status": "requested"}

@router.put("/samples/cancel")
async def cancel_samples(request: Request):
    trainer = request.app.state.ti_trainer
    trainer.cancel_samples()
    return {"status": "cancelled"}

@router.get("/status")
async def training_status(request: Request):
    trainer = request.app.state.ti_trainer
    return trainer.state.__dict__

@router.get("/embeddings")
async def list_embeddings(request: Request):
    service = request.app.state.embeddings
    return [e.__dict__ for e in service.all()]

@router.delete("/embeddings/{name}")
async def delete_embedding(request: Request, name: str):
    service = request.app.state.embeddings
    if service.delete(name):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Embedding not found")


@router.get("/projects")
async def list_projects():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            TIProject.__table__.select().order_by(TIProject.created_at.desc())
        )
        rows = result.fetchall()
        return [
            {"id": r.id, "name": r.name, "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ]


class ProjectCreateRequest(BaseModel):
    name: str
    config_version: int = 1
    gallery_id: str
    placeholder_token: str = "<sks>"
    initializer_token: str = "dog"
    num_vectors: int = 1
    learnable_property: str = "object"
    class_prompt: str = ""
    repeats: int = 100
    model_key: str = ""
    resolution: int = 512
    center_crop: bool = False
    train_batch_size: int = 1
    max_train_steps: int = 500
    learning_rate: float = 5e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    mixed_precision: str = "bf16"
    seed: int = 42
    validation_prompt: str = ""
    validation_steps: int = 100
    num_validation_images: int = 4
    validation_seed: int = 42
    save_steps: int = 100
    output_name: str = "learned_embeds"


@router.post("/projects")
async def create_project(body: ProjectCreateRequest):
    config_fields = body.model_dump(exclude={"name"})
    async with AsyncSessionLocal() as session:
        project = TIProject(name=body.name, config=json.dumps(config_fields))
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return {
            "id": project.id,
            "name": project.name,
            "created_at": project.created_at.isoformat() if project.created_at else None,
        }


@router.get("/projects/{project_id}")
async def get_project(project_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            TIProject.__table__.select().where(TIProject.id == project_id)
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        config = json.loads(row.config) if row.config else {}
        return {
            "id": row.id,
            "name": row.name,
            "config": config,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


@router.put("/projects/{project_id}")
async def update_project(project_id: int, body: ProjectCreateRequest):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            TIProject.__table__.select().where(TIProject.id == project_id)
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        config_fields = body.model_dump(exclude={"name"})
        await session.execute(
            TIProject.__table__.update()
            .where(TIProject.id == project_id)
            .values(name=body.name, config=json.dumps(config_fields))
        )
        await session.commit()
        return {
            "id": project_id,
            "name": body.name,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


@router.delete("/projects/{project_id}")
async def delete_project(project_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            TIProject.__table__.select().where(TIProject.id == project_id)
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        await session.execute(
            TIProject.__table__.delete().where(TIProject.id == project_id)
        )
        await session.commit()
        return {"ok": True}
