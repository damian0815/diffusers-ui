from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime
from backend.gallery.db import Base


class TIProject(Base):
    __tablename__ = "ti_projects"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    config = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
