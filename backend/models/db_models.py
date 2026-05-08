"""
AgroSense — SQLAlchemy DB Models & prediction logging helper
SQLite by default; swap DATABASE_URL env var for PostgreSQL in production.
"""
import os, json, datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "agrosense.db")}')

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if 'sqlite' in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Stores every AI inference made through the platform."""
    __tablename__ = 'prediction_logs'

    id         = Column(Integer, primary_key=True, index=True)
    module     = Column(String(32),  nullable=False, index=True)   # e.g. 'crop', 'disease'
    summary    = Column(String(256), nullable=False)               # human-readable one-liner
    result     = Column(String(256), nullable=False)               # top prediction
    confidence = Column(Float, nullable=True)                      # 0–100
    inputs_json  = Column(Text, nullable=True)                     # JSON dump of request inputs
    output_json  = Column(Text, nullable=True)                     # JSON dump of full response
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)

    def to_dict(self):
        return {
            'id':         self.id,
            'module':     self.module,
            'summary':    self.summary,
            'result':     self.result,
            'confidence': self.confidence,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
        }


def init_db():
    """Create all tables (safe to call multiple times)."""
    Base.metadata.create_all(bind=engine)


def log_prediction(module: str, summary: str, result: str,
                   inputs: dict = None, output: dict = None,
                   confidence: float = None):
    """
    Atomically write one prediction record to the DB.
    Silently swallows DB errors so they never break the API response.
    """
    try:
        db = SessionLocal()
        record = PredictionLog(
            module       = module,
            summary      = summary[:255],
            result       = str(result)[:255],
            confidence   = confidence,
            inputs_json  = json.dumps(inputs, default=str)  if inputs  else None,
            output_json  = json.dumps(output,  default=str) if output  else None,
        )
        db.add(record)
        db.commit()
    except Exception:
        if db:
            db.rollback()
    finally:
        try:
            if db:
                db.close()
        except Exception:
            pass
