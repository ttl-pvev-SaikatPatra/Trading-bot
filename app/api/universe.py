# File: app/api/universe.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.universe_builder import UniverseBuilder
from app.models.pydantic_models import UniverseStock
from app.db.models import Universe

router = APIRouter()

@router.get("", response_model=list[UniverseStock], tags=["Universe"])
def get_current_universe(db: Session = Depends(get_db)):
    """Get the list of stocks in the current trading universe."""
    stocks = db.query(Universe).all()
    return stocks

@router.post("/rebuild", tags=["Universe"])
def rebuild_universe(db: Session = Depends(get_db)):
    """Manually trigger a rebuild of the trading universe."""
    builder = UniverseBuilder(db)
    if builder.build_and_store_universe():
        return {"status": "success", "message": "Universe rebuilt successfully."}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rebuild universe. Check logs."
        )
