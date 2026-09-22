import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Date, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._datetime_helpers import utc_now
from app.models.types import GUID

if TYPE_CHECKING:
    from app.models.user import User


class UserQuota(Base):
    __tablename__ = "user_quotas"

    id:                 Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id:            Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    requests_today:     Mapped[int]       = mapped_column(Integer(), default=0)
    requests_month:     Mapped[int]       = mapped_column(Integer(), default=0)
    tokens_today:       Mapped[int]       = mapped_column(Integer(), default=0)
    tokens_month:       Mapped[int]       = mapped_column(Integer(), default=0)
    daily_limit:        Mapped[int]       = mapped_column(Integer(), server_default="100")
    monthly_limit:      Mapped[int]       = mapped_column(Integer(), server_default="2000")
    last_daily_reset:   Mapped[date]      = mapped_column(Date(), default=date.today)
    # First day of the current month, computed in Python — SQLite has no
    # `date_trunc`.
    last_monthly_reset: Mapped[date]      = mapped_column(
        Date(), default=lambda: date.today().replace(day=1)
    )
    updated_at:         Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=utc_now)

    user: Mapped["User"] = relationship(back_populates="quota")
