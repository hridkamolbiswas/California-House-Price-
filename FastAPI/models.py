from sqlalchemy import Column, Integer, String, Numeric
from sqlalchemy.orm import relationship

from database import Base


class house_price(Base):
    __tablename__ = "house_price"

    id = Column(Integer, primary_key=True, index=True)
    MedInc = Column(Numeric(10,2))
    HouseAge = Column(Numeric(10,2))
    AveRooms = Column(Numeric(10,2))
    AveBedrms = Column(Numeric(10,2))
    Population = Column(Numeric(10,2))
    AveOccup = Column(Numeric(10,2))
    Latitude = Column(Numeric(10,2))
    Longitude = Column(Numeric(10,2))



    #email = Column(String, unique=True, index=True)
    #hashed_password = Column(String)
    #is_active = Column(Boolean, default=True)
