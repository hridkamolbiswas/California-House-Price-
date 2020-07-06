from typing import Optional
from fastapi import FastAPI, Request, Depends, BackgroundTasks
import uvicorn
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import pandas as pd

import models
from sqlalchemy.orm import Session
from database import SessionLocal, engine

from models import house_price

app = FastAPI()
models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates", )

def read_testdata():
    df= pd.read_pickle('test_data.pickle')
    df = df.reset_index(drop=True)
    #print(df.head())
    return df


class sample_request(BaseModel):
    id: int


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()




@app.get("/")
def home(request : Request):

    return templates.TemplateResponse("home.html",  {
        "request":request,
    })

def fetch_the_data(id: int):
    db = SessionLocal()

    house = db.query(house_price).filter(house_price.id == id)

    
    house.MedInc = house['MedInc']
    db.add(house)
    db.commit()

@app.post("/price")
def get_info(sample_request: sample_request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):

    house = house_price()
    house.id = sample_request.id
    print(house.id)
    df = read_testdata()
    one_sample = df.loc[house.id]
    print(one_sample)

    db.add(house)
    db.commit()

    background_tasks.add_task(fetch_the_data, house.id)


    return {
        'code' : "success",
        'message': "created",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True, debug=True)