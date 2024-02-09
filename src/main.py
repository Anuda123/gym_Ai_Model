from ml_obj_detection import mygymapp 
from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Members(BaseModel):
    exercise: str
    age: int
    gender: str
    duration: int
    heartRate: int
    bmi: int
    weatherConditions: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/TraingData")
async def traing(members: Members):
    print(members)
    print(members.age)
    print(members.bmi)
    
   
    if members.gender.lower() == 'female':
        basal_calories = 655 + (9.6 * members.bmi) - (4.7 * members.age)
    elif members.gender.lower() == 'male':
        basal_calories = 66 + (13.7 * members.bmi) - (6.8 * members.age)
    else:
        return {"error": "Invalid gender provided. Please provide either 'male' or 'female'."}
    
    return {"age": members.age, "bmi": members.bmi, "basal_calories": basal_calories}

