from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI()

# Define a Pydantic model for input validation
class MatrixInput(BaseModel):
    matrix: list[list[float]]

# Predefined matrices


# use the post decorator directly below this
@app.post("/calculate")
def f(x):
    pass
 
#Implement the formula MX + B

#Have two function one using numpy and another not using numpy
#Return 

#initialize x as a 5 * 5 matrix


#Make a call to the function

#Recreate the function with the sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
