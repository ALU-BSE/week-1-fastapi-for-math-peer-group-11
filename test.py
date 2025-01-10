from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI()

# Define a Pydantic model for input validation
class MatrixInput(BaseModel):
    matrix: list[list[float]]

# Predefined matrices
M = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])

B = np.array([[0, 2, 4, 6, 8],
              [1, 3, 5, 7, 9],
              [2, 4, 6, 8, 10],
              [3, 5, 7, 9, 11],
              [4, 6, 8, 10, 12]])



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
