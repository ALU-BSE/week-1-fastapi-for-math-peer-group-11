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

# Function with NumPy
def f(x):
    num = np.matmul(M, x) + B
    return num

# Function without NumPy
def calculate_without_numpy(M, X, B):
    result = []
    for i in range(len(M)):
        row = []
        for j in range(len(X[0])):
            sum_val = sum(M[i][k] * X[k][j] for k in range(len(X))) + B[i][j]
            row.append(sum_val)
        result.append(row)
    return result

# Sigmoid function
def sigmoid(x):
    x = np.array(x)  # Ensure x is a NumPy array
    return 1 / (1 + np.exp(-x))

# POST endpoint for /calculate
@app.post("/calculate")
def calculate(matrix_input: MatrixInput):
    x = np.array(matrix_input.matrix)
    
    # Validate matrix dimensions
    if x.shape != (5, 5):
        return {"error": "Input matrix must be 5x5."}
    
    # Calculate without NumPy
    result_no_numpy = calculate_without_numpy(M.tolist(), matrix_input.matrix, B.tolist())
    
    # Calculate with NumPy
    result_numpy = f(x)
    
    # Apply sigmoid function
    sigmoid_result = sigmoid(result_numpy)
    
    return {
        "matrix_multiplication_with_numpy": result_numpy.tolist(),
        "matrix_multiplication_without_numpy": result_no_numpy,
        "sigmoid_result": sigmoid_result.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
