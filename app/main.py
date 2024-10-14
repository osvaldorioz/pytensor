from fastapi import FastAPI
from lib import tensor
import time
import json
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Definir el modelo para la matriz
class ItemValues(BaseModel):
    values: List[float]

class ItemDim(BaseModel):
    values: List[int]

@app.post("/tensor")
async def tt(tensor_values: ItemValues, tensor_dim: ItemDim):
   
    start = time.time()

     # Creamos un modelo generativo 
    dimensionality = tensor_dim.values
    
    # Crear un tensor nxnxn
    t = tensor.Tensor(dimensionality)

    # Establecer algunos valores en el tensor

    valor_inicial = 0
    
    for i in tensor_values:
        print(i)
        #values=[valor_inicial,valor_inicial,valor_inicial]
        #t.set_value([1, 1, 1], i)
        #valor_inicial += 1

    t.set_value([0, 0, 0], 1.1)
    t.set_value([1, 1, 1], 2.4654)
    t.set_value([2, 2, 2], 5.567567)

    # Obtener un valor del tensor
    output1 = t.get_value([1, 1, 1])
    output2 = t.get_dimensions()
    output3 = t.get_tensor()

    end = time.time()

    var1 = 'Time taken in seconds: '
    var2 = end - start

    str = f'{var1}{var2}'.format(var1=var1, var2=var2)
    
    data = {
        "Valor del tensor": output1,
        "Dimension": output2, # Obtener las dimensiones del tensor
        "Tensor lineal": output3,
        "Lapso": str
    }
    jj = json.dumps(data)
    
    return jj