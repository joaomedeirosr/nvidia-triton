import numpy as np
import requests

input_data = np.array([[2.5]], dtype=np.float32)

model_name = "linear_regression"
model_version = "1"

url = f"http://localhost:8000/v2/models/{model_name}/versions/{model_version}/infer"

payload = {
    "inputs": [
        {
            "name": "input",
            "datatype": "FP32",
            "shape": list(input_data.shape),
            "data": input_data.tolist(),
        }
    ],
    "outputs": [
        {
            "name": "output"
        }
    ]
}

response = requests.post(url, json=payload, timeout=30)
print(f"Sucess request - Code:{response.status_code}")
print(response.text)
response.raise_for_status()

inference_result = response.json()
output_data = np.array(inference_result["outputs"][0]["data"], dtype=np.float32)
print(output_data)