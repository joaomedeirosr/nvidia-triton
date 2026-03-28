import numpy as np
import tritonclient.http as httpclient

input_data = np.array([[2.5]], dtype=np.float32)

client = httpclient.InferenceServerClient(url = "localhost:8000")

# Triton pode inferir atraves dos config values
inputs = httpclient.InferInput("input", input_data.shape, datatype="FP32")
inputs.set_data_from_numpy(input_data)
print(inputs)

# output config
outputs = httpclient.InferRequestedOutput("output")
print(outputs)

# Sample inference
response = client.infer(
    model_name="linear_regression",
    inputs = [inputs], 
    outputs = [outputs]
    
)

inference_output = response.as_numpy("output")
print(inference_output)

for i in range(100):
    response = client.infer(
        model_name="linear_regression", 
        inputs = [inputs], 
        outputs=[outputs],
    )