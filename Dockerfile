FROM nvcr.io/nvidia/tritonserver:23.02-py3

# Install dependencies
RUN pip install --no-cache-dir transformers torch