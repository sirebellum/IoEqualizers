docker run --runtime=nvidia -p 8501:8501 \
--mount type=bind,\
source=$(pwd)/deploy/feedback,\
target=/models/feedback \
-e MODEL_NAME=feedback -t tensorflow/serving:latest-gpu
