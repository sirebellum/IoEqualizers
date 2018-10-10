model=$1
if [ -z $1 ]
then
    echo "Please specify a model!"
    exit
fi

docker run -t -i --runtime=nvidia -p 8500:8500 -p 8501:8501 \
--mount type=bind,\
source=$(pwd)/deploy/$model,\
target=/models/$model \
-e MODEL_NAME=$model -t tensorflow/serving:latest-gpu
