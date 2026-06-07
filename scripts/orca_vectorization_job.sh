python src/benchmark_vectorization.py \
    --device cuda \
    --output results/orca/vectorization_orca_gpu.csv \
    --sizes 1000 10000 100000 1000000 \
    --repeats 100 \
    --loop-repeats 1 \
    --max-loop-n 1000000