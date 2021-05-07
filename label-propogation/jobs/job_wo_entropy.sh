#!/bin/bash
for lambda in 0.01 0.1 0.5 1.0 5.0 10.0 100.0; do
    python3 ../run.py  --bound=1 --lambda_1=$lambda
done


