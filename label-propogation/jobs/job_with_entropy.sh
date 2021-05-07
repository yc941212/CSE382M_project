#!/bin/bash


for lambda1 in 0.001 0.01 0.1 1.0; do
    ratio=0.01
    lambda2=$(echo $lambda1 $ratio | awk '{printf "%.7f\n",$1*$2}')
    python3 ../run.py --bound=0 --lambda_1=$lambda1 --lambda_2=$lambda2 --method='TNC' --loss_l='CE'
done

for lambda1 in 0.001 0.01 0.1 1.0; do
    ratio=0.1
    lambda2=$(echo $lambda1 $ratio | awk '{printf "%.7f\n",$1*$2}')
    python3 ../run.py --bound=0 --lambda_1=$lambda1 --lambda_2=$lambda2 --method='TNC' --loss_l='CE'
done

for lambda1 in 0.001 0.01 0.1 1.0; do
    ratio=1.0
    lambda2=$(echo $lambda1 $ratio | awk '{printf "%.7f\n",$1*$2}')
    python3 ../run.py --bound=0 --lambda_1=$lambda1 --lambda_2=$lambda2 --method='TNC' --loss_l='CE'
done





