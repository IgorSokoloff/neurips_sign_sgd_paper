#!/bin/bash
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.1 &&
echo 0.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.5 &&
echo 2.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1.0 &&
echo 4.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 2.0 &&
echo 6.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 5.0 &&
echo 8.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 10.0 &&
echo 10.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 20.0 &&
echo 12.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 50.0 &&
echo 14.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 100.0 &&
echo 16.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 200.0 &&
echo 18.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 500.0 &&
echo 20.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1000.0 &&
echo 22.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.1 &&
echo 25.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.5 &&
echo 27.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1.0 &&
echo 29.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 2.0 &&
echo 31.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 5.0 &&
echo 33.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 10.0 &&
echo 35.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 20.0 &&
echo 37.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 50.0 &&
echo 39.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 100.0 &&
echo 41.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 200.0 &&
echo 43.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 500.0 &&
echo 45.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1000.0 &&
echo 47.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.1 &&
echo 50.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.5 &&
echo 52.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1.0 &&
echo 54.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 2.0 &&
echo 56.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 5.0 &&
echo 58.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 10.0 &&
echo 60.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 20.0 &&
echo 62.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 50.0 &&
echo 64.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 100.0 &&
echo 66.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 200.0 &&
echo 68.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 500.0 &&
echo 70.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1000.0 &&
echo 72.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.1 &&
echo 75.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.5 &&
echo 77.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1.0 &&
echo 79.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 2.0 &&
echo 81.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 5.0 &&
echo 83.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 10.0 &&
echo 85.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 20.0 &&
echo 87.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 50.0 &&
echo 89.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 100.0 &&
echo 91.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 200.0 &&
echo 93.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 500.0 &&
echo 95.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1000.0 &&
echo 97.92 %


open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_one-point_sigmoid_var-step_20_5,0_1/output_sign_sgd_majority_one-point_sigmoid_var-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_one-point_sigmoid_fix-step_20_5,0_1/output_sign_sgd_majority_one-point_sigmoid_fix-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_two-point_sigmoid_var-step_20_5,0_1/output_sign_sgd_majority_two-point_sigmoid_var-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_two-point_sigmoid_fix-step_20_5,0_1/output_sign_sgd_majority_two-point_sigmoid_fix-step_20_5,0_1.txt
