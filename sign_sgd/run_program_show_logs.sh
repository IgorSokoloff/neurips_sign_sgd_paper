#!/bin/bash
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 0.1 &&
echo 0.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 0.5 &&
echo 1.04 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 1.0 &&
echo 2.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 2.0 &&
echo 3.12 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 5.0 &&
echo 4.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 10.0 &&
echo 5.21 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 20.0 &&
echo 6.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 50.0 &&
echo 7.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 100.0 &&
echo 8.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 200.0 &&
echo 9.38 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 500.0 &&
echo 10.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 1000.0 &&
echo 11.46 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 0.1 &&
echo 12.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 0.5 &&
echo 13.54 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 1.0 &&
echo 14.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 2.0 &&
echo 15.62 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 5.0 &&
echo 16.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 10.0 &&
echo 17.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 20.0 &&
echo 18.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 50.0 &&
echo 19.79 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 100.0 &&
echo 20.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 200.0 &&
echo 21.88 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 500.0 &&
echo 22.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 1000.0 &&
echo 23.96 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 0.1 &&
echo 25.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 0.5 &&
echo 26.04 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 1.0 &&
echo 27.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 2.0 &&
echo 28.12 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 5.0 &&
echo 29.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 10.0 &&
echo 30.21 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 20.0 &&
echo 31.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 50.0 &&
echo 32.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 100.0 &&
echo 33.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 200.0 &&
echo 34.38 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 500.0 &&
echo 35.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 2000 --gamma_0 1000.0 &&
echo 36.46 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 0.1 &&
echo 37.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 0.5 &&
echo 38.54 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 1.0 &&
echo 39.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 2.0 &&
echo 40.62 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 5.0 &&
echo 41.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 10.0 &&
echo 42.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 20.0 &&
echo 43.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 50.0 &&
echo 44.79 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 100.0 &&
echo 45.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 200.0 &&
echo 46.88 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 500.0 &&
echo 47.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 2000 --gamma_0 1000.0 &&
echo 48.96 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.1 &&
echo 50.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.5 &&
echo 51.04 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1.0 &&
echo 52.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 2.0 &&
echo 53.12 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 5.0 &&
echo 54.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 10.0 &&
echo 55.21 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 20.0 &&
echo 56.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 50.0 &&
echo 57.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 100.0 &&
echo 58.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 200.0 &&
echo 59.38 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 500.0 &&
echo 60.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1000.0 &&
echo 61.46 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.1 &&
echo 62.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.5 &&
echo 63.54 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1.0 &&
echo 64.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 2.0 &&
echo 65.62 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 5.0 &&
echo 66.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 10.0 &&
echo 67.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 20.0 &&
echo 68.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 50.0 &&
echo 69.79 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 100.0 &&
echo 70.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 200.0 &&
echo 71.88 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 500.0 &&
echo 72.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1000.0 &&
echo 73.96 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.1 &&
echo 75.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 0.5 &&
echo 76.04 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1.0 &&
echo 77.08 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 2.0 &&
echo 78.12 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 5.0 &&
echo 79.17 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 10.0 &&
echo 80.21 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 20.0 &&
echo 81.25 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 50.0 &&
echo 82.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 100.0 &&
echo 83.33 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 200.0 &&
echo 84.38 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 500.0 &&
echo 85.42 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type var-step --max_it 2000 --gamma_0 1000.0 &&
echo 86.46 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.1 &&
echo 87.5 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 0.5 &&
echo 88.54 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1.0 &&
echo 89.58 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 2.0 &&
echo 90.62 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 5.0 &&
echo 91.67 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 10.0 &&
echo 92.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 20.0 &&
echo 93.75 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 50.0 &&
echo 94.79 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 100.0 &&
echo 95.83 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 200.0 &&
echo 96.88 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 500.0 &&
echo 97.92 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func sigmoid --step_type fix-step --max_it 2000 --gamma_0 1000.0 &&
echo 98.96 %


open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_one-point_log-reg_var-step_20_5,0_1/output_sign_sgd_majority_one-point_log-reg_var-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_one-point_log-reg_fix-step_20_5,0_1/output_sign_sgd_majority_one-point_log-reg_fix-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_two-point_log-reg_var-step_20_5,0_1/output_sign_sgd_majority_two-point_log-reg_var-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_two-point_log-reg_fix-step_20_5,0_1/output_sign_sgd_majority_two-point_log-reg_fix-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_one-point_sigmoid_var-step_20_5,0_1/output_sign_sgd_majority_one-point_sigmoid_var-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_one-point_sigmoid_fix-step_20_5,0_1/output_sign_sgd_majority_one-point_sigmoid_fix-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_two-point_sigmoid_var-step_20_5,0_1/output_sign_sgd_majority_two-point_sigmoid_var-step_20_5,0_1.txt
open /Users/igorsokolov/Google_Drive/sign_sgd/logs_australian_sign_sgd_majority_two-point_sigmoid_fix-step_20_5,0_1/output_sign_sgd_majority_two-point_sigmoid_fix-step_20_5,0_1.txt
