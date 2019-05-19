#!/bin/bash
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 0.1 &&
echo 0.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 0.5 &&
echo 3.57 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 1 &&
echo 7.14 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 2 &&
echo 10.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 5 &&
echo 14.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 10 &&
echo 17.86 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 20 &&
echo 21.43 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 0.1 &&
echo 25.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 0.5 &&
echo 28.57 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 1 &&
echo 32.14 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 2 &&
echo 35.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 5 &&
echo 39.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 10 &&
echo 42.86 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 20 &&
echo 46.43 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 0.1 &&
echo 50.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 0.5 &&
echo 53.57 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 1 &&
echo 57.14 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 2 &&
echo 60.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 5 &&
echo 64.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 10 &&
echo 67.86 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type var-step --max_it 500 --gamma_0 20 &&
echo 71.43 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 0.1 &&
echo 75.0 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 0.5 &&
echo 78.57 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 1 &&
echo 82.14 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 2 &&
echo 85.71 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 5 &&
echo 89.29 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 10 &&
echo 92.86 %
mpirun -n 21 python3 sign_sgd_majority.py --dataset australian --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 500 --gamma_0 20 &&
echo 96.43 %


open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_australian_sign_sgd_majority_one-point_log-reg_var-step_20_10,0_1/output_sign_sgd_majority_one-point_log-reg_var-step_20_10,0_1.txt
open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_australian_sign_sgd_majority_one-point_log-reg_fix-step_20_10,0_1/output_sign_sgd_majority_one-point_log-reg_fix-step_20_10,0_1.txt
open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_australian_sign_sgd_majority_two-point_log-reg_var-step_20_10,0_1/output_sign_sgd_majority_two-point_log-reg_var-step_20_10,0_1.txt
open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_australian_sign_sgd_majority_two-point_log-reg_fix-step_20_10,0_1/output_sign_sgd_majority_two-point_log-reg_fix-step_20_10,0_1.txt
