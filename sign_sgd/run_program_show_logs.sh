#!/bin/bash
mpirun -n 4 python3 sign_sgd_majority.py --upd_option one-point --loss_func log-reg --step_type var-step --max_it 200 --gamma_0 10 &&
mpirun -n 4 python3 sign_sgd_majority.py --upd_option one-point --loss_func log-reg --step_type fix-step --max_it 200 --gamma_0 10 &&
mpirun -n 4 python3 sign_sgd_majority.py --upd_option two-point --loss_func log-reg --step_type var-step --max_it 200 --gamma_0 10 &&
mpirun -n 4 python3 sign_sgd_majority.py --upd_option two-point --loss_func log-reg --step_type fix-step --max_it 200 --gamma_0 10 &&


open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_mushrooms_sign_sgd_majority_one-point_log-reg_var-step_3_10,0_1/output_sign_sgd_majority_one-point_log-reg_var-step_3_10,0_1.txt
open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_mushrooms_sign_sgd_majority_one-point_log-reg_fix-step_3_10,0_1/output_sign_sgd_majority_one-point_log-reg_fix-step_3_10,0_1.txt
open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_mushrooms_sign_sgd_majority_two-point_log-reg_var-step_3_10,0_1/output_sign_sgd_majority_two-point_log-reg_var-step_3_10,0_1.txt
open /Users/igorsokolov/Yandex.Disk.localized/MIPT/Science/Richtarik/signSGD/experiments/sign_sgd/logs_mushrooms_sign_sgd_majority_two-point_log-reg_fix-step_3_10,0_1/output_sign_sgd_majority_two-point_log-reg_fix-step_3_10,0_1.txt
