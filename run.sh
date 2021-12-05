source ../../envs/python3.6/bin/activate
nohup python -u main_with_attack.py > logs/bilstm_fgm_apex.log 2>&1 &
#nohup python -u main_with_apex.py > logs/bilstm_apex_ga.log 2>&1 &
#nohup python -u kd_main.py > logs/kd_bilstm_main_T_20.log 2>&1 &
#nohup python -u main_with_gradient_accumulation.py > logs/bilstm_gradient_accumulation.log 2>&1 &
#nohup python -u main.py > logs/bilstm_main.log 2>&1 &
