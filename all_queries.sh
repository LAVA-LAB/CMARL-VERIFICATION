##################################################################
# Normal RL verification
#Model Building Time: 8440.37231183052
#Model Size: 11629
#Transitions 27157
#Pmin=? [F ALL_OPS_ZERO=true ]
#Parse Properties...
#Model Checking Time: 0.012749671936035156
#Pmin=? [F ALL_OPS_ZERO=true ] : 0.5862828089454577
#Model Size 11629
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F ALL_OPS_ZERO=true ]"
#Model Building Time: 6432.212338685989
#Model Size: 11479
#Transitions 27007
#Pmin=? [F COLLISION=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.005870819091796875
#Write to file test.drn.
#Pmin=? [F COLLISION=true ] : 0.22329460330932524
#Model Size 11479
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F COLLISION=true ]"
#Model Building Time: 5065.15648317337
#Model Size: 9241
#Transitions 24769
#Pmin=? [F NO_BUDGET_AVAILABLE=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.004967927932739258
#Write to file test.drn.
#Pmin=? [F NO_BUDGET_AVAILABLE=true ] : 0.004158619320844406
#Model Size 9241
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F NO_BUDGET_AVAILABLE=true ]"
#Model Building Time: 8335.766705989838
#Model Size: 11871
#Transitions 27399
#Pmin=? [F TIME_OVER=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.0039365291595458984
#Write to file test.drn.
#Pmin=? [F TIME_OVER=true ] : 4.226941679918599e-21
#Model Size 11871
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F TIME_OVER=true ]"
#Model Building Time: 8162.931264877319
#Model Size: 11711
#Transitions 27239
#Pmin=? [F WRONG_OPERATION_ORDER=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.0036809444427490234
#Write to file test.drn.
#Pmin=? [F WRONG_OPERATION_ORDER=true ] : 0.25163271764904765
#Model Size 11711
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F WRONG_OPERATION_ORDER=true ]"
##################################################################
# Adversarial RL verification
#Model Building Time: 28214.193786621094
#Model Size: 14762
#Transitions 34511
#Attack Config (empty=No Attack): fgsm,0.1
#Pmin=? [F ALL_OPS_ZERO=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.01045846939086914
#Write to file test.drn.
#Pmin=? [F ALL_OPS_ZERO=true ] : 0.5648878559718088
#Model Size 14762
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F ALL_OPS_ZERO=true ]" --attack_config="fgsm,0.1"
#Model Building Time: 26093.359257936478
#Model Size: 14610
#Transitions 34359
#Attack Config (empty=No Attack): fgsm,0.1
#Pmin=? [F COLLISION=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.010269403457641602
#Write to file test.drn.
#Pmin=? [F COLLISION=true ] : 0.26050514540845005
#Model Size 14610
#python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F COLLISION=true ]" --attack_config="fgsm,0.1"
#Model Building Time: 17854.116832733154
#Model Size: 11818
#Transitions 31567
#Attack Config (empty=No Attack): fgsm,0.1
#Pmin=? [F NO_BUDGET_AVAILABLE=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.0071985721588134766
#Write to file test.drn.
#Pmin=? [F NO_BUDGET_AVAILABLE=true ] : 0.014245890263015112
#Model Size 11818
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F NO_BUDGET_AVAILABLE=true ]" --attack_config="fgsm,0.1"
#Model Building Time: 27683.46271967888
#Model Size: 15086
#Transitions 34835
#Attack Config (empty=No Attack): fgsm,0.1
#Pmin=? [F TIME_OVER=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.006321907043457031
#Write to file test.drn.
#Pmin=? [F TIME_OVER=true ] : 1.925350660491063e-21
#Model Size 15086
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F TIME_OVER=true ]" --attack_config="fgsm,0.1"
#Model Building Time: 26189.31150841713
#Model Size: 14571
#Transitions 34320
#Attack Config (empty=No Attack): fgsm,0.1
#Pmin=? [F WRONG_OPERATION_ORDER=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.0047149658203125
#Write to file test.drn.
#Pmin=? [F WRONG_OPERATION_ORDER=true ] : 0.25183844066598504
#Model Size 14571
python cool_mc.py --parent_run_id=b932000bf6764b338ba34b66af43f64d --task=rl_model_checking --project_name="experiments" --prop="P=? [F WRONG_OPERATION_ORDER=true ]" --attack_config="fgsm,0.1"

##################################################################
# Adversarial RL + Autoencoder verification
#Model Building Time: 10135.352601528168
#Model Size: 12546
#Transitions 28525
#Pmin=? [F ALL_OPS_ZERO=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.00833892822265625
#Write to file test.drn.
#Pmin=? [F ALL_OPS_ZERO=true ] : 0.5862828089454577
#Model Size 12546
#python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F ALL_OPS_ZERO=true ]" --prism_file_path="scheduling_task.prism"
#Model Building Time: 23372.255109786987
#Model Size: 12623
#Transitions 28660
#Pmin=? [F ALL_OPS_ZERO=true ]
#Parse Properties...
#Model Cheking...
#Model Checking Time: 0.005775928497314453
#Write to file test.drn.
#Pmin=? [F ALL_OPS_ZERO=true ] : 0.5862828089454577
#Model Size 12623
#python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F ALL_OPS_ZERO=true ]" --prism_file_path="scheduling_task.prism" --permissive_input="autoencoder,0.1"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F COLLISION=true ]" --prism_file_path="scheduling_task.prism"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F COLLISION=true ]" --prism_file_path="scheduling_task.prism" --permissive_input="autoencoder,0.1"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F NO_BUDGET_AVAILABLE=true ]" --prism_file_path="scheduling_task.prism"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F NO_BUDGET_AVAILABLE=true ]" --prism_file_path="scheduling_task.prism" --permissive_input="autoencoder,0.1"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F TIME_OVER=true ]" --prism_file_path="scheduling_task.prism"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F TIME_OVER=true ]" --prism_file_path="scheduling_task.prism" --permissive_input="autoencoder,0.1"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F WRONG_OPERATION_ORDER=true ]" --prism_file_path="scheduling_task.prism"
python cool_mc.py --parent_run_id=prol3tabhhdc9irx3uxxpdw8lsz9g10vzlhdb6vpgt --task=rl_model_checking --project_name="experiments" --constant_definitions="" --prop="P=? [F WRONG_OPERATION_ORDER=true ]" --prism_file_path="scheduling_task.prism" --permissive_input="autoencoder,0.1"