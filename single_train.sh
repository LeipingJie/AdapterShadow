#!/bin/bash
#SBATCH --job-name go
#SBATCH --nodelist hpcgpu05
#SBATCH --output=job_logs/%A_freeze_pt_small_grid16.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

# activate your enviornment
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate env_aaron

python pl_train_exp.py -net sam -exp_name freeze_pt_small_grid16 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point -sample grid -grid_out_size 16

# python pl_train_exp.py -net sam -exp_name freeze_b1_neg_5 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# python pl_train_exp.py -net sam -exp_name freeze_5_pts_mask -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point -type gen_mask

# python pl_train_exp.py -net sam -exp_name freeze_b1_1_1 -npts 1 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_2_2 -npts 2 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_3_3 -npts 3 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_4_4 -npts 4 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_5_5 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_6_6 -npts 6 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_7_7 -npts 7 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_8_8 -npts 8 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_9_9 -npts 9 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_10_10 -npts 10 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_15_15 -npts 15 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_20_20 -npts 20 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_5_0 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b1_10_0 -npts 10 -backbone b1 -plug_image_adapter -all -freeze_backbone
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b0_5_5 -npts 5 -backbone b0 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b2_5_5 -npts 5 -backbone b2 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b3_5_5 -npts 5 -backbone b3 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b4_5_5 -npts 5 -backbone b4 -plug_image_adapter -all -freeze_backbone -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_b5_5_5 -npts 5 -backbone b5 -plug_image_adapter -all -freeze_backbone -use_neg_point

# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_1 -npts 1 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_2 -npts 2 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_3 -npts 3 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_4 -npts 4 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_5 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_6 -npts 6 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_7 -npts 7 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_8 -npts 8 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_9 -npts 9 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_10 -npts 10 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_15 -npts 15 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b1_neg_20 -npts 20 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point

# python pl_train_exp.py -net sam -exp_name freeze_small_istd_200 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -dataset_name istd -epochs 200
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_cuhk_200 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -dataset_name cuhk -epochs 200

# python pl_train_exp.py -net sam -exp_name freeze_small_b5 -npts 5 -backbone b5 -plug_image_adapter -all -freeze_backbone -small_size
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_b5_neg -npts 5 -backbone b5 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_vpt -npts 5 -backbone b5 -plug_image_adapter -all -freeze_backbone -small_size -vpt
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small_vpt_neg -npts 5 -backbone b5 -plug_image_adapter -all -freeze_backbone -small_size -vpt -use_neg_point

# python pl_train_exp.py -net sam -exp_name freeze_a1a2 -npts 5 -backbone b1 -plug_image_adapter -a1 -freeze_backbone -small_size -vpt -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_a1a2 -npts 5 -backbone b1 -plug_image_adapter -a2 -freeze_backbone -small_size -vpt -use_neg_point

# python pl_train_exp.py -net sam -exp_name freeze_vpt_16x16 -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -vpt

# python pl_train_exp.py -net sam -exp_name freeze_small -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -use_neg_point
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -vpt
# wait
# python pl_train_exp.py -net sam -exp_name freeze_small -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -small_size -vpt -use_neg_point

# python pl_train_exp.py -net sam -exp_name multi_branch_0.25 -npts 5 -backbone b1 -plug_image_adapter -all -multi_branch -mb_ratio 0.25
# wait
# python pl_train_exp.py -net sam -exp_name multi_branch_0.5 -npts 5 -backbone b1 -plug_image_adapter -all -multi_branch -mb_ratio 0.5
# wait
# python pl_train_exp.py -net sam -exp_name multi_branch_0.75 -npts 5 -backbone b1 -plug_image_adapter -all -multi_branch -mb_ratio 0.75
# wait
# python pl_train_exp.py -net sam -exp_name multi_branch_1.0 -npts 5 -backbone b1 -plug_image_adapter -all -multi_branch -mb_ratio 1
# wait
# python pl_train_exp.py -net sam -exp_name multi_branch_2 -npts 5 -backbone b1 -plug_image_adapter -all -multi_branch -mb_ratio 2

# python pl_train_exp.py -net sam -exp_name retrain_a1 -npts 5 -backbone b1 -plug_image_adapter -a1
# wait
# python pl_train_exp.py -net sam -exp_name retrain_a2 -npts 5 -backbone b1 -plug_image_adapter -a2
# wait
# python pl_train_exp.py -net sam -exp_name multi_branch0.25 -npts 5 -backbone b1 -plug_image_adapter -all -multi_branch -mb_ratio 0.25
# wait
# python pl_train_exp.py -net sam -exp_name gen_pt5_cuhk_100 -npts 5 -bs 2 -plug_image_adapter -all -dataset_name cuhk -epochs 100
# wait
# python pl_train_exp.py -net sam -exp_name gen_pt5_istd_100 -npts 5 -bs 2 -plug_image_adapter -all -dataset_name istd -epochs 100

# python pl_train.py -net sam -exp_name gen_pt5_adapter_local_vit -npts 5 -bs 2 -plug_image_adapter -local_vit -down_ratio 0.25
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_local_vit -npts 5 -bs 2 -plug_image_adapter -local_vit -down_ratio 0.5

# python pl_train.py -net sam -exp_name gen_pt5_adapter_all_tba_128 -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -all

# python pl_train.py -net sam -exp_name gen_pt5_adapter_a1_tba -backbone b1 -npts 5 -type gen_pt -bs 2 -tba
# wait

# python pl_train.py -net sam -exp_name gen_pt5_adapter_a1_tba -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -a1
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_a2_tba -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -a2
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_all_tba -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -all
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_a1_tba_neg -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -a1 -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_a2_tba_neg -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -a2 -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_all_tba_neg -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -tba -all -use_neg_point
# wait

# python pl_train.py -net sam -exp_name gen_pt2_adapter_neg -npts 2 -bs 2 -plug_image_adapter -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt3_adapter_neg -npts 3 -bs 2 -plug_image_adapter -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt4_adapter_neg -npts 4 -bs 2 -plug_image_adapter -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter_neg -npts 5 -bs 2 -plug_image_adapter -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt6_adapter_neg -npts 6 -bs 2 -plug_image_adapter -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt2_adapter -npts 2 -bs 2 -plug_image_adapter
# wait
# python pl_train.py -net sam -exp_name gen_pt3_adapter -npts 3 -bs 2 -plug_image_adapter
# wait
# python pl_train.py -net sam -exp_name gen_pt4_adapter -npts 4 -bs 2 -plug_image_adapter
# wait
# python pl_train.py -net sam -exp_name gen_pt5_adapter -npts 5 -bs 2 -plug_image_adapter
# wait
# python pl_train.py -net sam -exp_name gen_pt6_adapter -npts 6 -bs 2 -plug_image_adapter

# python pl_train.py -net sam -exp_name gen_fusion_pt4_adapter_neg -backbone b1 -npts 4 -bs 2 -plug_image_adapter -use_neg_point -plug_features_fusion
# wait 
# python pl_train.py -net sam -exp_name gen_fusion_pt5_adapter_neg -backbone b1 -npts 5 -bs 2 -plug_image_adapter -use_neg_point -plug_features_fusion
# wait 
# python pl_train.py -net sam -exp_name gen_fusion_pt6_adapter_neg -backbone b1 -npts 6 -bs 2 -plug_image_adapter -use_neg_point -plug_features_fusion
# wait 
# python pl_train.py -net sam -exp_name gen_fusion_pt4_adapter -backbone b1 -npts 4 -bs 2 -plug_image_adapter -plug_features_fusion
# wait 
# python pl_train.py -net sam -exp_name gen_fusion_pt5_adapter -backbone b1 -npts 5 -bs 2 -plug_image_adapter -plug_features_fusion
# wait 
# python pl_train.py -net sam -exp_name gen_fusion_pt6_adapter -backbone b1 -npts 6 -bs 2 -plug_image_adapter -plug_features_fusion

# python pl_train.py -net sam -exp_name gen_pt5_adapter_negpts_b5 -backbone b5 -npts 5 -type gen_pt -loss_type focal -bs 2 -plug_image_adapter -use_neg_point



# wait
# python pl_train.py -net sam -exp_name gen_pt5_noadapter_b1_negpts -backbone b1 -npts 5 -type gen_pt -loss_type focal -bs 2  -use_neg_point
# wait
# python pl_train.py -net sam -exp_name gen_pt5_noadapter_b5 -backbone b5 -npts 5 -type gen_pt -loss_type focal -bs 2
# wait
# python pl_train.py -net sam -exp_name gen_pt5_noadapter_b5_negpts -backbone b5 -npts 5 -type gen_pt -loss_type focal -bs 2 -use_neg_point
# test cmd
# python pl_test.py -net sam -exp_name test -backbone b5 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -use_neg_point -gpus 3
# python pl_train.py -net sam -exp_name test -backbone b1 -npts 5 -type gen_pt -bs 2 -plug_image_adapter -use_neg_point -gpus 3
# python pl_train_exp.py -net sam -exp_name test -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -vpt -gpus 3