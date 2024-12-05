python train.py --model 26 --gpus 0 --sample_frame_len 2 --sample_frame_num 2 --sample_frame_stride 2 --num_layers 1 --train_bs 16 --exp_name my_exp59
clear
python test.py --model 26 --gpus 0 --sample_frame_len 2 --sample_frame_num 2 --sample_frame_stride 2 --num_layers 1 --test_bs 16 --exp_name my_exp59 \
               --test_ckpt my_exp59/epoch149.pth --similarity_calibration

cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /mnt/banana/student/khanh/iKUN-v2/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /mnt/banana/student/khanh/iKUN-v2/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp59/results \
--TRACKERS_FOLDER /mnt/banana/student/khanh/iKUN-v2/plugins/my_exp59/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/mnt/banana/student/khanh/iKUN-v2/plugins/my_exp59/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False