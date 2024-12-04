<<COMMENT
python train.py --model 0 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp4
python test.py --model 0 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp4 \
               --test_ckpt my_exp4/epoch149.pth --similarity_calibration

python train.py --model 1 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp5
python test.py --model 1 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp5 \
               --test_ckpt my_exp5/epoch149.pth --similarity_calibration

python train.py --model 2 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp6
python test.py --model 2 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp6 \
                         --test_ckpt my_exp6/epoch149.pth --similarity_calibration

python train.py --model 3 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp7
python test.py --model 3 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp7 \
               --test_ckpt my_exp7/epoch149.pth --similarity_calibration

python train.py --model 4 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp8
python test.py --model 4 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp8 \
               --test_ckpt my_exp8/epoch149.pth --similarity_calibration

python train.py --model 5 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp9
python test.py --model 5 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp9 \
               --test_ckpt my_exp9/epoch149.pth --similarity_calibration

python train.py --model 6 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp10
python test.py --model 6 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp10 \
               --test_ckpt my_exp10/epoch149.pth --similarity_calibration

python train.py --model 7 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp11
python test.py --model 7 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp11 \
               --test_ckpt my_exp11/epoch149.pth --similarity_calibration

python train.py --model 8 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp12
python test.py --model 8 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp12 \
               --test_ckpt my_exp12/epoch149.pth --similarity_calibration

python train.py --model 9 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp13
python test.py --model 9 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp13 \
               --test_ckpt my_exp13/epoch149.pth --similarity_calibration

python train.py --model 10 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp14
python test.py --model 10 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp14 \
               --test_ckpt my_exp14/epoch149.pth --similarity_calibration

python train.py --model 11 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp15
python test.py --model 11 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp15 \
               --test_ckpt my_exp15/epoch149.pth --similarity_calibration

python train.py --model 12 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp16
python test.py --model 12 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp16 \
               --test_ckpt my_exp16/epoch149.pth --similarity_calibration

python train.py --model 13 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp17
python test.py --model 13 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp17 \
               --test_ckpt my_exp17/epoch149.pth --similarity_calibration

python train.py --model 14 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp18
python test.py --model 14 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp18 \
               --test_ckpt my_exp18/epoch149.pth --similarity_calibration

python train.py --model 15 --gpus 1 --sample_frame_len 8 --sample_frame_num 4 --sample_frame_stride 4 --train_bs 8 --exp_name my_exp19
python test.py --model 15 --gpus 1 --sample_frame_len 8 --sample_frame_num 4 --sample_frame_stride 4 --test_bs 8 --exp_name my_exp19 \
               --test_ckpt my_exp19/epoch149.pth --similarity_calibration

python train.py --model 7 --gpus 1 --clip_model ViT-B-16 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp20
python test.py --model 7 --gpus 1 --clip_model ViT-B-16 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 16 --exp_name my_exp20 \
               --test_ckpt my_exp20/epoch149.pth --similarity_calibration

python train.py --model 7 --gpus 1 --clip_model ViT-B-32 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp21
python test.py --model 7 --gpus 1 --clip_model ViT-B-32 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 16 --exp_name my_exp21 \
               --test_ckpt my_exp21/epoch149.pth --similarity_calibration

python train.py --model 16 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp22
python test.py --model 16 --gpus 1 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 16 --exp_name my_exp22 \
               --test_ckpt my_exp22/epoch149.pth --similarity_calibration

python train.py --model 7 --gpus 1 --sample_frame_len 4 --num_layers 2 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp23
python test.py --model 7 --gpus 1 --sample_frame_len 4 --num_layers 2 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 16 --exp_name my_exp23 \
               --test_ckpt my_exp23/epoch149.pth --similarity_calibration

python train.py --model 7 --gpus 1 --sample_frame_len 4 --num_layers 3 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp24
python test.py --model 7 --gpus 1 --sample_frame_len 4 --num_layers 3 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 16 --exp_name my_exp24 \
               --test_ckpt my_exp24/epoch149.pth --similarity_calibration

python train.py --model 17 --gpus 1 --sample_frame_len 4 --num_layers 3 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp25
python test.py --model 17 --gpus 1 --sample_frame_len 4 --num_layers 3 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 16 --exp_name my_exp25 \
               --test_ckpt my_exp25/epoch149.pth --similarity_calibration

python train.py --model 18 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --train_bs 16 --exp_name my_exp26
python test.py --model 18 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 2 --test_bs 1 --exp_name my_exp26 \
               --test_ckpt my_exp26/epoch149.pth --similarity_calibration

python train.py --model 19 --gpus 0 --num_layers 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp27
python test.py --model 19 --gpus 0  --num_layers 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp27 \
               --test_ckpt my_exp27/epoch149.pth --similarity_calibration

python train.py --model 18 --gpus 2 --num_layers 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp28
python test.py --model 18 --gpus 2  --num_layers 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp28 \
               --test_ckpt my_exp28/epoch149.pth --similarity_calibration

python train.py --model 18 --gpus 2 --num_layers 3 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp29
python test.py --model 18 --gpus 2  --num_layers 3 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp29 \
               --test_ckpt my_exp29/epoch149.pth --similarity_calibration

python train.py --model 20 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp30
python test.py --model 20 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp30 \
               --test_ckpt my_exp30/epoch149.pth --similarity_calibration

python train.py --model 20 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp30
python test.py --model 20 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp30 \
               --test_ckpt my_exp30/epoch149.pth --similarity_calibration

python train.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp31
python test.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp31 \
               --test_ckpt my_exp31/epoch149.pth --similarity_calibration

python train.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp32
python test.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp32 \
               --test_ckpt my_exp32/epoch149.pth --similarity_calibration

python train.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp33
python test.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp33 \
               --test_ckpt my_exp33/epoch149.pth --similarity_calibration

python train.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp34
clear
python test.py --model 21 --gpus 0 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp34 \
               --test_ckpt my_exp34/epoch149.pth --similarity_calibration
cd TrackEval/scripts
sh evaluate.sh

python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --train_bs 16 --exp_name my_exp35
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --test_bs 16 --exp_name my_exp35 \
               --test_ckpt my_exp35/epoch149.pth --similarity_calibration
cd TrackEval/scripts

python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp35/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp35/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp35/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False


python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 2 --train_bs 16 --exp_name my_exp36
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 2 --test_bs 16 --exp_name my_exp36 \
               --test_ckpt my_exp36/epoch149.pth --similarity_calibration
cd TrackEval/scripts

python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp36/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp36/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp36/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 3 --train_bs 16 --exp_name my_exp37
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 3 --test_bs 16 --exp_name my_exp37 \
               --test_ckpt my_exp37/epoch149.pth --similarity_calibration
cd TrackEval/scripts

python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp37/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp37/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp37/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp38
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp38 \
               --test_ckpt my_exp38/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp38/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp38/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp38/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 5 --train_bs 16 --exp_name my_exp39
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 2 --sample_frame_stride 4 --num_layers 5 --test_bs 16 --exp_name my_exp39 \
               --test_ckpt my_exp39/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp39/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp39/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp39/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp40
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp40 \
               --test_ckpt my_exp40/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp40/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp40/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp40/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 1 --truncation 20 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp41
clear
python test.py --model 22 --gpus 2 --sample_frame_len 4 --sample_frame_num 1 --truncation 20 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp41 \
               --test_ckpt my_exp41/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp41/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp41/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp41/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp42
clear
python test.py --model 22 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp42 \
               --test_ckpt my_exp42/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp42/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp42/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp42/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 23 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp43
clear
python test.py --model 23 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp43 \
               --test_ckpt my_exp43/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp43/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp43/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp43/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 23 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp44
clear
python test.py --model 23 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp44 \
               --test_ckpt my_exp44/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp44/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp44/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp44/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 23 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp45
clear
python test.py --model 23 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp45 \
               --test_ckpt my_exp45/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp45/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp45/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp45/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 22 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp46
clear
python test.py --model 22 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp46 \
               --test_ckpt my_exp46/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp46/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp46/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp46/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp47
clear
python test.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp47 \
               --test_ckpt my_exp47/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp47/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp47/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp47/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp48
clear
python test.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp48 \
               --test_ckpt my_exp48/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp48/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp48/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp48/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp49
clear
python test.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp49 \
               --test_ckpt my_exp49/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp49/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp49/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp49/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp50
clear
python test.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp50 \
               --test_ckpt my_exp50/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp50/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp50/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp50/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp51
clear
python test.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp51 \
               --test_ckpt my_exp51/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp51/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp51/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp51/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp52
clear
python test.py --model 24 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp52 \
               --test_ckpt my_exp52/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp52/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp52/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp52/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --train_bs 16 --exp_name my_exp53
clear
python test.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 4 --test_bs 16 --exp_name my_exp53 \
               --test_ckpt my_exp53/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp53/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp53/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp53/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 5 --train_bs 16 --exp_name my_exp54
clear
python test.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 5 --test_bs 16 --exp_name my_exp54 \
               --test_ckpt my_exp54/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp54/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp54/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp54/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 6 --train_bs 16 --exp_name my_exp55
clear
python test.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 6 --test_bs 16 --exp_name my_exp55 \
               --test_ckpt my_exp55/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp55/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp55/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp55/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 6 --train_bs 16 --exp_name my_exp56
clear
python test.py --model 25 --gpus 1 --sample_frame_len 4 --sample_frame_num 1 --sample_frame_stride 4 --num_layers 6 --test_bs 16 --exp_name my_exp56 \
               --test_ckpt my_exp56/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp56/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp56/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp56/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 2 --sample_frame_num 1 --sample_frame_stride 2 --num_layers 5 --train_bs 16 --exp_name my_exp57
clear
python test.py --model 25 --gpus 1 --sample_frame_len 2 --sample_frame_num 1 --sample_frame_stride 2 --num_layers 5 --test_bs 16 --exp_name my_exp57 \
               --test_ckpt my_exp57/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp57/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp57/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp57/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

cd ../..

python train.py --model 25 --gpus 1 --sample_frame_len 2 --sample_frame_num 1 --sample_frame_stride 2 --num_layers 1 --train_bs 16 --exp_name my_exp58
clear
python test.py --model 25 --gpus 1 --sample_frame_len 2 --sample_frame_num 1 --sample_frame_stride 2 --num_layers 1 --test_bs 16 --exp_name my_exp58 \
               --test_ckpt my_exp58/epoch149.pth --similarity_calibration
cd TrackEval/scripts
python run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data/hpc/khanh/iKUN/plugins/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data/hpc/khanh/iKUN/plugins/Refer-KITTI/KITTI/training/image_02 \
--OUTPUT_FOLDER ./plugins/my_exp58/results \
--TRACKERS_FOLDER /data/hpc/khanh/iKUN/plugins/my_exp58/results \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL '/data/hpc/khanh/iKUN/plugins/my_exp58/results' \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

COMMENT