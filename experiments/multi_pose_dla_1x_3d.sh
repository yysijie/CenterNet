cd src
# test 3d model transformed from 2d official model (identically)
python test.py multi_pose --exp_id dla3d_i_1x --dataset coco_hp --keep_res --load_model ../models/multi_pose_dla_1x.dla3d_i.pth --gpus 4 --arch dla3di_34 --model_to3d identical
# test 3d model transformed from 2d official model (mean)
python test.py multi_pose --exp_id dla3d_i_1x --dataset coco_hp --keep_res --load_model ../models/multi_pose_dla_1x.dla3d_i.pth --gpus 4 --arch dla3di_34 --model_to3d mean
# train
python main.py multi_pose --exp_id dla3d_i_1x --dataset coco_hp --batch_size 2 --master_batch 1 --lr 5e-4 --load_model ../models/multi_pose_dla_1x.pth --arch dla3di_34 --gpus 6,7 --num_workers 16 --model_to3d identical



# train 3d model from 2d official model
python main.py multi_pose --exp_id dla3d_1x --arch dla3d_34 --dataset coco_hp \
 --lr 5e-4 --load_model ../models/multi_pose_dla_3x.pth  --num_workers 16 --model_to3d mean \
 --batch_size 16 --master_batch 2 --gpus 0,1,2,3,4,5,6,7 --num_epochs 20 --lr_step 5,15

cd ..