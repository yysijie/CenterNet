cd src
# train
# python main.py multi_pose --exp_id res --dataset coco_hp --batch_size 128 --master_batch 9 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0,1,2,3,4,5,6,7 --num_workers 16
python main.py multi_pose --exp_id resdcn18 --dataset coco_hp --arch resdcn_18 --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16

