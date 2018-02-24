export PYTHONUNBUFFERED="True"
LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
# python train_pose.py --gpu 0 1 --train_dir /home/code/panhongyu/datasets/coco/filelist/traincoco_pytorch.txt /home/code/panhongyu/datasets/coco/masklist/traincoco_pytorch.txt /home/code/panhongyu/datasets/coco/json/traincoco_pytorch.json --val_dir /home/code/panhongyu/datasets/coco/filelist/valcoco_pytorch.txt /home/code/panhongyu/datasets/coco/masklist/valcoco_pytorch.txt /home/code/panhongyu/datasets/coco/json/valcoco_pytorch.json --config config.yml > $LOG
# --pretrained /data/xiaobing.wang/qy.feng/Pytorch_RMPE/training/openpose_coco_latest.pth.tar 


python train_pose.py --gpu 0 --train_dir /data/root/data/samsung_pose/datasets/img_list/valminusminival2014.txt /data/root/data/samsung_pose/datasets/mask_list/valminusminival2014.txt /data/root/data/samsung_pose/datasets/json/valminusminival2014.json --val_dir /data/root/data/samsung_pose/datasets/img_list/minival2014.txt /data/root/data/samsung_pose/datasets/mask_list/minival2014.txt /data/root/data/samsung_pose/datasets/json/minival2014.json --config config.yml > $LOG
