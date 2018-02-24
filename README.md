# Realtime_Multi-Person_Pose_Estimation.PyTorch
Pytorch implementation of Realtime_Multi-Person_Pose_Estimation

Train
1. prepare training data
   - Dowload COCO_train2014 val2014 from official website
   - cd ./preprocessing
   - configure generate_json_mask.py (ann_dir .... )
   - run
   
2. start training
   - cd ./experiments/baseline/
   - configure coco_loader (line 198 img_path)
   - configure train_pose.py (--train_dir )
   - run



Test and eval
1. test single image
   - ./evaluation/test_pose.py
2. evaluate caffemodel dowload from authur
   - ./evaluation/eval_caffe.py    53.8% (50 images)
3. evaluate pytorch model converted from caffemodel
   - ./preprocessing/convert_model.py
   - ./evaluation/eval_pytorch.py  54.4% (50 images) 54.1% (1000 images)
4. evaluate pytorch model trained by yourself
   - ./evaluation/eval_pytorch.py  
   
   
results
1. caffemodel evaluated by python scripts
   - 53.8% (50 images)
2. pytorch model converted from caffe  by python scripts
   - 54.4% (50 images) 54.1% (1000 images)
3. pytorch model trained on train2014
   - 45.9% (50 images)  60000 iters (stepsize = 50000)

