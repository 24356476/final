#python train_condition.py --cuda True --gpu_ids 0 --keep_step 5 --save_count 1 --tensorboard_count 1 --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion --batch-size 4

python train_condition.py --cuda True --gpu_ids 0 --keep_step 1500 --save_count 100 --tensorboard_count 100 --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion --batch-size 4

#python train_condition.py --cuda True --gpu_ids 0 --keep_step 2000 --save_count 100 --tensorboard_count 100 --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion --batch-size 4 --tocg_checkpoint ./checkpoints/test/tocg_final.pth --D_checkpoint ./checkpoints/test/D_final.pth --load_step 1500

#python train_condition.py --cuda True --gpu_ids 0 --shuffle --keep_step 1000 --save_count 100 --tensorboard_count 500 --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion --batch-size 4 --tocg_checkpoint ./checkpoints/test/tocg_final.pth --D_checkpoint ./checkpoints/test/D_final.pth --load_step 900




#python test_condition.py --gpu_ids 0 --shuffle --tocg_checkpoint 'checkpoints/test/tocg_final.pth' --D_checkpoint 'checkpoints/test/D_final.pth' --batch-size 4 --datasetting 'paired' --semantic_nc 13 --output_nc 13 --occlusion --warp_feature 'T1' --upsample 'bilinear' --norm_const 1.0

python test_condition.py --gpu_ids 0 --tocg_checkpoint 'checkpoints/test/tocg_final.pth' --D_checkpoint 'checkpoints/test/D_final.pth' --batch-size 4 --datasetting 'paired' --semantic_nc 13 --output_nc 13 --occlusion --warp_feature 'T1' --upsample 'bilinear' --norm_const 1.0

python test_condition.py --gpu_ids 0 --shuffle --tocg_checkpoint 'checkpoints/test/tocg_final.pth' --D_checkpoint 'checkpoints/test/D_final.pth' --batch-size 4 --datasetting 'paired' --semantic_nc 13 --output_nc 13 --occlusion --warp_feature 'T1' --upsample 'bilinear' --norm_const 1.0




$env:KMP_DUPLICATE_LIB_OK="TRUE"

$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


#python train_generator.py --name "train_generator" --keep_step 100 --save_count 100 --tensorboard_count 100 --batch_size 1 --tocg_checkpoint 'checkpoints/test/tocg_final.pth' --dis_checkpoint 'checkpoints/test/D_final.pth' --cuda True --gpu_ids 0 --fp16 --occlusion

#python train_generator.py --cuda True --name test --keep_step 100 --save_count 50 --batch_size 1 --workers 4 --gpu_ids 0 --fp16 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion

#python train_generator.py --cuda True --name test --keep_step 500 --decay_step 500 --save_count 100 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion

#python train_generator.py --cuda True --name test --keep_step 1000 --decay_step 1000 --save_count 200 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion --gen_checkpoint checkpoints/test/gen_model_final.pth --dis_checkpoint checkpoints/test/dis_model_final.pth --load_step 1000

#python train_generator.py --cuda True --name "train_generator" --keep_step 1000 --decay_step 1000 --save_count 100 --tensorboard_count 100 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion


python train_generator.py --cuda True --name "train_generator" --keep_step 2500 --decay_step 2500 --save_count 500 --tensorboard_count 100 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion

python train_generator.py --cuda True --name "train_generator" --keep_step 5000 --decay_step 5000 --save_count 500 --tensorboard_count 100 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion --gen_checkpoint checkpoints/train_generator/gen_model_final.pth --dis_checkpoint checkpoints/train_generator/dis_model_final.pth --load_step 5000 --shuffle


python train_generator.py --cuda True --name "train_generator" --lambda_vgg 25 --keep_step 7500 --decay_step 7500 --save_count 500 --tensorboard_count 100 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion --gen_checkpoint checkpoints/train_generator/gen_model_final.pth --dis_checkpoint checkpoints/train_generator/dis_model_final.pth --load_step 10000 #--shuffle

python train_generator.py --cuda True --name "train_generator" --lambda_vgg 20 --keep_step 10000 --decay_step 10000 --save_count 500 --tensorboard_count 100 --batch_size 1 --workers 4 --gpu_ids 0 --tocg_checkpoint checkpoints/test/tocg_final.pth --occlusion --gen_checkpoint checkpoints/train_generator/gen_model_final.pth --dis_checkpoint checkpoints/train_generator/dis_model_final.pth --load_step 15000 #--shuffle





#tensorboard --logdir=tensorboard --port=6006

#tensorboard --logdir=tensorboard --port=6006 --load_fast=false

#tensorboard --logdir=tensorboard --port=6006 --load_fast=false --samples_per_plugin scalars=0

#tensorboard --logdir=tensorboard --port=6006 --load_fast=false --bind_all --samples_per_plugin=images=0,scalars=0,graphs=0,distributions=0,histograms=0,hparams=0

#tensorboard --logdir=tensorboard --port=6006 --load_fast=false --samples_per_plugin=images=0,scalars=0,graphs=0,distributions=0,histograms=0,hparams=0

tensorboard --logdir=tensorboard --host=127.0.0.1 --port=6006



#python test_generator.py --cuda True --gpu_ids 0 --batch-size 4 --tocg_checkpoint ./eval_models/weights/v0.1/tocg_final.pth --gen_checkpoint ./eval_models/weights/v0.1/gen.pth --test_name "test_run"

#python test_generator.py --cuda True --gpu_ids 0 --batch-size 4 --tocg_checkpoint ./eval_models/weights/v0.1/tocg_final.pth --gen_checkpoint ./eval_models/weights/v0.1/gen.pth --test_name "test_run" --shuffle

#python test_generator.py --cuda True --gpu_ids 0 --batch-size 1 --tocg_checkpoint ./eval_models/weights/v0.1/tocg_final.pth --gen_checkpoint ./eval_models/weights/v0.1/gen.pth --test_name "test_run" --shuffle

python test_generator.py --cuda True --gpu_ids 0 --batch-size 1 --gen_checkpoint ./eval_models/weights/v0.1/gen_model_final.pth --tocg_checkpoint ./eval_models/weights/v0.1/tocg_final.pth --test_name "test_run"

python test_generator.py --cuda True --gpu_ids 0 --shuffle --batch-size 1 --gen_checkpoint ./eval_models/weights/v0.1/gen_model_final.pth --tocg_checkpoint ./eval_models/weights/v0.1/tocg_final.pth --test_name "test_run"




#with pretrained condition generator
#python test_generator.py --cuda True --gpu_ids 0 --batch-size 1 --gen_checkpoint ./eval_models/weights/v0.1/gen.pth --test_name "test_run" --shuffle

