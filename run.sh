
### train
### horse2zebra cityscapes summer2winter
python train.py  \
--dataroot ./datasets/summer2winter \
--name summer2winter \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--direction BtoA \
--patch_size 64

python train.py  \
--dataroot ./datasets/cityscapes \
--name cityscapes \
--model sc \
--gpu_ids 1 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--direction BtoA \
--patch_size 64
### horse2zebra
python train.py  \
--dataroot ./datasets/horse2zebra \
--name horse2zebra-d4 \
--model sc \
--gpu_ids 1 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--patch_size 64

python train.py  \
--dataroot ./datasets/night2day \
--name night2day \
--model sc \
--gpu_ids 1 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--patch_size 64

python train.py  \
--dataroot ./datasets/cat2dog \
--name cat2dog1 \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--patch_size 64


### val for fid
python test_fid.py \
--dataroot ./datasets/horse2zebra \
--checkpoints_dir ./checkpoints \
--name horse2zebra \
--gpu_ids 0 \
--model sc \
--num_test 0

python test_fid.py \
--dataroot ./datasets/night2day \
--checkpoints_dir ./checkpoints \
--name night2day \
--gpu_ids 0 \
--model sc \
--num_test 0

python test_fid.py \
--dataroot ./datasets/cat2dog \
--checkpoints_dir ./checkpoints \
--name cat2dog \
--gpu_ids 0 \
--model sc \
--num_test 0


python test_fid.py \
--dataroot ./datasets/summer2winter \
--checkpoints_dir ./checkpoints \
--name summer2winter-d8 \
--gpu_ids 0 \
--model sc \
--direction BtoA \
--num_test 0

python test_fid.py \
--dataroot ./datasets/cityscapes \
--checkpoints_dir ./checkpoints \
--name cityscapes \
--gpu_ids 1 \
--model sc \
--direction BtoA \
--num_test 0


#### continue
python train.py  \
--dataroot ./datasets/horse2zebra \
--name horse2zebra \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--patch_size 32 \
--learned_attn \
--augment \
--continue_train \
--epoch_count 165

python train.py  \
--dataroot ./datasets/summer2winter \
--name summer2winter \
--model sc \
--gpu_ids 1 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--direction BtoA \
--patch_size 64 \
--continue_train \
--epoch_count 130

