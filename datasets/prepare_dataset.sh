### cat2dog
bash download.sh afhq-dataset

mkdir cat2dog
cd cat2dog
ln -s ../afhq/train/cat/ trainA
ln -s ../afhq/train/dog/ trainB
ln -s ../afhq/val/cat/ testA
ln -s ../afhq/val/dog/ testB

### cityscapes
python prepare_cityscapes_dataset.py --gtFine_dir [path_to_afhq]/gtFine/ --leftImg8bit_dir [path_to_afhq]/leftImg8bit --output_dir ./cityscapes/
python make_dataset_aligned.py --dataset-path ./cityscapes

### night2day
bash download_pix2pix_dataset.sh night2day
python make_dataset_aligned.py --dataset-path ./night2day
### summer2winter_yosemite horse2zebra monet2photo
bash ./datasets/download_cyclegan_dataset.sh monet2photo horse2zebra/summer2winter_yosemite

