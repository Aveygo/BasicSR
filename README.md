A fork of [BasicSR](https://github.com/XPixelGroup/BasicSR) to train SRNext.

## Dataset

### Data
Download and manually merge the HR images into a single folder at datasets/DF2K/DF2K_HR
1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3. OST: https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip

### Processing
Create metadata file for loading the images
```
python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR --root datasets/DF2K datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```

### Training
Example code for multi-gpu training (assumed 4 here)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/SRNext/train_srnext_x4_scratch.yml --launcher pytorch --auto_resume
```
Simple single GPU train
```
python realesrgan/train.py -opt options/train/SRNext/train_srnext_x4_scratch.yml --auto_resume
```