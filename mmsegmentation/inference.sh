python inference.py \
--work_dir '/opt/ml/input/mmsegmentation/work_dirs/HRNet' \
--model_config_path '/opt/ml/input/mmsegmentation/work_dirs/HRNet/HRNet.py' \
--model_ckpt_path '/opt/ml/input/mmsegmentation/work_dirs/HRNet/best_mIoU638_epoch_94.pth' \
--test_imgfile_path '/opt/ml/input/data/images/test' \
--output_file_name 'HRNet_67Ep'

# output_file은 work_Dir에 저장됨