import os

name = ['balloon1', 'balloon2', 'dynamicface', 'jumping', 'playground', 'skating', 'truck', 'umbrella']
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
for i in range(int(len(name))):
    os.system('python evaluation.py --config configs/config_nvidia/{}.txt'.format(str(name[i])))