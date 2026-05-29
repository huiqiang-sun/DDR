import os
import torchvision
import glob
import cv2
import torch
import skimage
import numpy as np
import argparse
import skimage.morphology

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_maskrcnn(model, img_path): #, intWidth=1024, intHeight=576):
    import PIL
    threshold = 0.5

    o_image = PIL.Image.open(img_path)

    width, height = o_image.size

    if width > height:
        intWidth = 960
        intHeight = int(round( float(intWidth) / width * height))        
    else:
        intHeight = 960
        intWidth = int(round( float(intHeight) / height * width))        

    print('Semantic Seg Width %d Height %d'%(intWidth, intHeight))

    image = o_image.resize((intWidth, intHeight), PIL.Image.ANTIALIAS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions['masks'].size(0)):
        if objPredictions['scores'][intMask].item() > threshold:
            for i in range(200):
                if objPredictions['labels'][intMask].item() == i: # person
                    tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0
                    break

            # if objPredictions['labels'][intMask].item() == 1: # person
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 4: # motorcycle
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 2: # bicycle
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 8: # truck
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 28: # umbrella
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 17: # cat
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 18: # dog
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 36: # snowboard
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0
                
            # if objPredictions['labels'][intMask].item() == 37: # sports_ball
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            # if objPredictions['labels'][intMask].item() == 41: # skateboard
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0
            
            # if objPredictions['labels'][intMask].item() == 3: # car
            #     tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(tenHumans.cpu().numpy(),
                                         skimage.morphology.disk(1))
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask



def maskrcnn(basedir):

    img_dir = glob.glob(basedir + '/images_*x*')[0] 
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png')))
    semantic_mask_dir = os.path.join(basedir, 'motion_masks')
    netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    os.makedirs(semantic_mask_dir, exist_ok=True)

    for i in range(0, len(img_path_list)):
        img_path = img_path_list[i]
        img_name = img_path.split('/')[-1]

        shape = cv2.imread(img_path).shape
        h, w = shape[0], shape[1]

        semantic_mask = run_maskrcnn(netMaskrcnn, 
                                     img_path)
        semantic_mask = cv2.resize(semantic_mask, (w, h), 
                                   interpolation=cv2.INTER_NEAREST)
        semantic_mask = skimage.morphology.dilation(semantic_mask, skimage.morphology.disk(8))
        semantic_mask = 255 - semantic_mask
        cv2.imwrite(os.path.join(semantic_mask_dir, 
                                img_name + '.png'), semantic_mask)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, 
                        help='COLMAP Directory')

    args = parser.parse_args()

    maskrcnn(args.data_path)