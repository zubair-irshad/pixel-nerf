import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
import sys
sys.path.append('/home/ubuntu/pixel-nerf')
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
# from .google_scanned_utils import load_image_from_exr, load_seg_from_exr
import struct
# from .ray_utils import *
# from .nocs_utils import rebalance_mask
import glob
from objectron.schema import annotation_data_pb2 as annotation_protocol
import glob
from random import choice
import random
import pickle

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    # if image_size > 0:
    #     ops.append(T.Resize(image_size))
    ops.extend(
        [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return T.Compose(ops)

def normalize_image():
    ops = []
    ops.extend(
        [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    # ops.extend(
    #     [T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])]
    # )
    return T.Compose(ops)

class ObjectronMultiDataset(Dataset):
    def __init__(self, root_dir, max_imgs = 150, stage='train', img_wh=(160, 120), num_instances_per_obj = 1, crop_img = False):
        self.base_dir = root_dir
        self.max_imgs = max_imgs
        self.split = stage
        self.ids = np.sort([f.name for f in os.scandir(self.base_dir)])
        self.crop_img = crop_img
        self.num_instances_per_obj = num_instances_per_obj
        self.lenids = len(self.ids)
        self.normalize_img = normalize_image()
        self.img_wh = img_wh
        self.define_transforms()
        # self.near = 0.02
        # self.far = 1.0
        self.z_near = 0.08
        self.z_far = 0.8
        self.val_instances = [30, 60, 90, 120, 145]
        # self.val_instances = [10, 20, 30, 40, 45]
        self.white_back = False
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.lindisp = False

    def load_img(self, instance_dir, idxs = [], ids=None, bbox_2d = None):
        masks_dir = os.path.join(instance_dir, 'masks_12/*.png')
        allmaskfiles = sorted(glob.glob(masks_dir))
        maskfiles = np.array(allmaskfiles)[idxs]
        all_imgs =[]
        for seg_name in maskfiles:
            img_name = os.path.basename(str(seg_name)).split('_')[1]
            img = cv2.imread(os.path.join(instance_dir, 'images_12', img_name))
            focal_idx = img_name.split('.')[0]
            img = img[...,::-1]
            if self.crop_img:
                x_min, y_min, x_max, y_max = bbox_2d
                if y_min<0:
                    y_min=0
                if x_min<0:
                    x_min=0
                img = img[y_min: y_max, x_min:x_max]
            
            W,H = img.shape[0], img.shape[1]
            new_img_wh = (W,H)
            img = Image.fromarray(img)
            img = img.transpose(Image.ROTATE_90)
            img = self.image_to_tensor(img)
            # img = self.transform(img) # (h, w, 3)

            #Get masks
            seg_mask = cv2.imread(os.path.join(instance_dir, 'masks_12', os.path.basename(seg_name)), cv2.IMREAD_GRAYSCALE)
            if self.crop_img:
                x_min, y_min, x_max, y_max = bbox_2d
                if y_min<0:
                    y_min=0
                if x_min<0:
                    x_min=0
                seg_mask = seg_mask[y_min: y_max, x_min:x_max]
            seg_mask = np.rot90(np.array(seg_mask), axes=(0,1))

            # valid_mask = seg_mask>0
            # valid_mask = self.transform(valid_mask).view(
            #         -1)
            instance_mask = seg_mask >0
            instance_mask = self.transform(instance_mask)

            #img = img.contiguous().view(3, -1).permute(1, 0) # (h*w, 3) RGBA
        if len(idxs) == 1:
            return img, instance_mask, new_img_wh, int(focal_idx)
        else:
            return all_imgs, new_img_wh, int(focal_idx)


    def return_train_data(self, instance_name, idx, instances):
        instance_dir = os.path.join(self.base_dir, instance_name)
        meta_data_filename = instance_dir + '/'+ instance_name+'_metadata.pickle'
        with open(meta_data_filename, 'rb') as handle:
            meta_data = pickle.load(handle)
            
        all_c2w = meta_data["poses"]
        all_focal = meta_data['focal'] 
        all_c = meta_data['c'] 
        axis_align_mat = meta_data['RT'] 
        scale = meta_data['scale'] 
        
        self.axis_align_mat = torch.FloatTensor(np.linalg.inv(axis_align_mat))
        #save relevant bonding box info with obj ids for inference
        RTs_dict = {'RT':axis_align_mat, 'scale': scale}
                    
        img, instance_mask, new_img_wh, focal_idx = self.load_img(instance_dir, [instances], idx)
        w, h = self.img_wh
        if self.crop_img:
            w, h = new_img_wh
            # w -= (2*32)
            # h -=  (2*32)

        c2w = np.array(all_c2w)[focal_idx]
        focal = all_focal[focal_idx]
        c = all_c[focal_idx]
        #incase of just one instance
        c2w = np.squeeze(c2w)
        return img,instance_mask, c2w, focal, c

    def return_val_data(self, instance_name, idx, instances):
        instance_dir = os.path.join(self.base_dir, instance_name)
        meta_data_filename = instance_dir + '/'+ instance_name+'_metadata.pickle'
        with open(meta_data_filename, 'rb') as handle:
            meta_data = pickle.load(handle)
            
        all_c2w = meta_data["poses"]
        all_focal = meta_data['focal'] 
        all_c = meta_data['c'] 
        axis_align_mat = meta_data['RT'] 
        scale = meta_data['scale'] 
        
        self.axis_align_mat = torch.FloatTensor(np.linalg.inv(axis_align_mat))
        #save relevant bonding box info with obj ids for inference
        RTs_dict = {'RT':axis_align_mat, 'scale': scale}
                    
        img, instance_mask, new_img_wh, focal_idx = self.load_img(instance_dir, [instances], idx)
        w, h = self.img_wh
        if self.crop_img:
            w, h = new_img_wh
            # w -= (2*32)
            # h -=  (2*32)

        c2w = np.array(all_c2w)[focal_idx]
        focal = all_focal[focal_idx]
        c = all_c[focal_idx]
        #incase of just one instance
        c2w = np.squeeze(c2w)
        return img,instance_mask, c2w, focal, c

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return self.lenids

    def __getitem__(self, idx):
        obj_id = self.ids[idx]
        if self.split == "train":
            numbers = range(0,self.max_imgs)
            all_imgs = []
            all_poses = []
            all_masks = []
            # all_focal = []
            # all_c = []
            train_indices = [n for n in numbers if n not in self.val_instances]
            random.shuffle(train_indices)
            for j, index in enumerate(train_indices):
                img, instance_mask, c2w, focal, c = self.return_train_data(obj_id, idx, index)
                all_imgs.append(img)
                all_masks.append(instance_mask)
                all_poses.append(torch.tensor(c2w, dtype=torch.float32))
                if j ==0:
                    all_focal = focal
                    all_c = c
                # all_focal.append(focal)
                # all_c.append(c)
            all_imgs = torch.stack(all_imgs)
            all_masks = torch.stack(all_masks)
            all_poses = torch.stack(all_poses)
            # all_focal = torch.stack(all_focal)
            # all_c = torch.stack(all_c)
            result = {
                "path": self.base_dir,
                "img_id": idx,
                "focal": all_focal,
                "images": all_imgs,
                "masks": all_masks,
                "poses": all_poses,
                "c": all_c
            }
            return result
        else:
            # numbers = self.val_instances
            numbers = range(0,self.max_imgs)
            all_imgs = []
            all_poses = []
            all_masks = []
            # all_focal = []
            # all_c = []
            val_indices = numbers
            # random.shuffle(val_indices)
            for j, index in enumerate(val_indices):
                img, instance_mask, c2w, focal, c = self.return_train_data(obj_id, idx, index)
                all_imgs.append(img)
                all_masks.append(instance_mask)
                all_poses.append(torch.tensor(c2w, dtype=torch.float32))
                if j ==0:
                    all_focal = focal
                    all_c = c
                # all_focal.append(focal)
                # all_c.append(c)
            all_imgs = torch.stack(all_imgs)
            all_masks = torch.stack(all_masks)
            all_poses = torch.stack(all_poses)
            # all_focal = torch.stack(all_focal)
            # all_c = torch.stack(all_c)
            result = {
                "path": self.base_dir,
                "img_id": idx,
                "focal": all_focal,
                "images": all_imgs,
                "masks": all_masks,
                "poses": all_poses,
                "c": all_c
            }
            return result
            