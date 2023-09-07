import pickle
import torch
import numpy as np
import cv2
import os
import h5py
import rawpy
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .unprocess import *
from .process import *
from utils import *

class RealBase_Dataset(Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        # self.initialization() # 基类不调用
    
    def default_args(self):
        self.args = {}
        self.args['crop_per_image'] = 8
        self.args['crop_size'] = 512
        self.args['ori'] = False
        self.args['dstname'] = 'SID'
        self.args['camera_type'] = 'SonyA7S2'
        self.args['mode'] = 'train'
        self.args['croptype'] = 'non-overlapped'
        self.args['command'] = ''
        self.args['wp'] = 16383
        self.args['bl'] = 512
        self.args['clip'] = False

    def initialization(self):
        # 获取数据地址
        self.suffix = 'ARW'
        self.dataset_file = f'SID_{self.args["mode"]}.info'
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos = pkl.load(info_file)
            print(f'>> Successfully load "{self.dataset_file}" (Length: {len(self.infos)})')
        self.length = len(self.infos)
        self.get_shape()
        self.darkshading = {}
        self.noiseparam = {}
        self.blc_mean = {}
        self.naive = False if '++' in self.args['command'] else True
        for i in range(self.length):
            iso = self.infos[i]['ISO']
            # read darkshaidng
            if 'darkshading' in self.args['command']:
                self.get_darkshading(iso, naive=self.naive)
                if 'darkshading2' in self.args['command'] and iso not in self.noiseparam:
                    get_camera_noisy_params_max(f'SonyA7S2_{iso}')
                    self.noiseparam[iso] = get_camera_noisy_params_max(f'SonyA7S2_{iso}')
                    self.noiseparam[iso]['Kmax'] = 0.0009563*iso

    def lr_idremap_table_init(self):
        self.lr_idremap_table = [None] * self.length
        for idx in range(len(self.infos)):
            self.get_lr_id(idx)
        log('Successfully finish id_remap')

    def get_lr_id(self, idx):
        if 'idremap' in self.args['command']:
            # 没有的话就构建remap table
            if self.lr_idremap_table[idx] is None:
                ratio_dict = {}
                for i, ratio in enumerate(self.infos[idx]['ratio']):
                    if ratio not in ratio_dict:
                        ratio_dict[ratio] = [i]
                    else:
                        ratio_dict[ratio].append(i)
                self.lr_idremap_table[idx] = []
                for ratio in ratio_dict:
                    self.lr_idremap_table[idx].append(ratio_dict[ratio])
            
            # 选择100, 250, 300
            ratio_id = np.random.randint(len(self.lr_idremap_table[idx]))
            id = np.random.randint(len(self.lr_idremap_table[idx][ratio_id]))
            lr_id = self.lr_idremap_table[idx][ratio_id][id]
        else:
            lr_id = np.random.randint(len(self.infos[idx]['ratio']))
        return lr_id

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4

    def init_random_crop_point(self, mode='non-overlapped', raw_crop=False):
        self.h_start = []
        self.w_start = []
        self.h_end = []
        self.w_end = []
        # 这里比paired-data多了额外的旋转选项，因为行噪声是有方向的
        self.aug = np.random.randint(4, size=self.args['crop_per_image'])
        h, w = self.h, self.w
        if raw_crop:
            h, w = self.H, self.W
        if mode == 'non-overlapped':
            nh = h // self.args["patch_size"]
            nw = w // self.args["patch_size"]
            h_start = np.random.randint(0, h - nh*self.args["patch_size"] + 1)
            w_start = np.random.randint(0, w - nw*self.args["patch_size"] + 1)
            for i in range(nh):
                for j in range(nw):
                    self.h_start.append(h_start + i * self.args["patch_size"])
                    self.w_start.append(w_start + j * self.args["patch_size"])
                    self.h_end.append(h_start + (i+1) * self.args["patch_size"])
                    self.w_end.append(w_start + (j+1) * self.args["patch_size"])

        else: # random_crop
            for i in range(self.args['crop_per_image']):
                h_start = np.random.randint(0, h - self.args["patch_size"] + 1)
                w_start = np.random.randint(0, w - self.args["patch_size"] + 1)
                self.h_start.append(h_start)
                self.w_start.append(w_start)
                self.h_end.append(h_start + self.args["patch_size"])
                self.w_end.append(w_start + self.args["patch_size"])
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 2
        flip = mode // 2
        if rot:
            data = np.rot90(data, k=2, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data

    def eval_crop(self, data, base=64):
        crop_size = self.args["patch_size"]
        # crop setting
        d = base//2
        l = crop_size - base
        nh = self.h // l + 1
        nw = self.w // l + 1
        data = F.pad(data, (d, d, d, d), mode='reflect')
        croped_data = torch.empty((nh, nw, self.c, crop_size, crop_size),
                    dtype=data.dtype, device=data.device)
        # 分块crop主体区域
        for i in range(nh-1):
            for j in range(nw-1):
                croped_data[i][j] = data[..., i*l:i*l+crop_size,j*l:j*l+crop_size]
        # 补边
        for i in range(nh-1):
            j = nw - 1
            croped_data[i][j] = data[..., i*l:i*l+crop_size,-crop_size:]
        for j in range(nw-1):
            i = nh - 1
            croped_data[i][j] = data[..., -crop_size:,j*l:j*l+crop_size]
        # 补角
        croped_data[nh-1][nw-1] = data[..., -crop_size:,-crop_size:]
        # 整合为tensor
        croped_data = croped_data.view(-1, self.c, crop_size, crop_size)
        return croped_data
    
    def eval_merge(self, croped_data, base=64):
        crop_size = self.args["patch_size"]
        data = torch.empty((1, self.c, self.h, self.w), dtype=croped_data.dtype, device=croped_data.device)
        # crop setting
        d = base//2
        l = crop_size - base
        nh = self.h // l + 1
        nw = self.w // l + 1
        croped_data = croped_data.view(nh, nw, self.c, crop_size, crop_size)
        # 分块crop主体区域
        for i in range(nh-1):
            for j in range(nw-1):
                data[..., i*l:i*l+l,j*l:j*l+l] = croped_data[i, j, :, d:-d, d:-d]
        # 补边
        for i in range(nh-1):
            j = nw - 1
            data[..., i*l:i*l+l, -l:] = croped_data[i, j, :, d:-d, d:-d]
        for j in range(nw-1):
            i = nh - 1
            data[..., -l:, j*l:j*l+l] = croped_data[i, j, :, d:-d, d:-d]
        # 补角
        data[..., -l:, -l:] = croped_data[nh-1, nw-1, :, d:-d, d:-d]
        
        return data

    # 因为视频随机crop位置是固定的，所以同一个视频的任意分量都可以调用random_crop函数
    def random_crop(self, img):
        # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
        c, h, w = img.shape
        # 创建空numpy做画布, [crops, h, w]
        crops = np.empty((self.args["crop_per_image"], c, self.args["patch_size"], self.args["patch_size"]), dtype=np.float32)
        # 往空tensor的通道上贴patchs
        for i in range(self.args["crop_per_image"]):
            crop = img[:, self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i]]
            crop = self.data_aug(crop, mode=self.aug[i])
            crops[i] = crop

        return crops

    def get_BLE_long(self, iso, exp=10000, naive=True):
        if naive:
            return self.blc_mean[iso]
        else:
            branch = '_highISO' if iso>1600 else '_lowISO'
            kt = np.poly1d(self.blc_mean[f'kt{branch}_long'])
            # kt = self.blc_mean[iso]['k']
            BLE = kt(iso) * exp
            return BLE

    def get_darkshading(self, iso, exp=25, naive=True, num=None, remake=False):
        branch = '_highISO' if iso>1600 else '_lowISO'
        if iso not in self.darkshading or remake is True:
            ds_path = os.path.join(self.args['ds_dir'], f'darkshading-iso-{iso}.npy')
            ds_k = np.load(os.path.join(self.args['ds_dir'], f'darkshading{branch}_k.npy'))
            ds_b = np.load(os.path.join(self.args['ds_dir'], f'darkshading{branch}_b.npy'))
            if naive: # naive linear darkshading - BLE(ISO)
                with open(os.path.join(self.args['ds_dir'], f'darkshading_BLE.pkl'), 'rb') as f:
                    self.blc_mean = pkl.load(f)
                BLE = self.blc_mean[iso]
            else: # new linear darkshading - BLE(ISO, t)
                with open(os.path.join(self.args['ds_dir'], f'BLE_t.pkl'),'rb') as f:
                    self.blc_mean = pickle.load(f)
                # BLE bias only here
                # kt = np.poly1d(self.blc_mean[f'kt{branch}'])
                BLE = self.blc_mean[iso]['b'] # + kt(iso) * exp
            # D_{ds} = D_{FPNk} + D_{FPNb} + BLE(ISO, t)
            self.darkshading[iso] = ds_k * iso + ds_b + BLE

        if naive:
            return self.darkshading[iso]
        else:
            kt = np.poly1d(self.blc_mean[f'kt{branch}'])
            BLE = kt(iso) * exp
            return self.darkshading[iso] + BLE
    
    def hot_check(self, name_id):
        hot_ids = [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 
        143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 
        177, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 
        199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 
        217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 230, 231, 232]

        hot = True if int(name_id[2:5]) in hot_ids else False
        return hot
    
    def record_bias_frames(self):
        log('Recording Bias Frames...')
        self.black_dirs = sorted(os.listdir(self.args['bias_dir']), key=lambda x: int(x))
        self.legalISO = np.array([int(dirname) for dirname in self.black_dirs])
        self.black_dirs = [os.path.join(self.args['bias_dir'], dirname) for dirname in self.black_dirs]
        self.blacks = [None] * len(self.legalISO)
        for i in range(len(self.legalISO)):
            self.blacks[i] = [os.path.join(self.black_dirs[i], filename) for filename in os.listdir(self.black_dirs[i])]
    
    def remap_darkshading(self, num=None):
        log(f'Using {num} to remap dark shading')
        for i in range(self.length):
            iso = self.infos[i]['ISO']
            self.get_darkshading(iso, num, remake=True)
    
    def remap_BLE(self, num=None):
        log(f'Using {num} to remap BLE')
        with open(os.path.join(f'/data/SonyA7S2/ds_ablation/darkshading_BLE_{num}.pkl'), 'rb') as f:
            self.blc_mean = pkl.load(f)
        for i in range(self.length):
            iso = self.infos[i]['ISO']
            self.get_darkshading(iso, remake=True)

class SID_Dataset(RealBase_Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
        if self.args['mode'] == 'train':
            # 只使用有限范围的ISO
            if 'limitediso' in self.args['command'].lower():
                new_infos = []
                lower = 400
                upper = 6400
                for i in range(self.length):
                    if lower <= self.infos[i]['ISO'] <= upper:
                        new_infos.append(self.infos[i])
                self.infos = new_infos
            # 只使用ELD验证集会出现的ISO
            elif 'exactiso' in self.args['command'].lower():
                new_infos = []
                exactiso_list = [800, 1600, 3200]
                for i in range(self.length):
                    if self.infos[i]['ISO'] in exactiso_list:
                        new_infos.append(self.infos[i])
                self.infos = new_infos
            self.length = len(self.infos)
            self.lr_idremap_table_init()
        else:
            self.evaltest_remap()
            self.change_eval_ratio(ratio=250)
            self.length = len(self.infos)

    def evaltest_remap(self):
        self.infos_all = [self.infos[:40], self.infos[40:80], self.infos[80:]]
        # 镀一层包装，就不用getitem改代码了
        for rid in range(3):
            for i in range(len(self.infos_all[rid])):
                self.infos_all[rid][i]['short'] = [self.infos_all[rid][i]['short']]
                self.infos_all[rid][i]['ratio'] = [self.infos_all[rid][i]['ratio']]

    def change_eval_ratio(self, idx=None, ratio=None):
        ratio_list = [100, 250, 300]
        assert idx is not None or ratio is not None, 'Check please!'
        if idx is not None:
            assert idx in [0,1,2], 'idx must in [0,1,2]'
            ratio = ratio_list[idx]
        elif ratio is not None:
            assert int(ratio) in ratio_list, 'ratio must in [100,250,300]'
            idx = int(ratio) // 100 - 1

        self.infos = self.infos_all[idx]
        self.length = len(self.infos)
        log(f'Eval ratio {ratio}')

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = f"{self.infos[idx]['name'][:5]}_{self.infos[idx]['ratio']}"
        data['ISO'] = self.infos[idx]['ISO']
        data['ExposureTime'] = self.infos[idx]['ExposureTime'] * 1000
        
        hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        lr_id = self.get_lr_id(idx) if self.args['mode']=='train' else 0
        lr_raw = np.array(dataload(self.infos[idx]['short'][lr_id])).reshape(self.H,self.W)
        data['ratio'] = self.infos[idx]['ratio'][lr_id]
        data['exp'] = data['ExposureTime'] / data['ratio']

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=data['ISO'], exp=data['exp'], naive=self.naive)
            if 'd' in self.args['noise_code']:
                lr_raw = lr_raw + self.get_darkshading(iso=data['ISO'], exp=data['exp'], naive=self.naive).mean()
            if 'darkshading2' in self.args['command'] and self.args["mode"] == 'train':
                lr_raw += np.random.randn() * self.noiseparam[data['ISO']]['biassig']
        elif 'blc' in self.args['command'] and 'HB' not in self.args['command']:
            # 只使用均值矫正
            lr_raw = lr_raw - self.blc_mean[data['ISO']]

        # 数据转换
        lr_imgs = raw2bayer(lr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=False)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'])
            hr_crops = self.random_crop(hr_imgs)
            lr_crops = self.random_crop(lr_imgs)
        else:
            hr_crops = hr_imgs[None,:]
            lr_crops = lr_imgs[None,:]

        if self.args['ori'] is False:
            lr_crops *= data['ratio']

        if self.args['clip']:
            # -100 -> -inf
            lb = -100 if 'HB' in self.args['command'] else 0
            lr_crops = lr_crops.clip(lb, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

class Mix_Dataset(SID_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
        self.lr_idremap_table_init()
        print(f'Datasets Command:\t{self.args["command"]}')
        if 'aug' not in self.args['command']:
            log('Warning! You have not choose the version of SignalAug! Use default(v3) version...')
            raise NotImplementedError
            # self.args['command'] += 'aug_v3'
        self.record_bias_frames()
        self.HBR = HighBitRecovery(camera_type=self.args['camera_type'], noise_code=self.args['noise_code'])
        if 'darkshading' in self.args['command']:
            # darkshaidng会矫正输入，不能算减去的分布
            self.HBR.get_lut(self.legalISO, blc_mean=None)
        elif 'blc' in self.args['command']:
            # 只使用均值矫正
            self.HBR.get_lut(self.legalISO, self.blc_mean)
        else:
            # 假装不知道有bias这回事
            self.HBR.get_lut(self.legalISO, blc_mean=None)

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']
        data['ISO'] = self.infos[idx]['ISO']
        data['ExposureTime'] = self.infos[idx]['ExposureTime'] * 1000
        
        hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        data['black_lr'] = True if 'HB' in self.args['command'] and not np.random.randint(4) else False
        if data['black_lr']:
            # SNA+黑图
            iso_index = np.argmin(np.abs(self.legalISO-data['ISO']))
            lr_id = np.random.randint(len(self.blacks[iso_index])) if self.args['mode']=='train' else 0
            if 'lr10' in self.args['command']:
                lr_id = np.random.randint(10)
            lr_raw = rawpy.imread(self.blacks[iso_index][lr_id]).raw_image_visible
            dgain = 400
        else:
            lr_id = np.random.randint(len(self.infos[idx]['short'])) if self.args['mode']=='train' else 0
            lr_raw = np.array(dataload(self.infos[idx]['short'][lr_id])).reshape(self.H,self.W)
            dgain = self.infos[idx]['ratio'][lr_id]
        
        data['exp'] = data['ExposureTime'] / dgain

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=data['ISO'], exp=data['exp'], naive=self.naive)
            if 'darkshading2' in self.args['command'] and self.args["mode"] == 'train':
                lr_raw += np.random.randn() * self.noiseparam[data['ISO']]['biassig']
        elif 'blc' in self.args['command']:
            # 只使用均值矫正
            lr_raw = lr_raw - self.blc_mean[data['ISO']]

        # 数据转换
        lr_imgs = raw2bayer(lr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=False)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        # 前半部分和paired data的读取方式一样
        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'])
            hr_crops = self.random_crop(hr_imgs)
            if data['black_lr']:
                self.init_random_crop_point(mode='random_crop')
                lr_crops = self.random_crop(lr_imgs)
                # 贴信号无关的read noise! HB矫正！
                if 'preHB' not in self.args['command'] and 'HB' in self.args['command']:
                    lr_crops = self.HBR.map(lr_crops, data['ISO'], norm=True)
            else:
                lr_crops = self.random_crop(lr_imgs)
            lr_shape = lr_crops.shape
            data['ratio'] = np.ones(lr_shape[0], dtype=np.float32) * dgain
            # cpu preprocess
            if self.args['gpu_preprocess'] is False:
                # not check yet, please use GPU
                raise NotImplementedError
                aug_r, aug_g, aug_b = get_aug_param_torch(data, b=1, numpy=True, camera_type=self.args['camera_type'])
                aug_wb = np.array([aug_r, aug_g, aug_b, aug_g])
                data['rgb_gain'] = np.ones(lr_shape[0], dtype=np.float32) * (aug_g + 1)
                if np.abs(aug_wb).max() != 0:
                    data['wb'] *= (1+aug_wb) / (1+aug_g)
                    # data['ratio'] /= (1+aug_g) # 两边信号同时增加，ratio不该动
                    for i in range(lr_shape[0]):
                        lr_crops[i], hr_crops[i] = raw_wb_aug(lr_crops[i], hr_crops[i], iso=data['ISO'],
                            aug_wb=aug_wb, camera_type=self.args['camera_type'], ratio=dgain, ori=self.args['ori'])
                else:
                    lr_crops = lr_crops if self.args['ori'] else lr_crops * dgain
                
                if self.args['clip']:
                    lr_crops = lr_crops.clip(0, 1)
                    hr_crops = hr_crops.clip(0, 1)
        else:
            raise NotImplementedError

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

class ELD_Dataset(RealBase_Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()
        self.args['ori'] = False
        self.args['dstname'] = 'ELD'
        self.args['mode'] = 'eval'

    def initialization(self):
        # 获取数据地址
        self.suffix = 'ARW'
        self.dataset_file = f'ELD_SonyA7S2.info'
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos = pkl.load(info_file)
            print(f'>> Successfully load "{self.dataset_file}" (Length: {len(self.infos)})')
        self.iso_list = self.args['iso_list']
        self.ratio_list = self.args['ratio_list']
        self.imgs_per_scene = len(self.iso_list) * len(self.ratio_list)
        self.length = len(self.infos) * len(self.iso_list) * len(self.ratio_list)
        self.get_shape()
        self.darkshading = {}
        self.blc_mean = {}
        self.naive = False if '++' in self.args['command'] else True
        if 'darkshading' in self.args['command'] or 'blc'  in self.args['command']:
            for iso in self.iso_list:
                # read darkshaidng
                if self.naive:
                    if os.path.exists(os.path.join(self.args['ds_dir'], f'darkshading_BLE.pkl')):
                        with open(os.path.join(self.args['ds_dir'], f'darkshading_BLE.pkl'), 'rb') as f:
                            self.blc_mean = pkl.load(f)
                    self.get_darkshading(iso)
                    self.blc_mean[iso] = raw2bayer(self.darkshading[iso], norm=False, clip=False, 
                                            wp=self.args['wp']-self.args['bl'], bl=0)
                    self.blc_mean[iso] = np.mean(self.blc_mean[iso])
                else:
                    self.get_darkshading(iso, naive=self.naive)

    def __len__(self):
        return self.length

    def get_raw_id(self, scene_id, iso, ratio):
        for i in range(len(self.infos[scene_id])):
            raw_iso = self.infos[scene_id][i]['ISO']
            raw_ratio = self.infos[scene_id][i]['ratio']
            if raw_iso == iso and raw_ratio == ratio:
                img_id = i + 1
                break
        # 就近选gt
        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(img_id - gt_ids))
        gt_id = gt_ids[ind]
        return img_id-1, gt_id-1

    def fast_eval(self, on=True):
        if on:
            # self.iso_list = self.args['iso_list'][-1:]
            self.infos_backup = self.infos.copy()
            self.infos = [self.infos[-3], self.infos[-1]]
            self.ratio_list = self.args['ratio_list'][-1:]
            self.recheck_length()
        else:
            self.infos = self.infos_backup.copy()
            self.iso_list = self.args['iso_list']
            self.ratio_list = self.args['ratio_list']
            self.recheck_length()

    def recheck_length(self):
        self.imgs_per_scene = len(self.iso_list) * len(self.ratio_list)
        self.length = len(self.infos) * len(self.iso_list) * len(self.ratio_list)
    
    def remap_darkshading(self, num=None):
        log(f'Using {num} to remap dark shading')
        for iso in [800,1600,3200]:
            self.get_darkshading(iso, num, remake=True)
    
    def remap_BLE(self, num=None):
        log(f'Using {num} to remap BLE')
        with open(os.path.join(f'/data/SonyA7S2/ds_ablation/darkshading_BLE_{num}.pkl'), 'rb') as f:
            self.blc_mean = pkl.load(f)
        for iso in [800,1600,3200]:
            self.get_darkshading(iso, remake=True)

    def __getitem__(self, idx):
        data = {}
        # 划分数据，get id
        scene_id = idx // self.imgs_per_scene
        img_idx = idx % self.imgs_per_scene
        iso_idx = img_idx // len(self.ratio_list)
        ratio_idx = img_idx % len(self.ratio_list)
        data['ISO'] = self.iso_list[iso_idx]
        data['ratio'] = self.ratio_list[ratio_idx]
        lr_id, hr_id = self.get_raw_id(scene_id, data['ISO'], data['ratio'])
        # 读取数据
        data['wb'] = self.infos[scene_id][hr_id]['wb']
        data['ccm'] = self.infos[scene_id][hr_id]['ccm']
        data['name'] = f"scene-{scene_id+1:02d}_{self.infos[scene_id][lr_id]['name']}"
        data['ExposureTime'] = self.infos[scene_id][hr_id]['ExposureTime'] * 1000
        data['exp'] = data['ExposureTime'] / data['ratio']
        
        hr_raw = np.array(dataload(self.infos[scene_id][hr_id]['data'])).reshape(self.H,self.W)
        lr_raw = np.array(dataload(self.infos[scene_id][lr_id]['data'])).reshape(self.H,self.W)

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=data['ISO'], exp=data['exp'], naive=self.naive)
            if 'd' in self.args['noise_code']:
                lr_raw = lr_raw + self.get_darkshading(iso=data['ISO'], exp=data['exp'], naive=self.naive).mean()
        if 'blc' in self.args['command'] and 'HB' not in self.args['command']:
            lr_raw = lr_raw - self.blc_mean[data['ISO']]

        # 数据转换
        lr_imgs = raw2bayer(lr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=False)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        hr_crops = hr_imgs[None,:]
        lr_crops = lr_imgs[None,:]

        if self.args['ori'] is False:
            lr_crops *= data['ratio']

        if self.args['clip']:
            # -100 -> -inf
            lb = -100 if 'HB' in self.args['command'] else 0
            lr_crops = lr_crops.clip(lb, 1)
            
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

class TestDataset(RealBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.data_dir = self.args['data_dir']
        self.suffix = '.' + self.args['suffix']
        self.initialization()
    
    def default_args(self):
        self.args = {}
        self.args['data_dir'] = '/data/SID/Sony/long'
        self.args['suffix'] = 'ARW'
        self.args['ori'] = False
        self.args['ratio'] = 1
        self.args['dstname'] = 'SID'
        self.args['camera_type'] = 'SonyA7S2'
        self.args['mode'] = 'eval'
        self.args['command'] = ''
        self.args['wp'] = 16383
        self.args['bl'] = 512
        self.args['clip'] = False

    def initialization(self):
        self.dataname = []
        self.datapath = []
        for name in sorted(os.listdir(self.data_dir)):
            if self.suffix not in name:
                continue
            if 'trainonly' in self.args['command'].lower():
                if name[0] == '0': continue
            datapath = os.path.join(self.data_dir, name)
            info = get_ISO_ExposureTime(datapath)
            iso = info['ISO']
            if 'lowISO' in self.args['command']:
                if iso > 1600: continue
            if 'highISO' in self.args['command']:
                if iso <= 1600: continue
            self.dataname.append(name[:-len(self.suffix)])
            self.datapath.append(os.path.join(self.data_dir, name))
        # if 'limitediso' in self.args['command'].lower():
        #     self.datapath = [path for path in self.datapath if path.split('/')[-1][0] == '0']

    def __len__(self):
        return len(self.dataname)
    
    def __getitem__(self, idx):
        data = {}
        data['rawpath'] = self.datapath[idx]
        gt_raw = rawpy.imread(self.datapath[idx])
        data['data'] = pack_raw_bayer(gt_raw, wp=16383, clip=False) * self.args['ratio']
        data['wb'], data['ccm'] = read_wb_ccm(gt_raw)
        data['name'] = self.dataname[idx]
        data['ratio'] = self.args['ratio']
        if self.args['clip']:
            data['data'] = data['data'].clip(0,1)

        return data

if __name__=='__main__':
    pass