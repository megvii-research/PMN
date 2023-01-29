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

class PhoneBase_Dataset(Dataset):
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
        self.args['root_dir'] = '/data/LRID/'
        self.args['suffix'] = 'dng'
        self.args['crop_per_image'] = 12
        self.args['crop_size'] = 512
        self.args['ori'] = True
        self.args['ratio'] = 4
        self.args['ratio_list'] = [1,2,4,8,16]
        self.args['dstname'] = 'indoor_x5'
        self.args['camera_type'] = 'IMX686'
        self.args['params'] = None
        self.args['noise_code'] = 'p'
        self.args['mode'] = 'train'
        self.args['croptype'] = 'non-overlapped'
        self.args['GT_type'] = 'GT_align_ours'
        self.args['command'] = 'alldg'
        self.args['H'] = 3472
        self.args['W'] = 4624
        self.args['wp'] = 1023
        self.args['bl'] = 64
        self.args['clip'] = False

    def initialization(self):
        # 获取数据地址
        self.suffix = 'dng'
        self.dataset_file = f'{self.args["dstname"]}_{self.args["GT_type"]}.info'
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos_gt = pkl.load(info_file)
        with open(f'infos/{self.args["dstname"]}_short.info', 'rb') as info_file:
            self.infos_short = pkl.load(info_file)
        self.infos = self.infos_gt
        for i in range(len(self.infos)):
            self.infos[i]['hr'] = self.infos[i]['data']
            self.infos[i]['lr'] = {dgain: self.infos_short[dgain][i] for dgain in self.infos_short}
            del self.infos[i]['data']
        print(f'>> Successfully load "{self.dataset_file}" (Length: {len(self.infos)})')
        self.data_split()
        self.change_ratio_list(self.args['ratio_list'])
        self.length = len(self.id_remap)
        self.get_shape()
        # train the pairs of all dgain in each epoch
        if 'small' in self.args['command'] and self.args['mode'] == 'train':
            div = 1/4
            if 'small2' in self.args['command']:
                div = 1/2
            elif 'small3' in self.args['command']:
                div = 3/4
            log(f'Go into small branch, [:{int(len(self.id_remap)*div)}] scenes will be used')
            self.id_remap = self.id_remap[:int(len(self.id_remap)*div)]
            self.length = len(self.id_remap)
        if 'alldg' in self.args['command'] and self.args['mode'] == 'train':
            self.lens_extend(on=True)
        # load darkshading
        self.darkshading = {}
        self.darkshading_hot = {}
        self.noiseparam = {}
        self.blc_mean = {6400:np.array([0,0,0,0],np.float32)}
        self.blc_mean_hot = {6400:np.array([0,0,0,0],np.float32)}
        self.iso = 6400
        iso = 6400
        if 'darkshading' in self.args['command']:
            log(f'Loading darkshading into buffer...')
            # read darkshaidng and blc
            self.get_darkshading(iso=iso, hot=False)
            self.get_darkshading(iso=iso, hot=True)

        if iso not in self.noiseparam:
            nlf_path = os.path.join(self.args['ds_dir'], f'noiseparam-iso-{iso}.h5')
            f = h5py.File(nlf_path, 'r')
            # raise NotImplementedError
            # Kmax没改
            self.noiseparam[iso] = {
                'K':8.7425333, 'lam':np.mean(f['lam']), 
                'sigGs':np.mean(f['sigmaGs']), 'sigGssig':np.std(f['sigmaGs']),
                'sigTL':np.mean(f['sigmaTL']), 'sigTLsig':np.std(f['sigmaTL']),
                'sigR':np.mean(f['sigmaR']), 'sigRsig':np.std(f['sigmaR']),
                'bias':np.array([-0.08113494,-0.04906388,-0.9408157,-1.2048522]), 
                'biassig':np.std(f['meanRead'], axis=1),
                'q':1/(2**10), 'wp':1023, 'bl':64
                }

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
    
    def lens_extend(self, on=True):
        if on:
            self.length = len(self.id_remap) * len(self.ratio_list)
        else:
            self.length = len(self.id_remap)
        log(f'Dataset length has change to {self.length}...')
    
    def data_split(self, eval_ids=None):
        def remap(id_remap, del_list=[]):
            for id in del_list:
                id_remap.remove(id)
            return id_remap

        self.id_remap = list(range(len(self.infos)))
        if eval_ids is None:
            if self.args['dstname'] == 'indoor_x5':
                eval_ids = [4,14,25,41,44,51,52,53,58]
            elif self.args['dstname'] == 'indoor_x3':
                eval_ids = []#[0,6,15]
            elif self.args['dstname'] == 'outdoor_x5':
                eval_ids = [1,2,5]
            elif self.args['dstname'] == 'outdoor_x3':
                eval_ids = [9,21,22,32,44,51]
            else:
                eval_ids = []

        if self.args['mode'] == 'train':
            self.id_remap = remap(self.id_remap, del_list=eval_ids)
        else:
            self.id_remap = eval_ids
    
    def fast_eval(self, on=True):
        if on:
            if self.args['dstname'] == 'indoor_x5':
                eval_ids = [44,51,53]
            elif self.args['dstname'] == 'indoor_x3':
                eval_ids = [0]
            elif self.args['dstname'] == 'outdoor_x5':
                eval_ids = [1,2,5]
            elif self.args['dstname'] == 'outdoor_x3':
                eval_ids = [44,51]
            self.data_split(eval_ids=eval_ids)
        else:
            self.data_split()
        self.change_ratio_list(self.ratio_list)
        self.length = len(self.id_remap)

    def change_ratio_list(self, ratio_list=[1,2,4,8,16]):
        self.ratio_list = ratio_list
        self.dgain = ratio_list[-1]
        log(f'ratio_list is {ratio_list}, default_ratio is {self.dgain}')

    def change_eval_ratio(self, idx=None, ratio=None):
        assert idx is not None or ratio is not None, 'Are you kidding me?'
        if idx is not None:
            assert idx in [0,1,2,3,4], 'idx must in [0,1,2]'
            ratio = self.ratio_list[idx]
        elif ratio is not None:
            assert int(ratio) in self.ratio_list, 'dgain must in [1, 2, 4]'
            # idx = int(math.log(int(dgain))) 
            ratio = int(ratio)

        self.dgain = ratio
        log(f'Eval ratio {ratio}')

    def blc_rggb(self, raw, bias):
        return rggb2bayer(bayer2rggb(raw) + bias.reshape(1,1,4))
    
    def get_bias(self, iso=6400, exp=30, naive=False, hot=False):
        if hot:
            bias = self.blc_mean_hot[iso][:,0] * exp + self.blc_mean_hot[iso][:,1] # RGGB: (4,)
        else:
            bias = self.blc_mean[iso][:,0] * exp + self.blc_mean[iso][:,1] # RGGB: (4,)
        return bias

    def record_bias_frames(self):
        log('Recording Bias Frames...')
        if 'blacks' in self.args:
            log('You have record bias before...')
        else:
            self.legalISO = np.array([6400])
            self.black_dirs = [os.path.join(self.args['bias_dir'], '6400')]
            self.black_hot_dirs = [os.path.join(self.args['bias_dir']+'-hot', '6400')]
            self.blacks = [None] * len(self.legalISO)
            self.blacks_hot = [None] * len(self.legalISO)
            for i in range(len(self.legalISO)):
                self.blacks[i] = [os.path.join(self.black_dirs[i], filename) for filename in os.listdir(self.black_dirs[i])]
                self.ExposureTime = [int(filename.split('_')[1][4:])/1.0e6 for filename in os.listdir(self.black_dirs[i])]
                self.blacks_hot[i] = [os.path.join(self.black_hot_dirs[i], filename) for filename in os.listdir(self.black_hot_dirs[i])]
                self.ExposureTime_hot = [int(filename.split('_')[1][4:])/1.0e6 for filename in os.listdir(self.black_hot_dirs[i])]

            if 'buffer' in self.args['command']:
                log('Loading Bias Frames into buffer...')
                self.buffer = []
                self.buffer_hot = []
                for lr_id in tqdm(range(len(self.blacks[0]))):
                    self.buffer.append(rawpy.imread(self.blacks[0][lr_id]).raw_image_visible)
                for lr_id in tqdm(range(len(self.blacks_hot[0]))):
                    self.buffer_hot.append(rawpy.imread(self.blacks_hot[0][lr_id]).raw_image_visible)

    def get_darkshading(self, iso=6400, exp=16, naive=False, hot=False):
        if iso not in self.darkshading:
            if naive: # naive darkshading
                ds_path = os.path.join(self.args['ds_dir'], f'darkshading-iso-{iso}.npy')
                self.darkshading[iso] = np.load(ds_path)
                self.darkshading_hot[iso] = np.load(ds_path[:-4]+'-hot.npy')
            else: # linear darkshading
                self.ds_tk = np.load(os.path.join(self.args['ds_dir'], f'darkshading_tk.npy'))
                self.ds_tk_hot = np.load(os.path.join(self.args['ds_dir'], f'darkshading_tk_hot.npy'))
                ds_tb = np.load(os.path.join(self.args['ds_dir'], f'darkshading_tb.npy'))
                ds_tb_hot= np.load(os.path.join(self.args['ds_dir'], f'darkshading_tb_hot.npy'))
                # rggb, k*exp+b
                with open(os.path.join(self.args['ds_dir'], f'BLE_t.pkl'),'rb') as f:
                    self.blc_mean = pickle.load(f)
                with open(os.path.join(self.args['ds_dir'], f'BLE_t_hot.pkl'),'rb') as f:
                    self.blc_mean_hot = pickle.load(f)
                # exp的单位是ms
                self.darkshading[iso] = self.ds_tk * 30 + ds_tb
                self.darkshading_hot[iso] = self.ds_tk_hot * 30 + ds_tb_hot
                bias = self.get_bias(iso, 30, naive=False, hot=False)
                bias_hot = self.get_bias(iso, 30, naive=False, hot=True)
                self.darkshading[iso] = self.blc_rggb(self.darkshading[iso], bias)
                self.darkshading_hot[iso] = self.blc_rggb(self.darkshading_hot[iso], bias_hot)
                ds_path = os.path.join(self.args['ds_dir'], f'darkshading-iso-{iso}.npy')
                # self.darkshading[iso] = np.load(ds_path)
                # self.darkshading_hot[iso] = np.load(ds_path[:-4]+'-hot.npy')
                log('Welcome to choose linear darkshading~~')

        if naive:
            ds = self.darkshading_hot[iso] if hot else self.darkshading[iso]
        else:
            ds = self.darkshading_hot[iso] if hot else self.darkshading[iso]
            # ds_tk均值接近0，方差极小，忽略完全无影响
            # ds_tk = self.ds_tk_hot if hot else self.ds_tk
            # ds_delta = (exp-30) * ds_tk
            bias_delta = self.get_bias(iso, exp, naive, hot) - self.get_bias(iso, 30, naive, hot)
            ds = ds + bias_delta.mean()
            # ds = ds + ds_delta
        return ds

    def hot_check(self, idx):
        hot_ids = []
        if self.args['dstname'] == 'indoor_x5':
            hot_ids = [6,15,33,35,39,46,37,59]
        elif self.args['dstname'] == 'indoor_x3':
            hot_ids = [1,2,4,5,6,10,12,13,14,15,16,17,18,19]
        elif self.args['dstname'] == 'outdoor_x3':
            hot_ids = [0,1,2,3,4,5,7,10,11,12,13,14,15,16,17,18,19,22,26,30,51,52,54,55,56]
        elif self.args['dstname'] == 'outdoor_x5':
            hot_ids = [0,1,2,3,4,5,6]

        hot = True if idx in hot_ids else False
        return hot

class Real_Dataset(PhoneBase_Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = 3472, 4624

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']
        data['ISO'] = self.infos[idx]['ISO']
        data['ExposureTime'] = self.infos[idx]['ExposureTime']
        
        hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        lr_id = np.random.randint(len(self.infos[idx]['short'])) if self.args['mode']=='train' else 0
        lr_raw = np.array(dataload(self.infos[idx]['short'][lr_id])).reshape(self.H,self.W)
        data['ratio'] = self.infos[idx]['ratio'][lr_id]

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(data['ISO'])

        # 数据转换
        lr_imgs = raw2bayer(lr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=False)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'])
            hr_crops = self.random_crop(hr_imgs)
            lr_crops = self.random_crop(lr_imgs)
        elif self.args["mode"] == 'eval':
            hr_crops = hr_imgs[None,:]
            lr_crops = lr_imgs[None,:]

        if self.args['clip']:
            lr_crops = lr_crops.clip(0,1)
        elif 'HB' in self.args['command']:
            lr_crops = np.minimum(lr_crops, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data


class IMX686_Dataset(PhoneBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.root_dir = self.args['root_dir']
        self.initialization()
    
    def default_args(self):
        super().default_args()
        self.args['ratio'] = 16
        self.args['ratio_list'] = [1,2,4,8,16]

    def initialization(self):
        super().initialization()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = {}
        dgain = self.dgain
        iso = self.iso
        idr = self.id_remap[idx % len(self.id_remap)]
        # choose train strategy (dgain-related)
        if self.args['mode']=='train':
            if 'alldg' in self.args['command']:
                dgain_id = idx // len(self.id_remap)
            elif 'rdg' in self.args['command']:
                dgain_id = np.random.randint(len(self.ratio_list))
            else:
                dgain_id = np.argwhere(np.array(self.ratio_list) == dgain)[0,0]
            dgain = self.ratio_list[dgain_id]
        else:
            dgain_id = np.argwhere(np.array(self.ratio_list) == dgain)[0,0]
        # dataload
        hr_raw = np.array(dataload(self.infos[idr]['hr'])).reshape(self.H,self.W)
        lr_random_range = len(self.infos[idr]['lr'][dgain]['data']) # 10
        if 'lr' in self.args['command']:
            lr_random_range = min(self.args['max_lr'], lr_random_range) # 10
        lr_id = np.random.randint(lr_random_range) if self.args['mode']=='train' else 0
        lr_raw = np.array(dataload(self.infos[idr]['lr'][dgain]['data'][lr_id])).reshape(self.H,self.W)

        data['name'] = f"{self.infos[idr]['name']}_x{dgain:02d}"
        data['ratio'] = dgain
        data['ccm'] = self.infos[idr]['ccm']
        data['wb'] = self.infos[idr]['wb']
        data['ISO'] = iso
        data['ExposureTime'] = self.infos[idr]['lr'][dgain]['metadata'][lr_id]['ExposureTime'] * 1000

        naive = False if '++' in self.args['command'] else True
        hot = self.hot_check(int(self.infos[idr]['name'][-3:]))

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=iso, exp=data['ExposureTime'], naive=naive, hot=hot)
            if 'darkshading2' in self.args['command'] and self.args['mode'] == 'train':
                # there are no darkshading on phone's GT
                bias_jitter = np.random.randn() * 0.1
                lr_raw = lr_raw + bias_jitter
        
        if 'blc' in self.args['command']:
            bias = self.get_bias(iso, data['ExposureTime'], naive, hot)
            lr_raw = self.blc_rggb(lr_raw, -bias)
            if 'blc2' in self.args['command'] and self.args['mode'] == 'train':
                bias_hr = self.get_bias(100, data['ExposureTime']*64*dgain, naive, hot)
                hr_raw = self.blc_rggb(hr_raw, -bias_hr)
            if 'nblc' in self.args['command']:
                bias_old = np.array([-0.08113494,-0.04906388,-1.2048522,-0.9408157], np.float32)
                lr_raw = self.blc_rggb(lr_raw, bias + bias_old)

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

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        if self.args['ori'] is False:
            data['lr'] = data['lr'] * dgain
        if self.args['clip']:
            data['lr'] = data['lr'].clip(0,1)

        return data

class IMX686_Mix_Dataset(PhoneBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
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
        dgain = self.dgain
        iso = self.iso
        idr = self.id_remap[idx % len(self.id_remap)]
        # choose train strategy (dgain-related)
        if self.args['mode']=='train':
            if 'alldg' in self.args['command']:
                dgain_id = idx // len(self.id_remap)
            elif 'rdg' in self.args['command']:
                dgain_id = np.random.randint(len(self.ratio_list))
            else:
                dgain_id = np.argwhere(np.array(self.ratio_list) == dgain)[0,0]
            dgain = self.ratio_list[dgain_id]
        else:
            dgain_id = np.argwhere(np.array(self.ratio_list) == dgain)[0,0]
        
        # dataload
        data['name'] = f"{self.infos[idr]['name']}_x{dgain:02d}"
        data['ccm'] = self.infos[idr]['ccm']
        data['wb'] = self.infos[idr]['wb']
        data['ISO'] = iso
        data['ratio'] = dgain
        naive = False if '++' in self.args['command'] else True
        hot = self.hot_check(int(self.infos[idr]['name'][-3:]))
        black_index = self.blacks_hot if hot else self.blacks

        hr_raw = np.array(dataload(self.infos[idr]['hr'])).reshape(self.H,self.W)

        data['black_lr'] = True if 'HB' in self.args['command'] and not np.random.randint(5) else False
        if data['black_lr']:
            # SNA+黑图
            iso_index = np.argmin(np.abs(self.legalISO - data['ISO']))
            lr_id = np.random.randint(len(black_index[iso_index])) if self.args['mode']=='train' else 0
            if 'lr10' in self.args['command']:
                lr_id = np.random.randint(10)
            if 'buffer' not in self.args['command']:
                lr_raw = rawpy.imread(black_index[iso_index][lr_id]).raw_image_visible
            else:
                buffer = self.buffer_hot if hot else self.buffer
                lr_raw = buffer[lr_id].copy()
            dgain = 20
            ExposureTime = self.ExposureTime_hot if hot else self.ExposureTime
            data['ExposureTime'] = ExposureTime[lr_id]
        else:
            lr_id = np.random.randint(len(self.infos[idr]['lr'][dgain]['data'])) if self.args['mode']=='train' else 0
            lr_raw = np.array(dataload(self.infos[idr]['lr'][dgain]['data'][lr_id])).reshape(self.H,self.W)
            data['ExposureTime'] = self.infos[idr]['lr'][dgain]['metadata'][lr_id]['ExposureTime'] * 1000

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=iso, exp=data['ExposureTime'], naive=naive, hot=hot)
            if 'darkshading2' in self.args['command'] and self.args['mode'] == 'train' and not data['black_lr']:
                # there are no darkshading on phone's GT
                bias_jitter = np.random.randn() * 0.1
                lr_raw = lr_raw + bias_jitter
        
        if 'blc' in self.args['command']:
            bias = self.get_bias(iso, data['ExposureTime'], naive, hot)
            lr_raw = self.blc_rggb(lr_raw, -bias)
            if 'blc2' in self.args['command'] and self.args['mode'] == 'train':
                bias_hr = self.get_bias(100, data['ExposureTime']*64*dgain, naive, hot)
                hr_raw = self.blc_rggb(hr_raw, -bias_hr)

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
                lr_crops = self.HBR.map(lr_crops, iso, norm=True)
            else:
                lr_crops = self.random_crop(lr_imgs)
                
            lr_shape = lr_crops.shape
            b = lr_shape[0]
            data['ratio'] = np.ones(b, dtype=np.float32) * dgain

            # cpu preprocess
            if self.args['gpu_preprocess'] is False:
                aug_r, aug_g, aug_b = get_aug_param_torch(data, b=b, command=self.args['command'], 
                                        camera_type=self.args['camera_type']).numpy()
                aug_wb = np.array([aug_r, aug_g, aug_b, aug_g])
                data['rgb_gain'] = np.ones(b, dtype=np.float32) * (aug_g + 1)
                if np.abs(aug_wb).max() != 0:
                    data['wb'] *= (1+aug_wb) / (1+aug_g)
                    # data['ratio'] /= (1+aug_g) # 两边信号同时增加，ratio不该动
                    for i in range(b):
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

class IMX686_Raw_Dataset(PhoneBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        idr = self.id_remap[idx % len(self.id_remap)]
        data['wb'] = self.infos[idr]['wb']
        data['ccm'] = self.infos[idr]['ccm']
        data['name'] = self.infos[idr]['name']
        hr_raw = np.array(dataload(self.infos[idr]['hr'])).reshape(self.H,self.W)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            hr_crops = self.random_crop(hr_imgs)
        elif self.args["mode"] == 'eval':
            hr_crops = hr_imgs[None,:]
        lr_shape = hr_crops.shape
        
        if self.args["lock_wb"] is False and np.random.randint(2):
            rgb_gain, red_gain, blue_gain = random_gains(camera_type='IMX686')
            red_gain = data['wb'][0] / red_gain.numpy()
            blue_gain = data['wb'][2] / blue_gain.numpy()
            hr_crops *= rgb_gain.numpy()
            hr_crops[:,0] = hr_crops[:,0] * red_gain
            hr_crops[:,2] = hr_crops[:,2] * blue_gain
            # data['rgb_gain'] = np.ones(lr_shape[0], dtype=np.float32) * rgb_gain.numpy()

        lr_crops = hr_crops.copy()
        # 人工加噪声
        data['ratio'] = np.ones(lr_shape[0], dtype=np.float32)
        if self.args['gpu_preprocess'] is False:
            for i in range(lr_shape[0]):
                if self.args['params'] is None:
                    # sample_params_max训练单ISO/dgain的模型
                    noise_param = self.noiseparam[self.iso].copy()
                    noise_param['K'] = noise_param['K'] * (1 + np.random.uniform(low=-0.01, high=+0.01))
                    noise_param['ratio'] = np.random.uniform(low=1, high=16)
                else:
                    noise_param = self.args['params']
                data['ratio'][i] = noise_param['ratio']
                lr_crops[i] = generate_noisy_obs(lr_crops[i], param=noise_param, noise_code=self.args['noise_code'], ori=self.args['ori'])
        
        if self.args['clip']:
            lr_crops = lr_crops.clip(0, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

class IMX686_SFRN_Raw_Dataset(PhoneBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
        self.record_bias_frames()
        # 创建HighBitRecovery类
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
        idr = self.id_remap[idx % len(self.id_remap)]
        data['wb'] = self.infos[idr]['wb']
        data['ccm'] = self.infos[idr]['ccm']
        data['name'] = self.infos[idr]['name']
        iso_index = np.random.randint(len(self.legalISO))
        data['ISO'] = self.legalISO[iso_index]
        iso = data['ISO']
        naive = False if '++' in self.args['command'] else True
        hot = self.hot_check(int(self.infos[idr]['name'][-3:]))
        black_index = self.blacks_hot if hot else self.blacks
        
        hr_raw = np.array(dataload(self.infos[idr]['hr'])).reshape(self.H,self.W)
        lr_id = np.random.randint(len(black_index[iso_index])) if self.args['mode']=='train' else 0
        if 'lr10' in self.args['command']:
            lr_id = np.random.randint(10)
        if 'buffer' not in self.args['command']:
            lr_raw = rawpy.imread(black_index[iso_index][lr_id]).raw_image_visible
        else:
            buffer = self.buffer_hot if hot else self.buffer
            lr_raw = buffer[lr_id].copy()

        ExposureTime = self.ExposureTime_hot if hot else self.ExposureTime
        data['ExposureTime'] = ExposureTime[lr_id]
        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=iso, exp=ExposureTime[lr_id], naive=naive, hot=hot)
            if 'darkshading2' in self.args['command'] and self.args['mode'] == 'train':
                # there are no darkshading on phone's GT
                pass

        lr_imgs = raw2bayer(lr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=False)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if 'blc' in self.args['command']:
            lr_imgs = lr_imgs - self.blc_mean[iso].reshape(-1,1,1)/ self.args['wp']
            if 'blc2' in self.args['command'] and self.args['mode'] == 'train':
                hr_imgs = hr_imgs - self.blc_mean[iso].reshape(-1,1,1)/ self.args['wp']

        if self.args['mode'] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            hr_crops = self.random_crop(hr_imgs)
            self.init_random_crop_point(mode='random_crop', raw_crop=False)
            black_crops = self.random_crop(lr_imgs)
            # Raw wb_aug
            if self.args['lock_wb'] is False and np.random.randint(2):
                wb = data["wb"]
                rgb_gain, red_gain, blue_gain = random_gains(camera_type='IMX686')
                red_gain = wb[0] / red_gain.numpy()
                blue_gain = wb[2] / blue_gain.numpy()
                hr_crops *= rgb_gain.numpy()
                hr_crops[:,0] = hr_crops[:,0] * red_gain
                hr_crops[:,2] = hr_crops[:,2] * blue_gain

            lr_crops = hr_crops.copy()
            lr_shape = lr_crops.shape
            data['ratio'] = np.ones(lr_shape[0], dtype=np.float32)
            
            # 加信号相关的shot noise
            for i in range(lr_shape[0]):
                if self.args['params'] is None:
                    # 按ISO生成噪声参数（其实只要泊松）
                    noise_param = self.noiseparam[iso].copy()
                    noise_param['K'] = noise_param['K'] * (1 + np.random.uniform(low=-0.01, high=+0.01))
                    noise_param['ratio'] = np.random.uniform(low=1, high=16)
                else:
                    noise_param = self.args['params']
                data['ratio'][i] = noise_param['ratio']
                lr_crops[i] = generate_noisy_obs(lr_crops[i], param=noise_param, 
                                                noise_code=self.args['noise_code']+'b', ori=self.args['ori'])
            # 贴信号无关的read noise! HB矫正！
            if 'preHB' not in self.args['command'] and 'HB' in self.args['command']:
                black_crops = self.HBR.map(black_crops, iso, norm=True)
            
            if self.args['ori'] is False:
                black_crops = black_crops * data['ratio'].reshape(-1, 1, 1, 1)
            lr_crops = black_crops + lr_crops

        if self.args['clip']:
            lr_crops = lr_crops.clip(0, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

if __name__=='__main__':
    pass