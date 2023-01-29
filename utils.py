import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
import cv2
cv2.setNumThreads(0)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import *
import glob
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image
import time
import socket
import scipy
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import wraps
from tqdm import tqdm
import exifread
import rawpy
import math
import random
import yaml
import pickle
import warnings
import h5py
import pickle as pkl
import warnings
# np.random.seed(7)
# torch.manual_seed(7)
# torch.cuda.manual_seed(7)

fn_time = {}

def timestamp(time_points, n):
    time_points[n] = time.time()
    return time_points[n] - time_points[n-1]

def fn_timer(function, print_log=False):
  @wraps(function)
  def function_timer(*args, **kwargs):
    global fn_timer
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    if print_log:
        print ("Total time running %s: %.6f seconds" %
            (function.__name__, t1-t0))
    if function.__name__ in fn_time :
        fn_time[function.__name__] += t1-t0
    else:
        fn_time[function.__name__] = t1-t0
    return result
  return function_timer

def log(string, log=None, str=False, end='\n', notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')
    else:
        pass
        # os.makedirs('worklog', exist_ok=True)
        # log = f'worklog/worklog-{time.strftime("%Y-%m-%d")}.txt'
        # with open(log,'a+') as f:
        #     f.write(log_string+'\n')
    if str:
        return string+end

def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns

def metrics_recorder(file, names, psnrs, ssims):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            metrics = pkl.load(f)
    else:
        metrics = {}
    for name, psnr, ssim in zip(names, psnrs, ssims):
        metrics[name] = [psnr, ssim]
    with open(file, 'wb') as f:
        pkl.dump(metrics, f)
    return metrics
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', log=True, last_epoch=0):
        self.name = name
        self.fmt = fmt
        self.log = log
        self.history = []
        self.last_epoch = last_epoch
        self.history_init_flag = False
        self.reset()

    def reset(self):
        if self.log:
            try:
                if self.avg>0: self.history.append(self.avg)
            except:
                pass#print(f'Start log {self.name}!')
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def plot_history(self, savefile='log.jpg', logfile='log.pkl'):
        # 读取老log
        if os.path.exists(logfile) and not self.history_init_flag:
            self.history_init_flag = True
            with open(logfile, 'rb') as f:
                history_old = pickle.load(f)
                if self.last_epoch: # 为0则重置
                    self.history = history_old + self.history[:self.last_epoch]
        # 记录log
        with open(logfile, 'wb') as f:
            pickle.dump(self.history, f)
        # 画图
        plt.figure(figsize=(12,9))
        plt.title(f'{self.name} log')
        x = list(range(len(self.history)))
        plt.plot(x, self.history)
        plt.xlabel('Epoch')
        plt.ylabel(self.name)
        plt.savefig(savefile, bbox_inches='tight')
        plt.close()

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def pkl_convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }

def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.color_matrix[:3, :3].astype(np.float32)
    if ccm[0,0] == 0:
        ccm = np.eye(3, dtype=np.float32)
    return wb, ccm

def get_ISO_ExposureTime(filepath):
    # 不限于RAW，RGB图片也适用
    raw_file = open(filepath, 'rb')
    exif_file = exifread.process_file(raw_file, details=False, strict=True)
    # 获取曝光时间
    if 'EXIF ExposureTime' in exif_file:
        exposure_str = exif_file['EXIF ExposureTime'].printable
    else:
        exposure_str = exif_file['Image ExposureTime'].printable
    if '/' in exposure_str:
        fenmu = float(exposure_str.split('/')[0])
        fenzi = float(exposure_str.split('/')[-1])
        exposure = fenmu / fenzi
    else:
        exposure = float(exposure_str)
    # 获取ISO
    if 'EXIF ISOSpeedRatings' in exif_file:
        ISO_str = exif_file['EXIF ISOSpeedRatings'].printable
    else:
        ISO_str = exif_file['Image ISOSpeedRatings'].printable
    if '/' in ISO_str:
        fenmu = float(ISO_str.split('/')[0])
        fenzi = float(ISO_str.split('/')[-1])
        ISO = fenmu / fenzi
    else:
        ISO = float(ISO_str)
    info = {'ISO':int(ISO), 'ExposureTime':exposure, 'name':filepath.split('/')[-1]}
    return info

def load_weights(model, pretrained_dict, multi_gpu=False, by_name=False):
    model_dict = model.module.state_dict() if multi_gpu else model.state_dict()
    # 1. filter out unnecessary keys
    tsm_replace = []
    for k in pretrained_dict:
        if 'tsm_shift' in k:
            k_new = k.replace('tsm_shift', 'tsm_buffer')
            tsm_replace.append((k, k_new))
    for k, k_new in tsm_replace:
        pretrained_dict[k_new] = pretrained_dict[k]
    if by_name:
        del_list = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape != pretrained_dict[k].shape:
                    # 1. Delete values not in key
                    del_list.append(k)
                    # 2. Cat it to the end
                    # diff = model_dict[k].size()[1] - pretrained_dict[k].size()[1]
                    # v = torch.cat((v, v[:,:diff]), dim=1)
                    # 3. Repeat it to same
                    # nframe = model_dict[k].shape[1] // pretrained_dict[k].shape[1]
                    # v = torch.repeat_interleave(v, nframe, dim=1)
                    # 4. Clip it to same
                    # b_model, c_model, h_model, w_model = model_dict[k].shape
                    # c_save = pretrained_dict[k].shape[1]
                    # c_diff = c_model - c_save
                    # if c_model > c_save:
                    #     v = torch.cat((v, torch.empty(b_model, c_diff, h_model, w_model).cuda()), dim=1)
                    # else:
                    #     v = v[:,:c_diff]
                    log(f'Warning:  "{k}":{pretrained_dict[k].shape}->{model_dict[k].shape}')
                pretrained_dict[k] = v
            else:
                del_list.append(k)
                log(f'Warning:  "{k}" is not exist and has been deleted!!')
        for k in del_list:
            del pretrained_dict[k]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    if multi_gpu:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    return model


def tensor_dim5to4(tensor):
    batchsize, crops, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, c, h, w)
    return tensor

def tensor_dim6to5(tensor):
    batchsize, crops, t, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, t, c, h, w)
    return tensor

def frame_index_splitor(nframes=1, pad=True, reflect=True):
    # [b, 7, c, h ,w]
    r = nframes // 2
    length = 7 if pad else 8-nframes
    frames = []
    for i in range(length):
        frames.append([None]*nframes)
    if pad:
        for i in range(7):
            for k in range(nframes):
                frames[i][k] = i+k-r
    else:
        for i in range(8-nframes):
            for k in range(nframes):
                frames[i][k] = i+k
    if reflect:
        frames = num_reflect(frames,0,6)
    else:
        frames = num_clip(frames, 0, 6)
    return frames

def multi_frame_loader(frames ,index, gt=False, keepdims=False):
    loader = []
    for ind in index:
        imgs = []
        if gt:
            r = len(index[0]) // 2
            tensor = frames[:,ind[r],:,:,:]
            if keepdims:
                tensor = tensor.unsqueeze(dim=1)
        else:
            for i in ind:
                imgs.append(frames[:,i,:,:,:])
                tensor = torch.stack(imgs, dim=1)
        loader.append(tensor)
    return torch.stack(loader, dim=0)

def num_clip(nums, mininum, maxinum):
    nums = np.array(nums)
    nums = np.clip(nums, mininum, maxinum)
    return nums

def num_reflect(nums, mininum, maxinum):
    nums = np.array(nums)
    nums = np.abs(nums-mininum)
    nums = maxinum-np.abs(maxinum-nums)
    return nums

def get_host_with_dir(dataset_name=''):
    multi_gpu = False
    hostname = socket.gethostname()
    log(f"User's hostname is '{hostname}'")
    if hostname == 'fenghansen':
        host = '/data'
    elif hostname == 'DESKTOP-FCAMIOQ':
        host = 'F:/datasets'
    elif hostname == 'BJ-DZ0101767':
        host = 'F:/Temp'
    else:
        host = '/data'
        multi_gpu = True if torch.cuda.device_count() > 1 else False
    return hostname, host + dataset_name, multi_gpu

def scale_down(img):
    return np.float32(img) / 255.

def scale_up(img):
    return np.uint8(img * 255.)

def feature_vis(tensor, name='out', save=False):
    feature = tensor.detach().cpu().numpy().transpose(0,2,3,1)
    if save:
        for i in range(len(feature)):
            cv2.imwrite(f'./test/{name}_{i}.png', np.uint8(feature[i,:,:,::-1]*255))
    return feature

def bayer2rggb(bayer):
    H, W = bayer.shape
    return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)

def rggb2bayer(rggb):
    H, W, _ = rggb.shape
    return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H*2, W*2)

def repair_bad_pixels(raw, bad_points, method='median'):
    fixed_raw = bayer2rggb(raw)
    for i in range(4):
        fixed_raw[:,:,i] = cv2.medianBlur(fixed_raw[:,:,i],3)
    fixed_raw = rggb2bayer(fixed_raw)
    # raw = (1-bpc_map) * raw + bpc_map * fixed_raw
    for p in bad_points:
        raw[p[0],p[1]] = fixed_raw[p[0],p[1]]
    return raw

def img4c_to_RGB(img4c, wb=None, ccm=None, gamma=2.2):
    c,h,w = img4c.shape
    H = h * 2
    W = w * 2
    raw = np.zeros((H,W), np.float32)
    red_gain = wb[0] if wb is not None else 2
    blue_gain = wb[2] if wb is not None else 2
    raw[0:H:2,0:W:2] = img4c[0] * red_gain # R
    raw[0:H:2,1:W:2] = img4c[1] # G1
    raw[1:H:2,1:W:2] = img4c[2] * blue_gain # B
    raw[1:H:2,0:W:2] = img4c[3] # G2
    raw = np.clip(raw, 0, 1)
    white_point = 16383
    raw = raw * white_point
    img = cv2.cvtColor(raw.astype(np.uint16), cv2.COLOR_BAYER_BG2RGB_EA) / white_point
    if ccm is None:
        ccm = np.array( [[ 1.9712269,-0.6789218, -0.29230508],
                        [-0.29104823, 1.748401 , -0.45735288],
                        [ 0.02051281,-0.5380369,  1.5175241 ]])
    img = img[:, :, None, :]
    ccm = ccm[None, None, :, :]
    img = np.sum(img * ccm, axis=-1)
    img = np.clip(img, 0, 1) ** (1/gamma)
    return img

def FastGuidedFilter(p,I,d=7,eps=1):
    p_lr = cv2.resize(p, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    I_lr = cv2.resize(I, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    mu_p = cv2.boxFilter(p_lr, -1, (d, d)) 
    mu_I = cv2.boxFilter(I_lr,-1, (d, d)) 
    
    II = cv2.boxFilter(np.multiply(I_lr,I_lr), -1, (d, d)) 
    Ip = cv2.boxFilter(np.multiply(I_lr,p_lr), -1, (d, d))
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.resize(cv2.boxFilter(a, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    mu_b = cv2.resize(cv2.boxFilter(b, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def GuidedFilter(p,I,d=7,eps=1):
    mu_p = cv2.boxFilter(p, -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    mu_I = cv2.boxFilter(I,-1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    
    II = cv2.boxFilter(np.multiply(I,I), -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    Ip = cv2.boxFilter(np.multiply(I,p), -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.boxFilter(a, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    mu_b = cv2.boxFilter(b, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def plot_sample(img_lr, img_dn, img_hr, filename='result', model_name='Unet', 
                epoch=-1, print_metrics=False, save_plot=True, save_path='./', res=None):
    if np.max(img_hr) <= 1:
        # 变回uint8
        img_lr = scale_up(img_lr)
        img_dn = scale_up(img_dn)
        img_hr = scale_up(img_hr)
    # 计算PSNR和SSIM
    if res is None:
        psnr = []
        ssim = []
        psnr.append(compare_psnr(img_hr, img_lr))
        psnr.append(compare_psnr(img_hr, img_dn))
        ssim.append(compare_ssim(img_hr, img_lr, channel_axis=-1))
        ssim.append(compare_ssim(img_hr, img_dn, channel_axis=-1))
        psnr.append(-1)
        ssim.append(-1)
    else:
        psnr = [res[0], res[2], -1]
        ssim = [res[1], res[3], -1]
    # Images and titles
    images = {
        'Noisy Image': img_lr,
        model_name: img_dn,
        'Ground Truth': img_hr
    }
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title("{} - {} - psnr:{:.2f} - ssim{:.4f}".format(title, img.shape, psnr[i], ssim[i]))
        axes[i].axis('off')
    plt.suptitle('{} - Epoch: {}'.format(filename, epoch))
    if print_metrics:
        log('PSNR:', psnr)
        log('SSIM:', ssim)
    # Save directory
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    savefile = os.path.join(save_path, "{}-Epoch{}.jpg".format(filename, epoch))
    if save_plot:
        denoisedfile = os.path.join(save_path, "{}_denoised.png".format(filename))
        cv2.imwrite(denoisedfile, img_dn[:,:,::-1])
        fig.savefig(savefile, bbox_inches='tight')
        plt.close()
    return psnr, ssim, filename

def save_picture(img_sr, save_path='./images/test',frame_id='0000'):
    # 变回uint8
    img_sr = scale_up(img_sr.transpose(1,2,0))
    if os._exists(save_path) is not True:
        os.makedirs(save_path, exist_ok=True)
    plt.imsave(os.path.join(save_path, frame_id+'.png'), img_sr)
    gc.collect()

def test_output_rename(root_dir):
    for dirs in os.listdir(root_dir):
        dirpath = root_dir + '/' + dirs
        f = os.listdir(dirpath)
        end = len(f)
        for i in range(len(f)):
            frame_id = int(f[end-i-1][:4])
            old_file = os.path.join(dirpath, "%04d.png" % frame_id)
            new_file = os.path.join(dirpath, "%04d.png" % (frame_id + 1))
            os.rename(old_file, new_file)
        log(f"path |{dirpath}|'s rename has finished...")

def datalist_rename(root_dir):
    src_file = os.path.join(root_dir, 'sep_testlist.txt')
    dst_file = os.path.join(root_dir, 'sep_evallist.txt')
    sub_dirs = []
    fw = open(dst_file, 'w')
    with open(src_file, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        for sub_path in lines:
            if sub_path[:5] in sub_dirs: continue
            sub_dirs.append(sub_path[:5])
            print(sub_path, file=fw)
    fw.close()
    return sub_dirs

def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy

def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        ssim = compare_ssim(Y, X, data_range=data_range, channel_axis=-1)
        return {'PSNR':psnr, 'SSIM': ssim}
    else:
        raise NotImplementedError

def bayer2rows(bayer):
    H, W = bayer.shape
    return np.stack((bayer[0:H:2], bayer[1:H:2]))

def rows2bayer(rows):
    c, H, W = rows.shape
    bayer = np.empty((H*2, W))
    bayer[0:H*2:2] = rows[0]
    bayer[1:H*2:2] = rows[1]
    return bayer

def mpop(func, idx, *args, **kwargs):
    data = func(*args, **kwargs)
    log(f'Finish task No.{idx}...')
    return idx, func(*args, **kwargs)

def dataload(path):
    suffix = path[-4:].lower()
    if suffix in ['.arw','.dng']:
        data = rawpy.imread(path).raw_image_visible
    elif suffix in ['.npy']:
        data = np.load(path)
    elif suffix in ['.jpg', '.png', '.bmp', 'tiff']:
        data = cv2.imread(path)
    return data

def row_denoise(path, iso, data=None):
    if data is None:
        raw = dataload(path)
    else:
        raw = data
    raw = bayer2rows(raw)
    raw_denoised = raw.copy()
    for i in range(len(raw)):
        rows = raw[i].mean(axis=1)
        rows2 = rows.reshape(1, -1)
        rows2 = cv2.bilateralFilter(rows2, 25, sigmaColor=10, sigmaSpace=1+iso/200, borderType=cv2.BORDER_REPLICATE)[0]
        row_diff = rows-rows2
        raw_denoised[i] = raw[i] - row_diff.reshape(-1, 1)
    raw = rows2bayer(raw)
    raw_denoised = rows2bayer(raw_denoised)
    return raw_denoised

def pth_transfer(src_path='/data/ELD/checkpoints/sid-ours-inc4/model_200_00257600.pt',
                dst_path='checkpoints/SonyA7S2_Official.pth',
                reverse=False):
    model_src = torch.load(src_path, map_location='cpu')
    if reverse:
        model_dst = torch.load(dst_path, map_location='cpu')
        model_src['netG'] = model_dst
        save_dir = os.path.join('pth_transfer', os.path.basename(dst_path)[9:-15])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(src_path))
        torch.save(model_src, save_path)
    else:
        model_src = model_src['netG']
        torch.save(model_src, dst_path)
    
if __name__ == '__main__':
    # pth_transfer('/data/ELD/checkpoints/sid-paired/model_200_00280000.pt', 'checkpoints/SonyA7S2_Paired_Official_last_model')
    # names = [name for name in os.listdir('checkpoints') if 'best_model' in name]
    # for name in tqdm(names):
        # pth_transfer(dst_path=f'checkpoints/{name}', reverse=True)
    root_dir = '/data/SonyA7S2/resources-2087'
    ds_files = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if name[0] == 'd' and name[-4:]=='.npy']
    # bpc_files = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if name[0] == 'b' and name[-4:]=='.npy']
    isos = []
    legal_iso = np.array([50, 64, 80, 8000, 10000, 12800, 16000, 20000, 25600])
    pbar = tqdm(ds_files)
    for file in pbar:
        iso = int(os.path.basename(file)[16:-4])
        isos.append(iso)
        # if iso in legal_iso:
        pbar.set_description_str(f'ISO-{iso}')
        ds_denoised = row_denoise(file,iso)
        np.save(file, ds_denoised)
    print(sorted(isos))