# from .VSR_data_process import *
# from .VSR_datasets import *
# from .optflow import *
from .unprocess import *
from .process import *
from .real_datasets import *
from .img_datasets import *
# from .raw2rgb_datasets import *
from .phone_datasets import *

class MultiDataset(Dataset):
    def __init__(self, dstname_list, args):
        self.args = args
        self.dst_num = len(dstname_list)
        self.dsts = [None] * self.dst_num
        for i in range(self.dst_num):
            dst_args = self.args.copy()
            dst_args['dstname'] = dstname_list[i]
            if 'x3' in dst_args['dstname']:
                dst_args['ratio_list'] = [1,2,4]
            self.dsts[i] = globals()[dst_args['dataset']](dst_args)
        self.lens = [self.dsts[i].__len__() for i in range(self.dst_num)]
    
    def __len__(self):
        return np.sum(self.lens)
    
    def fast_eval(self, on=True):
        for i in range(self.dst_num):
            self.dst[i].fast_eval(on)
    
    def change_eval_ratio(self, ratio=1):
        for i in range(self.dst_num):
            self.dst[i].change_eval_ratio(ratio=ratio)
    
    def __getitem__(self, index):
        new_idx = index
        for i in range(self.dst_num):
            if new_idx >= self.lens[i]:
                new_idx -= self.lens[i]
                continue
            data = self.dsts[i].__getitem__(new_idx)
        return data

class Multi_Real_Dataset(Dataset):
    def __init__(self, root_dir, crop_per_image=4, crop_size=256, ori=True, iso=6400,
                dstfile=None, dstname='indoor', camera_type='IMX686', mode='train', command=''):
        self.dst1 = Real_Dataset(root_dir.replace(dstname, 'indoor'), crop_per_image, crop_size, ori, iso, None, 'indoor', camera_type, mode)
        self.dst2 = Real_Dataset(root_dir, crop_per_image//4, crop_size, ori, iso, None, dstname, camera_type, mode)
        self.l1 = self.dst1.__len__()
        self.l2 = self.dst2.__len__()

    def __len__(self):
        return self.l1 + self.l2# // 4

    def __getitem__(self, index):
        if index < self.l1:
            data = self.dst1.__getitem__(index)
        else:
            new_idx = index - self.l1
            # new_idx = np.max([(index-self.l1)*4+np.random.randint(4), self.l2-1])
            data = self.dst2.__getitem__(new_idx)
        return data

class Multi_Sync_Dataset(Dataset):
    def __init__(self, root_dir, crop_per_image=4, crop_size=256, ori=True, iso=6400, params=None, lock_wb=False,
                use_gpu=False, noise_code='prq', dstname='indoor', camera_type='IMX686', mode='train'):
        self.dst1 = Img_Dataset(root_dir.replace(dstname, 'indoor'), crop_per_image, crop_size, ori, iso, params, lock_wb, 
                                use_gpu, noise_code, 'indoor', camera_type, mode)
        self.dst2 = Mix_Dataset(root_dir, crop_per_image//4, crop_size, ori, iso, params, lock_wb, 
                                use_gpu, noise_code, dstname, camera_type, mode)
        self.l1 = self.dst1.__len__()
        self.l2 = self.dst2.__len__()

    def __len__(self):
        return self.l1 + self.l2 // 4

    def __getitem__(self, index):
        if index < self.l1:
            data = self.dst1.__getitem__(index)
        else:
            new_idx = index - self.l1
            # new_idx = np.max([(index-self.l1)*4+np.random.randint(4), self.l2-1])
            data = self.dst2.__getitem__(new_idx*4)
            for k in range(1,4):
                data_temp = self.dst2.__getitem__(new_idx*4+k)
                data['lr'] = np.concatenate((data['lr'], data_temp['lr']), axis=0)
                data['hr'] = np.concatenate((data['hr'], data_temp['hr']), axis=0)
                data['ratio'] = np.concatenate((data['ratio'], data_temp['ratio']), axis=0)
        return data

class Multi_Mix_Dataset(Dataset):
    def __init__(self, root_dir, crop_per_image=4, crop_size=256, ori=True, iso=6400, params=None, lock_wb=False,
                use_gpu=False, noise_code='prq', dstname='indoor', camera_type='IMX686', mode='train'):
        self.dst1 = Mix_Dataset(root_dir.replace(dstname, 'indoor'), crop_per_image, crop_size, ori, iso, params, lock_wb, 
                                use_gpu, noise_code, 'indoor', camera_type, mode)
        self.dst2 = Mix_Dataset(root_dir, crop_per_image//4, crop_size, ori, iso, params, lock_wb, 
                                use_gpu, noise_code, dstname, camera_type, mode)
        self.l1 = self.dst1.__len__()
        self.l2 = self.dst2.__len__()

    def __len__(self):
        return self.l1 + self.l2 // 4

    def __getitem__(self, index):
        if index < self.l1:
            data = self.dst1.__getitem__(index)
        else:
            new_idx = index - self.l1
            # new_idx = np.max([(index-self.l1)*4+np.random.randint(4), self.l2-1])
            data = self.dst2.__getitem__(new_idx*4)
            for k in range(1,4):
                data_temp = self.dst2.__getitem__(new_idx*4+k)
                data['lr'] = np.concatenate((data['lr'], data_temp['lr']), axis=0)
                data['hr'] = np.concatenate((data['hr'], data_temp['hr']), axis=0)
                data['ratio'] = np.concatenate((data['ratio'], data_temp['ratio']), axis=0)
        return data

class Multi_Uproc_Dataset(Dataset):
    def __init__(self, root_dir, crop_per_image=4, crop_size=256, ori=True, iso=6400, params=None, lock_wb=False,
                use_gpu=False, noise_code='prq', dstname='indoor', camera_type='IMX686', mode='train'):
        self.dst1 = Img_Dataset(root_dir.replace(dstname, 'indoor'), crop_per_image, crop_size, ori, iso, params, lock_wb, 
                                use_gpu, noise_code, 'indoor', camera_type, mode)
        self.dst2 = Img_Dataset(root_dir, crop_per_image//4, crop_size, ori, iso, params, lock_wb, 
                                use_gpu, noise_code, dstname, camera_type, mode)
        self.l1 = self.dst1.__len__()
        self.l2 = self.dst2.__len__()

    def __len__(self):
        return self.l1 + self.l2 // 4

    def __getitem__(self, index):
        if index < self.l1:
            data = self.dst1.__getitem__(index)
        else:
            new_idx = index - self.l1
            # new_idx = np.max([(index-self.l1)*4+np.random.randint(4), self.l2-1])
            data = self.dst2.__getitem__(new_idx*4)
            for k in range(1,4):
                data_temp = self.dst2.__getitem__(new_idx*4+k)
                data['lr'] = np.concatenate((data['lr'], data_temp['lr']), axis=0)
                data['hr'] = np.concatenate((data['hr'], data_temp['hr']), axis=0)
                data['ratio'] = np.concatenate((data['ratio'], data_temp['ratio']), axis=0)
        return data

# ELD论文里的亮度对齐，用后处理弥补被严重放大的Black Level Error
class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()
    
    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])               
            else:                                     
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)                    
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape        
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]
        
        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)        
        output = num / den * predict
        # print(num / den)

        return output

if __name__ == '__main__':
    pass