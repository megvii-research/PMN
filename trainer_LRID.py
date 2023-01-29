import os
import time
import random
from torch.optim import Adam, lr_scheduler
from data_process import *
from utils import *
from archs import *
from losses import *
from base_trainer import *

class IMX686_Trainer(Base_Trainer):
    def __init__(self):
        super().__init__()
        # model
        self.net = globals()[self.arch['name']](self.arch)
        # load weight
        if self.hyper['last_epoch']:    # 不是初始化
            try:
                model_path = os.path.join(f'{self.fast_ckpt}',f'{self.model_name}_best_model.pth')
                if not os.path.exists(model_path):
                    model_path = os.path.join(f'{self.fast_ckpt}',f'{self.model_name}_last_model.pth')
                model = torch.load(model_path, map_location=self.device)
                self.net = load_weights(self.net, model, by_name=True)
            except:
                log('No checkpoint file!!!')
        else:
            log(f'Initializing {self.arch["name"]}...')
            # initialize_weights(self.net)

        self.optimizer = Adam(self.net.parameters(), lr=self.hyper['learning_rate'])
        
        self.infos = None
        if self.mode=='train':
            if isinstance(self.args['dst_train']['dstname'], list):
                self.dst_train = MultiDataset(self.args['dst_train']['dstname'], self.args['dst_train'])
            else:
                self.dst_train = globals()[self.args['dst_train']['dataset']](self.args['dst_train'])
            
            self.dataloader_train = DataLoader(self.dst_train, batch_size=self.hyper['batch_size'], worker_init_fn=self.worker_init_fn,
                                    shuffle=True, num_workers=self.args['num_workers'], pin_memory=False, timeout=60.)
            self.change_eval_dst('eval')
            self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)

        # Choose Learning Rate
        self.lr_lambda = self.get_lr_lambda_func()
        self.scheduler = LambdaScheduler(self.optimizer, self.lr_lambda)

        self.net = self.net.to(self.device)
        self.loss = Unet_Loss()
        self.corrector = IlluminanceCorrect()
        torch.backends.cudnn.benchmark = True
        # model log
        self.best_psnr = self.hyper['best_psnr'] if 'best_psnr' in self.hyper else 0
        last_eval_epoch = self.hyper['last_epoch'] // self.hyper['plot_freq']
        self.train_psnr = AverageMeter('PSNR', ':2f', last_epoch=self.hyper['last_epoch'])
        self.eval_psnr = AverageMeter('PSNR', ':2f', last_epoch=last_eval_epoch)
        self.eval_ssim = AverageMeter('SSIM', ':4f')
        self.eval_psnr_lr = AverageMeter('PSNR', ':2f')
        self.eval_ssim_lr = AverageMeter('SSIM', ':4f')
        self.eval_psnr_dn = AverageMeter('PSNR', ':2f')
        self.eval_ssim_dn = AverageMeter('SSIM', ':4f')
        self.logfile = f'./logs/log_{self.model_name}.log'
        log(f'Model Name:\t{self.model_name}', log=self.logfile, notime=True)
        log(f'Architecture:\t{self.arch["name"]}', log=self.logfile, notime=True)
        log(f'TrainDataset:\t{self.args["dst_train"]["dataset"]}', log=self.logfile, notime=True)
        log(f'EvalDataset:\t{self.args["dst_eval"]["dataset"]}', log=self.logfile, notime=True)
        log(f'CameraType:\t{self.dst["camera_type"]}', log=self.logfile, notime=True)
        log(f'NoiseType:\t{self.dst["noise_code"]}', log=self.logfile, notime=True)
        log(f'num_channels:\t{self.arch["nf"]}', log=self.logfile, notime=True)
        log(f'BatchSize:\t{self.hyper["batch_size"]}', log=self.logfile, notime=True)
        log(f'PatchSize:\t{self.dst["patch_size"]}', log=self.logfile, notime=True)
        log(f'LearningRate:\t{self.hyper["learning_rate"]}', log=self.logfile, notime=True)
        log(f'Epoch:\t\t{self.hyper["stop_epoch"]}', log=self.logfile, notime=True)
        log(f'num_workers:\t{self.args["num_workers"]}', log=self.logfile, notime=True)
        log(f'Command:\t{self.dst["command"]}', log=self.logfile, notime=True)
        log(f"Let's use {torch.cuda.device_count()} GPUs!", log=self.logfile, notime=True)
        # self.device != torch.device(type='cpu') 
        if 'gpu_preprocess' in self.dst and self.dst['gpu_preprocess']:
            log("Using PyTorch's GPU Preprocess...")
            self.use_gpu = True
        else:
            log(f"Using Numpy's CPU Preprocess")
            self.use_gpu = False 

        if torch.cuda.device_count() > 1:
            log("Using PyTorch's nn.DataParallel for multi-gpu...")
            self.multi_gpu = True
            self.net = nn.DataParallel(self.net)
        else:
            self.multi_gpu = False
        self.ratiofix = True if 'ratiofix' in self.dst['command'] else False
    
    def change_eval_dst(self, mode='eval'):
        self.dst = self.args[f'dst_{mode}']
        if isinstance(self.dst['dstname'], list):
            self.dstname = self.dst['dstname'][0]
            self.dst_eval = MultiDataset(self.dst['dstname'], self.dst)
        else:
            self.dstname = self.dst['dstname']
            self.dst_eval = globals()[self.dst['dataset']](self.dst)

        self.dst_eval = globals()[self.dst['dataset']](self.dst)
        self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)
        # self.cache_dir = f'/data/cache/{self.dst["dstname"]}'

    def train(self):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        for epoch in range(self.hyper['last_epoch']+1, self.hyper['stop_epoch']+1):
            # log init
            self.net.train()
            self.train_psnr.reset()
            runtime = {'preprocess':0, 'dataloader':0, 'net':0, 'bp':0, 'metric':0, 'total':1e-9}
            time_points = [0] * 10
            time_points[0] = time.time()

            t = tqdm(self.dataloader_train)
            try:
                for k, data in enumerate(self.dataloader_train):
                    runtime['dataloader'] += timestamp(time_points, 1)
                    # Preprocess
                    imgs_lr, imgs_hr, ratio = self.preprocess(data, mode='train', preprocess=True)
                    runtime['preprocess'] += timestamp(time_points, 2)
                    
                    # 训练
                    self.optimizer.zero_grad()
                    pred = self.net(imgs_lr)
                    runtime['net'] += timestamp(time_points, 3)
                    # 如果去噪没提前线性提亮，算loss的时候提亮上去
                    if self.dst['ori'] is True:
                        pred = pred * ratio
                    loss = self.loss(pred.clamp(0,1), imgs_hr)
                    loss.backward()
                    self.optimizer.step()
                    runtime['bp'] += timestamp(time_points, 4)

                    # 更新tqdm的参数
                    with torch.no_grad():
                        if self.arch['use_dpsv']: 
                            pred = pred[0]
                        pred = torch.clamp(pred, 0, 1)
                        imgs_hr = torch.clamp(imgs_hr, 0, 1)
                        psnr = PSNR_Loss(pred, imgs_hr)
                        self.train_psnr.update(psnr.item())
                    
                    runtime['total'] = runtime['preprocess']+runtime['dataloader']+runtime['net']+runtime['bp']
                    t.set_description(f'Epoch {epoch}')
                    t.set_postfix({'lr':f"{lr:.2e}", 'PSNR':f"{self.train_psnr.avg:.2f}",
                                    'loader':f"{100*runtime['dataloader']/runtime['total']:.1f}%",
                                    'process':f"{100*runtime['preprocess']/runtime['total']:.1f}%",
                                    'net':f"{100*runtime['net']/runtime['total']:.1f}%",
                                    'bp':f"{100*runtime['bp']/runtime['total']:.1f}%",})
                    t.update(1)
                    time_points[0] = time.time()
                    # break
            except RuntimeError as e:
                log(str(e), log=f'./logs/log_{self.model_name}.log')
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                continue

            # 更新学习率
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # 存储模型
            if epoch % self.hyper['save_freq'] == 0:
                model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                epoch_id = (epoch+self.hyper['plot_freq']-1) // self.hyper['plot_freq'] * self.hyper['plot_freq']
                save_path = os.path.join(self.model_dir, '%s_e%04d.pth'% (self.model_name, epoch_id))
                torch.save(model_dict, save_path)
            
            # 输出过程量，随时看
            savefile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.jpg')
            logfile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.pkl')
            self.train_psnr.plot_history(savefile=savefile, logfile=logfile)
            # if epoch % self.hyper['plot_freq'] == 0:
            wb = data['wb'][0].numpy()
            if self.dst['ori'] is True:
                imgs_lr = imgs_lr * ratio
                pred = pred
                imgs_hr = imgs_hr# * ratio
            
            if self.save_plot:
                inputs = imgs_lr[0].detach().cpu().numpy().clip(0,1)
                output = pred[0].detach().cpu().numpy()
                target = imgs_hr[0].detach().cpu().numpy()
                temp_img = np.concatenate((inputs, output, target),axis=2)[:3]
                temp_img[0] = temp_img[0] * wb[0]
                temp_img[2] = temp_img[2] * wb[2]
                filename = os.path.join(self.sample_dir, 'temp', f'temp_{epoch//10*10:04d}.png')
                temp_img = temp_img.transpose(1,2,0)[:,:,::-1] ** (1/2.2)
                cv2.imwrite(filename, np.uint8(temp_img*255))

            # fast eval
            if epoch % self.hyper['plot_freq'] == 0:
                log(f"learning_rate: {lr:.3e}")
                self.dst_eval.fast_eval(on=True)
                # ratio_list = self.dst_eval.args['ratio_list']
                # for dgain in ratio_list:
                #     log(f'{self.dstname} Datasets: Dgain={dgain}',log=f'./logs/log_{self.model_name}.log')
                #     trainer.dst_eval.change_eval_ratio(ratio=dgain)
                self.eval(epoch=epoch)
                self.dst_eval.fast_eval(on=False)
                model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
            
            # reload best model each period
            num_of_epochs = self.hyper['stop_epoch'] - self.hyper['last_epoch']
            T = self.hyper['T'] if 'T' in self.hyper else 1 
            period = num_of_epochs//T
            if (self.hyper['last_epoch']+epoch) % period == 0:
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
                if os.path.exists(model_path):
                    model = torch.load(model_path, map_location=self.device)
                    self.net = load_weights(self.net, model, by_name=True)
                    log(f'Successfully reload best model (Eval PSNR:{self.best_psnr})',
                        log=f'./logs/log_{self.model_name}.log')

    def eval(self, epoch=-1):
        self.net.eval()
        self.eval_psnr.reset()
        self.eval_ssim.reset()
        self.eval_psnr_lr.reset()
        self.eval_psnr_dn.reset()
        self.eval_ssim_lr.reset()
        self.eval_ssim_dn.reset()
        # record every metric
        metrics = {}
        metrics_path = f'./metrics/{self.model_name}_metrics.pkl'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pkl.load(f)
        # multiprocess
        if epoch > 0:
            pool = []
        else:
            pool = ProcessPoolExecutor(max_workers=max(4, self.args['num_workers']))
        task_list = []
        save_plot = self.save_plot #if self.multi_gpu is False else False
        with tqdm(total=len(self.dataloader_eval)) as t:
            for k, data in enumerate(self.dataloader_eval):
                # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                imgs_lr, imgs_hr, ratio = self.preprocess(data, mode='eval', preprocess=False)
                wb = data['wb'][0].numpy()
                ccm = data['ccm'][0].numpy()
                name = data['name'][0]
                ISO = data['ISO'].item()
                exp = data['ExposureTime'].item()
                # print(ISO)

                with torch.no_grad():
                    # 扛得住就pad再crop
                    if imgs_lr.shape[-1] % 16 != 0:
                        p2d = (4,4,4,4)
                        imgs_lr = F.pad(imgs_lr, p2d, mode='reflect')
                        imgs_dn = self.net(imgs_lr)
                        imgs_lr = imgs_lr[..., 4:-4, 4:-4]
                        imgs_dn = imgs_dn[..., 4:-4, 4:-4]
                    else:
                        imgs_dn = self.net(imgs_lr)
                    
                    # brighten
                    if self.dst['ori']:
                        imgs_lr = imgs_lr * ratio
                        imgs_dn = imgs_dn * ratio
                    imgs_lr = torch.clamp(imgs_lr, 0, 1)
                    imgs_dn = torch.clamp(imgs_dn, 0, 1)

                    # convert raw to rgb
                    output = tensor2im(imgs_dn)
                    target = tensor2im(imgs_hr)
                    res = quality_assess(output, target, data_range=255)
                    raw_metrics = [res['PSNR'], res['SSIM']]
                    self.eval_psnr.update(res['PSNR'])
                    self.eval_ssim.update(res['SSIM'])
                    metrics[name] = raw_metrics
                    
                    if save_plot:
                        if self.infos is None:
                            inputs = tensor2im(imgs_lr)
                            res_in = quality_assess(inputs, target, data_range=255)
                            raw_metrics = [res_in['PSNR'], res_in['SSIM']] + raw_metrics
                        else:
                            raw_metrics = [self.infos[k]['PSNR_raw'], self.infos[k]['SSIM_raw']] + raw_metrics
                        if epoch > 0:
                            pool.append(threading.Thread(target=self.multiprocess_plot, args=(imgs_lr, imgs_dn, imgs_hr, 
                                    wb, ccm, name, save_plot, epoch, raw_metrics, k)))
                            pool[k].start()
                        else:
                            infos = self.infos[k] if self.infos is not None else None
                            # 多进程
                            if infos is None:
                                inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
                                target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
                            else:
                                inputs = np.load(infos['path_npy_in'])
                                target = np.load(infos['path_npy_gt'])
                            output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
                            task_list.append(
                                pool.submit(plot_sample, inputs, output, target, 
                                    filename=name, save_plot=save_plot, epoch=epoch,
                                    model_name=self.model_name, save_path=self.sample_dir,
                                    res=raw_metrics
                                    )
                                )

                    t.set_description(f'{name}')
                    t.set_postfix({'PSNR':f"{self.eval_psnr.avg:.2f}"})
                    t.update(1)

        if save_plot:
            if epoch > 0:
                for i in range(len(pool)):
                    pool[i].join()
            else:
                pool.shutdown(wait=True)
                for task in as_completed(task_list):
                    psnr, ssim, _ = task.result()
                    self.eval_psnr_lr.update(psnr[0])
                    self.eval_psnr_dn.update(psnr[1])
                    self.eval_ssim_lr.update(ssim[0])
                    self.eval_ssim_dn.update(ssim[1])
        else:
            self.eval_psnr_dn = self.eval_psnr
            self.eval_ssim_dn = self.eval_ssim

        # 超过最好记录才保存
        if self.eval_psnr_dn.avg >= self.best_psnr and epoch > 0:
            self.best_psnr = self.eval_psnr_dn.avg
            log(f"Best PSNR is {self.best_psnr} now!!")
            model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
            torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_best_model.pth')

        log(f"Epoch {epoch}: PSNR={self.eval_psnr.avg:.2f}\n"
            +f"psnrs_lr={self.eval_psnr_lr.avg:.2f}, psnrs_dn={self.eval_psnr_dn.avg:.2f}"
            +f"\nssims_lr={self.eval_ssim_lr.avg:.4f}, ssims_dn={self.eval_ssim_dn.avg:.4f}",
            log=f'./logs/log_{self.model_name}.log')
        if epoch < 0:
            with open(metrics_path, 'wb') as f:
                pkl.dump(metrics, f)
        savefile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.jpg')
        logfile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.pkl')
        if epoch > 0:
            self.eval_psnr.plot_history(savefile=savefile, logfile=logfile)
        del pool
        plt.close('all')
        gc.collect()
        return metrics
    
    def multiprocess_plot(self, imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, raw_metrics, k):
        # if self.infos is None:
        inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
        # else:
        #     inputs = np.load(self.infos[k]['path_npy_in'])
        #     target = np.load(self.infos[k]['path_npy_gt'])
        output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
        
        psnr, ssim, _ = plot_sample(inputs, output, target, 
                        filename=name, 
                        save_plot=save_plot, epoch=epoch,
                        model_name=self.model_name,
                        save_path=self.sample_dir,
                        res=raw_metrics)
        self.eval_psnr_lr.update(psnr[0])
        self.eval_psnr_dn.update(psnr[1])
        self.eval_ssim_lr.update(ssim[0])
        self.eval_ssim_dn.update(ssim[1])

    def predict(self, raw, name='ds'):
        self.net.eval()
        img_lr = raw2bayer(raw+self.dst["bl"])[None, ...]
        img_lr = torch.from_numpy(img_lr)
        img_lr = img_lr.type(torch.FloatTensor).to(self.device)
        with torch.no_grad():
            croped_imgs_lr = self.dst_eval.eval_crop(img_lr)
            croped_imgs_dn = []
            for img_lr in tqdm(croped_imgs_lr):
                img_dn = self.net(img_lr)
                croped_imgs_dn.append(img_dn)
            croped_imgs_dn = torch.cat(croped_imgs_dn)
            img_dn = self.dst_eval.eval_merge(croped_imgs_dn)
            img_dn = img_dn
            img_dn = img_dn[0].detach().cpu().numpy()
        np.save(f'{name}.npy', img_dn)
    
    def preprocess(self, data, mode='train', preprocess=True):
        # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
        imgs_hr = tensor_dim5to4(data['hr']).type(torch.FloatTensor).to(self.device)
        imgs_lr = tensor_dim5to4(data['lr']).type(torch.FloatTensor).to(self.device)
        # self.use_gpu = True
        dst = self.dst_train if mode=='train' else self.dst_eval

        if self.use_gpu and mode=='train' and preprocess:
            b = imgs_lr.shape[0]
            if self.args['dst_train']['dataset'] == 'IMX686_Mix_Dataset':
                data['ratio'] = data['ratio'].view(-1).type(torch.FloatTensor).to(self.device)
                aug_r, aug_g, aug_b = get_aug_param_torch(data, b=b, command=self.dst['command'], 
                                                        camera_type=self.dst['camera_type'])
                aug_wbs = torch.stack((aug_r, aug_g, aug_b, aug_g), dim=1)
                data['rgb_gain'] = torch.ones(b) * (aug_g + 1)
                data['wb'] = data['wb'][0].repeat(b, 1)
                for i in range(b):
                    aug_wb = aug_wbs[i].numpy()
                    if data['black_lr'][0]: aug_wb += 1
                    dgain = data['ratio'][i]
                    imgs_lr[i] = imgs_lr[i] if self.dst['ori'] else imgs_lr[i] * dgain
                    if np.abs(aug_wb).max() != 0:
                        data['wb'][i] *= (1+aug_wb[1]) / (1+aug_wb)
                        iso = data['ISO'][i//self.dst['crop_per_image']].item()
                        dn, dy = SNA_torch(imgs_hr[i], aug_wb, iso=iso, ratio=dgain, black_lr=data['black_lr'][0],
                            camera_type=self.dst['camera_type'], ori=self.dst['ori'])
                        imgs_lr[i] = imgs_lr[i] + dn 
                        imgs_hr[i] = imgs_hr[i] + dy
            
            elif self.args['dst_train']['dataset'] == 'IMX686_SFRN_Raw_Dataset':
                data['ratio'] = data['ratio'].view(-1).type(torch.FloatTensor).to(self.device)
                for i in range(b):
                    aug_wb = np.ones(4, dtype=np.float32)
                    dgain = data['ratio'][i]
                    imgs_lr[i] = imgs_lr[i] if self.dst['ori'] else imgs_lr[i] * dgain
                    if np.abs(aug_wb).max() != 0:
                        iso = data['ISO'][0].item()
                        dn, _ = SNA_torch(imgs_hr[i], aug_wb, iso=iso, ratio=dgain, black_lr=True,
                            camera_type=self.dst['camera_type'], ori=self.dst['ori'])
                        imgs_lr[i] = imgs_lr[i] + dn 
                
            elif self.args['dst_train']['dataset'] == 'IMX686_Raw_Dataset':
                data['ratio'] = torch.ones(b, device=self.device)
                # 人工加噪声，注意，这里统一时间的视频应该共享相同的噪声参数！！
                for i in range(b):
                    # 单ISO的模型
                    if isinstance(dst, MultiDataset):
                        noise_param = dst.dsts[0].noiseparam[6400].copy()
                    else:
                        noise_param = dst.noiseparam[6400].copy()
                    noise_param['K'] = noise_param['K'] * (1 + np.random.uniform(low=-0.01, high=+0.01))
                    noise_param['ratio'] = np.random.uniform(low=1, high=16)
                    # numpy -> torch
                    for key in noise_param:
                        if torch.is_tensor(noise_param[key]) is False:
                            noise_param[key] = torch.from_numpy(np.array(noise_param[key], np.float32))
                        noise_param[key] = noise_param[key].to(self.device)
                    data['ratio'][i] = noise_param['ratio']
                    # generate noise
                    imgs_lr[i] = generate_noisy_torch(imgs_lr[i], param=noise_param,
                                noise_code=self.dst['noise_code'], ori=self.dst['ori'], clip=self.dst['clip'])
        else: # mode == 'eval'
            pass
        
        ratio = data['ratio'].type(torch.FloatTensor).to(self.device)
        ratio = ratio.view(-1,1,1,1)
        if 'rgb_gain' in data:
            data['rgb_gain'] = data['rgb_gain'].type(torch.FloatTensor).to(self.device).view_as(ratio)
        
        if self.dst['clip']:
            lb = -100 if 'HB' in self.dst['command'] else 0
            imgs_lr = imgs_lr.clamp(lb, 1)
            imgs_hr = imgs_hr.clamp(0, 1)
        return imgs_lr, imgs_hr, ratio

def MultiProcessPlot(imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, 
                    raw_metrics, infos, model_name, sample_dir):
    if infos is None:
        inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
    else:
        inputs = np.load(infos['path_npy_in'])
        target = np.load(infos['path_npy_gt'])
    output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
    
    psnr, ssim, _ = plot_sample(inputs, output, target, 
                    filename=name, 
                    save_plot=save_plot, epoch=epoch,
                    model_name=model_name,
                    save_path=sample_dir,
                    res=raw_metrics)
    return psnr, ssim

if __name__ == '__main__':
    trainer = IMX686_Trainer()
    if trainer.mode == 'train':
        trainer.train()
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        trainer.mode = 'evaltest'
    # best_model
    best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_best_model.pth')
    if os.path.exists(best_model_path) is False: 
        best_model_path = os.path.join(f'{trainer.fast_ckpt}',f'{trainer.model_name}_last_model.pth')
    best_model = torch.load(best_model_path, map_location=trainer.device)
    trainer.net = load_weights(trainer.net, best_model, multi_gpu=trainer.multi_gpu)
    # indoor_x5
    if 'eval' in trainer.mode:
        trainer.change_eval_dst('eval')
        ratio_list = trainer.dst_eval.args['ratio_list']
        for dgain in ratio_list[:]:
            log(f'{trainer.dstname} Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            metrics = trainer.eval(-1)
    # outdoor_x5
    if 'test' in trainer.mode:
        trainer.change_eval_dst('test')
        ratio_list = trainer.dst_eval.args['ratio_list']
        for dgain in ratio_list[:]:
            log(f'{trainer.dstname} Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            metrics = trainer.eval(-1)
    log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')