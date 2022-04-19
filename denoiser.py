from data_process import *
from utils import *
from archs import *
from base_trainer import *

class DenoiserParser(BaseParser):
    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/GT_denoiser.yml", type=Path, help="path to config")
        return self.parser.parse_args()

class Denoiser():
    def __init__(self):
        parser = DenoiserParser()
        self.parser = parser.parse()
        with open(self.parser.runfile, 'r', encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.dst = self.args['dst']
        self.arch = self.args['arch']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hostname = socket.gethostname()
        self.model_name = self.args['model_name']
        os.makedirs('./logs', exist_ok=True)
        os.makedirs(self.args['save_path'], exist_ok=True)

        # model
        self.net = globals()[self.arch['name']](self.arch)

        self.dst_test = globals()[self.dst['dataset']](self.dst)
        self.dataloader_test = DataLoader(self.dst_test, batch_size=1, shuffle=False, 
                                num_workers=self.args['num_workers']//2, pin_memory=False)
                                    
        self.net = self.net.to(self.device)
        self.corrector = IlluminanceCorrect()
        torch.backends.cudnn.benchmark = True
        # model log
        self.fake_psnr = AverageMeter('PSNR', ':2f')
        self.logfile = f'./logs/log_{self.model_name}.log'
        log(f'Model Name:\t{self.model_name}', log=self.logfile, notime=True)
        log(f'Architecture:\t{self.arch["name"]}', log=self.logfile, notime=True)
        log(f'Dataset:\t{self.dst["dataset"]}', log=self.logfile, notime=True)
        log(f'CameraType:\t{self.dst["camera_type"]}', log=self.logfile, notime=True)
        log(f'Command:\t{self.dst["command"]}', log=self.logfile, notime=True)
        log(f"Let's use {torch.cuda.device_count()} GPUs!", log=self.logfile, notime=True)
        
        log(f"Using Numpy's CPU Preprocess")
        self.use_gpu = False 

        if torch.cuda.device_count() > 1:
            log("Using PyTorch's nn.DataParallel for multi-gpu...")
            self.multi_gpu = True
            self.net = nn.DataParallel(self.net)
        else:
            self.multi_gpu = False
    
    def predict(self):
        self.net.eval()
        self.fake_psnr.reset()
        plots = []
        with tqdm(total=len(self.dataloader_test)) as t:
            for k, data in enumerate(self.dataloader_test):
                imgs_lr = data['data'].type(torch.FloatTensor).to(self.device)
                wb = data['wb'][0].numpy()
                ccm = data['ccm'][0].numpy()
                name = data['name'][0]
                ratio = data['ratio'][0]

                with torch.no_grad():
                    # # 太大了就用下面这个策略
                    # croped_imgs_lr = self.dst_eval.eval_crop(imgs_lr)
                    # croped_imgs_dn = []
                    # for img_lr in croped_imgs_lr:
                    #     img_dn = self.net(img_lr)
                    #     croped_imgs_dn.append(img_dn)
                    # croped_imgs_dn = torch.cat(croped_imgs_dn)
                    # imgs_lr = self.dst_eval.eval_merge(croped_imgs_lr)
                    # imgs_dn = self.dst_eval.eval_merge(croped_imgs_dn)

                    # 扛得住就pad再crop
                    if imgs_lr.shape[-1] % 16 != 0:
                        p2d = (4,4,4,4)
                        imgs_lr = F.pad(imgs_lr, p2d, mode='reflect')
                        imgs_dn = self.net(imgs_lr)
                        imgs_lr = imgs_lr[..., 4:-4, 4:-4]
                        imgs_dn = imgs_dn[..., 4:-4, 4:-4]
                    else:
                        imgs_dn = self.net(imgs_lr)

                    if self.dst['ori']:
                        imgs_lr = imgs_lr * ratio
                        imgs_dn = imgs_dn * ratio
                    # brighten
                    imgs_lr = torch.clamp(imgs_lr, 0, 1)
                    imgs_dn = torch.clamp(imgs_dn, 0, 1)
                    self.fake_psnr.update(PSNR_Loss(imgs_lr, imgs_dn).item())
                    
                    # # 只有在验证时才考虑这种骚操作
                    # if self.args['brightness_correct'] and epoch < 0:
                    #     imgs_dn = self.corrector(imgs_dn, imgs_lr)

                    # convert raw to rgb
                    if self.args['save_plot']:
                        plots.append(threading.Thread(target=self.multiprocess_plot, args=(imgs_lr, imgs_dn, name, wb, ccm,)))
                        plots[k].start()
                    
                    imgs_dn_np = imgs_dn[0].detach().clone().cpu().numpy()
                    np.save(os.path.join(self.args['save_path'], f'{name}.npy'), imgs_dn_np)

                    t.set_description(f'{name}')
                    t.set_postfix({'PSNR':f"{self.fake_psnr.avg:.2f}"})
                    t.update(1)

        if self.args['save_plot']:
            for i in range(len(self.dataloader_test)):
                plots[i].join()

        log(f"{self.model_name}: Fake_PSNR={self.fake_psnr.avg:.2f}\n", log=f'./logs/log_{self.model_name}.log')
    
    def multiprocess_plot(self, imgs_lr, imgs_dn, name, wb, ccm):
        # inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
        # gt_path = 'images/SID_GT'
        # gtfile = os.path.join(gt_path, "{}.png".format(name))
        os.makedirs(self.args['save_path'], exist_ok=True)
        denoisedfile = os.path.join(self.args['save_path'], "{}_denoised.png".format(name))
        # plt.imsave(gtfile, inputs)
        plt.imsave(denoisedfile, output)
    
    def plot_input(self):
        plots = []
        with tqdm(total=len(self.dataloader_test)) as t:
            for k, data in enumerate(self.dataloader_test):
                imgs_lr = data['data'].type(torch.FloatTensor).to(self.device)
                wb = data['wb'][0].numpy()
                ccm = data['ccm'][0].numpy()
                name = data['name'][0]
                ratio = data['ratio'][0]
                inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
                gt_path = 'images/SID_GT'
                gtfile = os.path.join(gt_path, "{}.png".format(name))
                plt.imsave(gtfile, inputs)
                t.set_description(f'{name}')
                t.update(1)

if __name__ == '__main__':
    denoiser = Denoiser()
    denoiser.plot_input()
    # model_path = denoiser.args['checkpoints']
    # if os.path.exists(model_path):
    #     best_model = torch.load(model_path, map_location=denoiser.device)
    #     denoiser.net = load_weights(denoiser.net, best_model)
    #     denoiser.predict()
        