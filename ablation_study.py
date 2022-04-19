from trainer_SID import *

class Ablation_Study(SID_Trainer):
    def ablation_eval(self, epoch=-1):
        self.net.eval()
        self.eval_psnr_dn.reset()
        self.eval_ssim_dn.reset()
        save_plot = False
        with tqdm(total=len(self.dataloader_eval)) as t:
            for k, data in enumerate(self.dataloader_eval):
                # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                imgs_lr, imgs_hr, ratio = self.preprocess(data, mode='eval', preprocess=False)
                wb = data['wb'][0].numpy()
                ccm = data['ccm'][0].numpy()
                name = data['name'][0]
                ISO = data['ISO'].item()

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
                    
                    # align to ELD
                    if self.args['brightness_correct']:
                        imgs_dn = self.corrector(imgs_dn, imgs_hr)
                    # convert raw to rgb
                    output = tensor2im(imgs_dn)
                    target = tensor2im(imgs_hr)
                    res = quality_assess(output, target, data_range=255)
                    self.eval_psnr_dn.update(res['PSNR'])
                    self.eval_ssim_dn.update(res['SSIM'])

                    t.set_description(f'{name}')
                    t.set_postfix_str(f'PSNR:{self.eval_psnr_dn.avg:.2f}, SSIM:{self.eval_ssim_dn.avg:.4f}')
                    t.update(1)
        psnr, ssim = self.eval_psnr_dn.avg, self.eval_ssim_dn.avg
        log(f"Num.{epoch}: psnrs_dn={psnr:.2f}, ssims_dn={ssim:.4f}",
            log=f'./logs/log_{self.model_name}.log')
        print(psnr, ssim)
        return psnr, ssim

if __name__ == '__main__':
    trainer = Ablation_Study()
    checkpoint = [1,4,9,16,25,36,49,64,81,100,150,200,250,300,350,400]
    # checkpoint = ['s2','s4','s6','s8']
    best_model_path = os.path.join(f'./checkpoints/{trainer.model_name}_best_model.pth')
    best_model = torch.load(best_model_path, map_location=trainer.device)
    trainer.net = load_weights(trainer.net, best_model)
    psnrs = []
    ssims = []
    if 'eval' in trainer.mode:
        # ELD
        dgain = 200
        trainer.change_eval_dst('eval')
        log(f'ELD Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
        for i, num in enumerate(checkpoint):
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            trainer.dst_eval.ratio_list=[dgain]
            trainer.dst_eval.recheck_length()
            # trainer.dst_eval.remap_darkshading(num)
            trainer.dst_eval.remap_BLE(num)
            psnr, ssim = trainer.ablation_eval(num)
            psnrs.append(psnr)
            ssims.append(ssim)
    print(psnrs)
    print(ssims)

    psnrs = []
    ssims = []
    if 'test' in trainer.mode:
        # SID
        dgain = 300
        trainer.change_eval_dst('test')
        log(f'SID Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
        for i, num in enumerate(checkpoint):
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            # trainer.dst_eval.remap_darkshading(num)
            trainer.dst_eval.remap_BLE(num)
            psnr, ssim = trainer.ablation_eval(num)
            psnrs.append(psnr)
            ssims.append(ssim)
            print(psnrs)
            print(ssims)
    # print(psnrs)
    # print(ssims)
