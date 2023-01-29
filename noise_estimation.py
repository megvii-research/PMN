from utils import *
from sklearn.linear_model import LinearRegression

class NoiseEstimation():
    def __init__(self, args):
        self.args = args
        self.bias_dir = args['bias_dir'], iso=args['ISO'], dst_dir=args['dst_dir']
        self.root_dir = args['bias_dir']
        self.iso = args['ISO']
        self.wp, self.bl = args['wp'], args['bl']
        self.src_dir = os.path.join(self.root_dir, str(iso))
        self.H, self.W = args['H'], args['W']
        self.dst_dir = args['dst_dir']
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(f'{self.dst_dir}/darkshading-iso-{self.iso}.npy'):
            self.darkshading, self.bad_points = get_darkshading(save=True)
        else:
            self.darkshading = np.load(f'{self.dst_dir}/darkshading-iso-{self.iso}.npy')
            self.bad_points = np.load(f'{self.dst_dir}/bpc-iso-{iso}.npy')

    def noise_level(self):
        sigReads = []
        uReads = []
        r2s = []
        # self.bpc_map = np.zeros_like(self.darkshading)
        # for p in self.bad_points:
        #     self.bpc_map[p[0],p[1]] = 1
        for filename in tqdm(os.listdir(self.src_dir)):
            if filename[-4:] != '.dng': continue
            filepath = os.path.join(self.src_dir, filename)
            raw = rawpy.imread(filepath)
            # rawpy.enhance.repair_bad_pixels(raw, self.bad_points, method='median')
            im = raw.raw_image_visible.astype(np.float32) - self.bl - self.darkshading
            im = repair_bad_pixels(im, self.bad_points)
            _, (sigRead, uRead, r2) = scipy.stats.probplot(im.reshape(-1), rvalue=True)
            sigReads.append(sigRead)
            uReads.append(uRead)
            r2s.append(r2)
        # sigReads = sigReads[:194]
        x = np.array(list(range(len(sigReads))))
        sigReads = np.array(sigReads)
        u = sigReads.mean()
        sig = sigReads.std()
        print(u,sig)
        plt.plot(x, sigReads, label=f'u={u:.2e}, sig={sig:.4e}')
        plt.hlines(u, colors='red', xmin=0, xmax=len(sigReads))
        plt.hlines(u+sig, colors='orange', xmin=0, xmax=len(sigReads))
        plt.hlines(u-sig, colors='orange', xmin=0, xmax=len(sigReads))
        # plt.show()
        print()
    
    def noise_param(self):
        # if os.path.exists(f'{self.dst_dir}/noiseparam-iso-{self.iso}.h5'):
        #     return
        r = 400
        raw_paths = [os.path.join(self.src_dir, name) for name in os.listdir(self.src_dir) if name[-4:]=='.dng']
        print(f"nums of raw: {len(raw_paths)}")
        # Read
        sigmaRead = np.zeros((4, len(raw_paths)), dtype=np.float32)
        meanRead = np.zeros((4, len(raw_paths)), dtype=np.float32)
        r2Read = np.zeros((4, len(raw_paths)), dtype=np.float32)
        # R
        sigmaR = np.zeros((4, len(raw_paths)), dtype=np.float32)
        meanR = np.zeros((4, len(raw_paths)), dtype=np.float32)
        r2R = np.zeros((4, len(raw_paths)), dtype=np.float32)
        # TL
        sigmaTL = np.zeros((4, len(raw_paths)), dtype=np.float32)
        meanTL = np.zeros((4, len(raw_paths)), dtype=np.float32)
        r2TL = np.zeros((4, len(raw_paths)), dtype=np.float32)
        lamda = np.zeros((4, len(raw_paths)), dtype=np.float32)
        # Gs
        sigmaGs = np.zeros((4, len(raw_paths)), dtype=np.float32)
        meanGs = np.zeros((4, len(raw_paths)), dtype=np.float32)
        r2Gs = np.zeros((4, len(raw_paths)), dtype=np.float32)

        pbar = tqdm(range(len(raw_paths)))
        for i in pbar:
            raw = rawpy.imread(raw_paths[i]).raw_image_visible.astype(np.float32) - self.darkshading - self.bl
            packed_raw = bayer2rggb(raw)
            packed_raw = packed_raw[self.H//2-r:self.H//2+r,self.W//2-r:self.W//2+r]
            for c in range(4):
                img = packed_raw[:,:,c]
                # Compute σR
                _, (sigRead, uRead, rRead) = scipy.stats.probplot(img.reshape(-1), rvalue=True)
                sigmaRead[c][i] = sigRead
                meanRead[c][i] = uRead
                r2Read[c][i] = rRead**2
                # print(f'uRead={uRead:.4f}, sigRead={sigRead:.4f}, read2={rRead**2:.4f}')
                # img = pack_center_crop(pack_raw_bayer(raw), size=1000)[1]
                row_all = np.mean(img, axis=1)
                _, (sigR, uR, rR) = scipy.stats.probplot(row_all, rvalue=True)
                sigmaR[c][i] = sigR
                meanR[c][i] = uR
                r2R[c][i] = rR**2
                # print(f'uR={uR:.4f}, sigR={sigR:.4f}, r2={rR**2:.4f}')

                # Compute σTL
                img = img - row_all.reshape(-1,1)
                X = img.reshape(-1)
                lam = scipy.stats.ppcc_max(X)
                lamda[c][i] = lam
                _, (sigGs, uGs, rGs) = scipy.stats.probplot(X, rvalue=True)
                # print(f'uGs={uGs:.4f}, sigGs={sigGs:.4f}, rGs2={rGs**2:.4f}')
                _, (sigTL, uTL, rTL) = scipy.stats.probplot(X, dist=scipy.stats.tukeylambda(lam), rvalue=True)
                sigmaTL[c][i] = sigTL
                meanTL[c][i] = uTL
                r2TL[c][i] = rTL**2
                # print(f'λ={lam:.3f}, uTL={uTL:.4f}, sigTL={sigTL:.4f}, rTL2={rTL**2:.4f}')

                _, (sigGs, uGs, rGs) = scipy.stats.probplot(X, rvalue=True)
                sigmaGs[c][i] = sigGs
                meanGs[c][i] = uGs
                r2Gs[c][i] = rGs**2
            pbar.set_description(f"Raw {i:03d}")
            #设置进度条右边显示的信息
            pbar.set_postfix_str(
                f'uRead={meanRead[:, :i+1].mean():.4f}, sigRead={sigmaRead[:, :i+1].mean():.4f}, lam={lamda[:, :i+1].mean()**2:.4f}, '+ 
                f'rR2={r2R[:, :i+1].mean():.4f}, rGs2={r2Gs[:, :i+1].mean():.4f}, rTL2={r2TL[:, :i+1].mean():.4f}'
                )
        param = {
            'lam':lamda, 'wp':self.wp, 'bl':self.bl,
            'sigmaRead':sigmaRead, 'meanRead':meanRead, 'r2Gs':r2Read,
            'sigmaR':sigmaR, 'meanR':meanR, 'r2R':r2R,
            'sigmaTL':sigmaTL, 'meanTL':meanTL, 'r2TL':r2TL,
            'sigmaGs':sigmaGs, 'meanGs':meanGs, 'r2Gs':r2Gs,
        }
        with h5py.File(f'{self.dst_dir}/noiseparam-iso-{self.iso}.h5', 'w') as f:
            for key in param:
                f.create_dataset(key, data=param[key])
        # with open(f'{self.dst_dir}/noiseparam-iso-{self.iso}.pkl','wb') as f:
        #     pickle.dump(param, f)
    
    def analysis_data(self, param_dir='resources', title='Noise Profile',
                    K_ISO=(0.76/800,0), dual_ISO=False, save_dir=None):
        fig = plt.figure(figsize=(20,8))
        fig2 = plt.figure(figsize=(20,8))
        fig.suptitle(title)
        fig2.suptitle(title)

        axR = fig2.add_subplot(1,3,1)
        axTL = fig2.add_subplot(2,3,2)
        axlam = fig2.add_subplot(2,3,3)
        axGs = fig2.add_subplot(1,3,2)
        axMean = fig2.add_subplot(1,3,3)
        
        params = []
        isos = [int(iso) for iso in sorted(os.listdir(self.bias_dir), key=lambda x: int(x)) if int(iso)>1600][:3]
        for iso in isos:
            param = {'ISO': iso}
            f = h5py.File(os.path.join(param_dir, f'noiseparam-iso-{iso}.h5'), 'r')
            for key in f:
                param[key] = np.array(f[key])
                if len(param[key].shape)>1: 
                    param[key] = param[key].mean(axis=0)
            params.append(param)
                
        axR.set_title('sigmaR | ISO')
        axTL.set_title('sigmaTL | ISO')
        axlam.set_title('lam | ISO')
        axGs.set_title('sigmaGs | ISO')
        axMean.set_title('uRead | ISO')

        # axR.scatter(isos, params[:]['sigmaR'].mean(axis=-1))
        # axGs.scatter(isos, param['sigmaGs'].mean(axis=-1))
        # axMean.scatter(isos, param['meanRead'].mean(axis=-1))
    
        axsigR = fig.add_subplot(2,3,1)
        axsigTL = fig.add_subplot(2,3,2)
        axlam = fig.add_subplot(2,3,3)
        axsigGs = fig.add_subplot(2,3,4)
        axMean = fig.add_subplot(2,3,5)
        axRead = fig.add_subplot(2,3,6)
        axsigTL.set_title('log(sigTL) | log(K)')
        axlam.set_title('lam | ISO')
        axsigR.set_title('log(sigR) | log(K)')
        axsigGs.set_title('log(sigGs) | log(K)')
        axMean.set_title('log(uRead) | log(K)')
        axRead.set_title('log(sigRead) | log(K)')
        # axsigR.set_title('sigR | K')
        # axsigTL.set_title('sigTL | K')
        # axlam.set_title('lam | K')
        # axsigGs.set_title('sigGs | K')
        # axMean.set_title('uRead | K')
        # axRead.set_title('sigRead | K')

        iso=[]
        iso_points = []
        sigR=[]
        sigTL=[]
        sigRead=[]
        uRead=[]
        lam=[]
        sigGs=[]
        split_point = [0]
        cnt = 0

        # print('params',params)

        for param in params:
            for i in range(len(param['sigmaR'])):
                point_iso = param['ISO']*K_ISO[0]+K_ISO[1]
                iso.append(point_iso)
                sigR.append(param['sigmaR'][i])
                sigTL.append(param['sigmaTL'][i])
                sigGs.append(param['sigmaGs'][i])
                lam.append(param['lam'][i])
                sigRead.append(param['sigmaRead'][i])
            iso_points.append(point_iso)
            uRead.append(param['meanRead'].std())

        iso = np.array(iso)
        for i, point_iso in enumerate(iso):
            if iso[split_point[cnt]] != iso[i]:
                print(split_point[cnt], iso[split_point[cnt]])
                cnt += 1
                split_point.append(i)
        split_point.append(len(iso))

        axsigR, dataR = self.regr_plot(iso, sigR, ax=axsigR, c1='red', c2='orange', log=True, label=True)
        axsigTL, dataTL = self.regr_plot(iso, sigTL, ax=axsigTL, c1='red', c2='orange', log=True, label=True)
        axsigRead, dataRead = self.regr_plot(iso, sigRead, ax=axRead, c1='red', c2='orange', log=True, label=True)
        axuRead, datauRead = self.regr_plot(iso_points, uRead, ax=axMean, c1='red', c2='orange', log=False, label=False)
        axsigGs, dataGs = self.regr_plot(iso, sigGs, ax=axsigGs, c1='red', c2='orange', log=True, label=True)
        axlam, datalam = self.regr_plot((iso-K_ISO[1])/K_ISO[0], lam, ax=axlam, log=False, c1='red', c2='orange', label=True)

        data = {
            'ISO_start':params[0]['ISO'], 'ISO_end':params[-1]['ISO'], 'q':1/self.wp, 'wp':self.wp, 'bl':self.bl,
            'Kmin':np.log(np.min(iso)), 'Kmax':np.log(np.max(iso)), 'lam':np.mean(lam),
            'sigTLk':dataTL['k'], 'sigTLb':dataTL['b'], 'sigTLsig':dataTL['sig'],
            'sigGsk':dataGs['k'], 'sigGsb':dataGs['b'], 'sigGssig':dataGs['sig'],
            'sigRk':dataR['k'], 'sigRb':dataR['b'], 'sigRsig':dataR['sig'],
            'sigReadk':dataRead['k'], 'sigReadb':dataRead['b'], 'sigReadsig':dataRead['sig'],
            'uReadk':datauRead['k'], 'uReadb':datauRead['b'], 'uReadsig':datauRead['sig'],
        }
        self.save_params(data, save_dir)
        # 分段拟合
        for i in range(len(split_point)-2):
            s = split_point[i]
            e = split_point[i+2]
            iso_part = np.array(iso[s:e])
            sigR_part = sigR[s:e]
            sigTL_part = sigTL[s:e]
            sigRead_part = sigRead[s:e]
            uRead_part = uRead[i:i+2]
            sigGs_part = sigGs[s:e]
            lam_part = lam[s:e]
            axsigR, dataR = self.regr_plot(iso_part, sigR_part, ax=axsigR, c1='blue', c2='green',log=True)
            axsigTL, dataTL = self.regr_plot(iso_part, sigTL_part, ax=axsigTL, c1='blue', c2='green', log=True)
            axsigRead, dataRead = self.regr_plot(iso_part, sigRead_part, ax=axRead, c1='blue', c2='green', log=True)
            axuRead, datauRead = self.regr_plot(iso_points[i:i+2], uRead_part, ax=axMean, c1='blue', c2='green', log=False)
            axsigGs, dataGs = self.regr_plot(iso_part, sigGs_part, ax=axsigGs, c1='blue', c2='green',log=True)
            axlam, datalam = self.regr_plot((iso_part-K_ISO[1])/K_ISO[0], lam_part, ax=axlam, log=False, c1='blue', c2='green')
            data = {
                'ISO_start':params[i]['ISO'], 'ISO_end':params[i+1]['ISO'], 'q':1/self.wp, 'wp':self.wp, 'bl':self.bl,
                'Kmin':np.log(np.min(iso_part)), 'Kmax':np.log(np.max(iso_part)), 'lam':np.mean(lam),
                'sigTLk':dataTL['k'], 'sigTLb':dataTL['b'], 'sigTLsig':dataTL['sig'],
                'sigGsk':dataGs['k'], 'sigGsb':dataGs['b'], 'sigGssig':dataGs['sig'],
                'sigRk':dataR['k'], 'sigRb':dataR['b'], 'sigRsig':dataR['sig'],
                'sigReadk':dataRead['k'], 'sigReadb':dataRead['b'], 'sigReadsig':dataRead['sig'],
                'uReadk':datauRead['k'], 'uReadb':datauRead['b'], 'uReadsig':datauRead['sig'],
            }
            self.save_params(data, save_dir)

        axsigR.legend()
        axsigTL.legend()
        axsigRead.legend()
        axuRead.legend()
        axlam.legend()
        axsigGs.legend()

        # plt.show()
        fig.savefig(f'{title}.png')
        plt.close()

        return params

    @staticmethod
    def save_params(data, save_path='iso_parts_params_SonyA7S2.txt'):
        if save_path is None:
            print("cam_noisy_params['ISO_%d_%d'] = {"%(data['ISO_start'], data['ISO_end']))
            print("    'Kmin':%.5f, 'Kmax':%.5f, 'lam':%.3f, 'q':%.3e, 'wp':%d, 'bl':%d,"%(data['Kmin'], data['Kmax'], data['lam'], data['q'], data['wp'],data['bl']))
            print("    'sigRk':%.5f,  'sigRb':%.5f,  'sigRsig':%.5f,"%(data['sigRk'], data['sigRb'], data['sigRsig']))
            print("    'sigTLk':%.5f, 'sigTLb':%.5f, 'sigTLsig':%.5f,"%(data['sigTLk'], data['sigTLb'], data['sigTLsig']))
            print("    'sigGsk':%.5f, 'sigGsb':%.5f, 'sigGssig':%.5f"%(data['sigGsk'], data['sigGsb'], data['sigGssig']))
            print("    'sigReadk':%.5f, 'sigReadb':%.5f, 'sigReadsig':%.5f,"%(data['sigReadk'], data['sigReadb'], data['sigReadsig']))
            print("    'uReadk':%.5f, 'uReadb':%.5f, 'uReadsig':%.5f}"%(data['uReadk'], data['uReadb'], data['uReadsig']))
        else:
            f=open(save_path, "a+")
            print("cam_noisy_params['ISO_%d_%d'] = {"%(data['ISO_start'], data['ISO_end']), file=f)
            print("    'Kmin':%.5f, 'Kmax':%.5f, 'lam':%.3f, q':%.3e, 'wp':%d, 'bl':%d,"%(data['Kmin'], data['Kmax'], data['lam'], data['q'], data['wp'],data['bl']), file=f)
            print("    'sigRk':%.5f,  'sigRb':%.5f,  'sigRsig':%.5f,"%(data['sigRk'], data['sigRb'], data['sigRsig']), file=f)
            print("    'sigTLk':%.5f, 'sigTLb':%.5f, 'sigTLsig':%.5f,"%(data['sigTLk'], data['sigTLb'], data['sigTLsig']), file=f)
            print("    'sigGsk':%.5f, 'sigGsb':%.5f, 'sigGssig':%.5f,"%(data['sigGsk'], data['sigGsb'], data['sigGssig']), file=f)
            print("    'sigReadk':%.5f, 'sigReadb':%.5f, 'sigReadsig':%.5f,"%(data['sigReadk'], data['sigReadb'], data['sigReadsig']), file=f)
            print("    'uReadk':%.5f, 'uReadb':%.5f, 'uReadsig':%.5f}"%(data['uReadk'], data['uReadb'], data['uReadsig']), file=f)

    @staticmethod
    # , c1='red', c2='orange'
    def regr_plot(x, y, log=True, ax=None, c1=None, c2=None, label=False):
        x = np.array(x)
        y = np.array(y)
        if log:
            x = np.log(x)
            y = np.log(y)
        ax.scatter(x, y)

        regr = LinearRegression()
        regr.fit(x.reshape(-1,1), y)
        a, b = float(regr.coef_), float(regr.intercept_)   
        # ax.set_title('log(sigR) | log(K)')
        x_range = np.array([np.min(x), np.max(x)])
        std = np.mean((a*x+b-y)**2) ** 0.5
        
        if c1 is not None:
            if label:
                label = f'k={a:.5f}, b={b:.5f}, std={std:.5f}'
                ax.plot(x, regr.predict(x.reshape(-1,1)), linewidth = 2, c=c1, label=label)
            else:
                ax.plot(x, regr.predict(x.reshape(-1,1)), linewidth = 2, c=c1)
        
        if c2 is not None:
            ax.plot(x_range, a*x_range+b+std, c=c2, linewidth = 1)
            ax.plot(x_range, a*x_range+b-std, c=c2, linewidth = 1)

        data = {'k':a,'b':b,'sig':std}

        return ax, data

def get_darkshading(args, save=True):
    log(f'getting darkshading of ISO-{args["ISO"]}', log='./ds_naive.log')
    raws = np.zeros((args["H"], args["W"]), np.float32)
    src_dir = os.path.join(args["src_dir"], args["ISO"])
    paths = os.listdir(src_dir)
    pbar = tqdm(paths)
    # naive dark shading
    for filename in pbar:
        filepath = os.path.join(src_dir, filename)
        raw = rawpy.imread(filepath).raw_image_visible.astype(np.float32) - args["bl"]
        raws += raw
        pbar.set_description(f'ISO-{args["ISO"]}')
    raw = raws / len(paths)
    # statistic
    raw = bayer2rggb(raw)   # ([H,W,4])
    mean_raw = raw.mean(axis=(0,1)) # [4]
    sigma_raw = raw.std(axis=(0,1)) # [4]
    pattern = 'RGGB'
    for c in range(4):
        log(f'{pattern[c]}, mean:{mean_raw[c]:.3f}, sigma:{sigma_raw[c]:.3e}', log='./ds_naive.log')
    # ignore bad points (because different camera usually have different bad points)
    noise_map = raw - mean_raw
    denoised = cv2.medianBlur(noise_map,5)
    sigma_dark = (noise_map-denoised).std(axis=(0,1))
    # get bad points
    bad_points_map = np.abs((noise_map-denoised))>6*sigma_dark.reshape(1,1,4)
    bad_pixels = np.array(np.where(bad_points_map==True)).transpose()
    bpc_img = bad_points_map.astype(np.uint8) * 255
    bpc_img = rggb2bayer(bpc_img)
    log(f'bad points:{len(bad_pixels)}, refine_std:{sigma_raw} -> {sigma_dark}', log='./ds_naive.log')
    # refine dark shading by throw bad points
    # dark_shading = denoised * bad_points_map + noise_map * (1-bad_points_map)
    dark_shading = noise_map
    dark_shading = rggb2bayer(dark_shading+mean_raw)
    denoised = rggb2bayer(denoised+mean_raw)
    if save:
        os.makedirs(args["dst_dir"], exist_ok=True)
        np.save(f'{args["dst_dir"]}/bpc-iso-{args["ISO"]}.npy', bad_pixels)
        cv2.imwrite(f'{args["dst_dir"]}/bpc-iso-{args["ISO"]}.png', bpc_img)
        np.save(f'{args["dst_dir"]}/darkshading-iso-{args["ISO"]}.npy', dark_shading)
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(dark_shading.clip(mean_raw.mean()-3*sigma_dark.mean(),mean_raw.mean()+3*sigma_dark.mean()))
        plt.colorbar()
        fig.savefig(f'{args["dst_dir"]}/darkshading-iso-{args["ISO"]}.png', bbox_inches='tight')
        # plt.show()
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(denoised.clip(mean_raw.mean()-3*sigma_dark.mean(),mean_raw.mean()+3*sigma_dark.mean()))
        plt.colorbar()
        fig.savefig(f'{args["dst_dir"]}/darkshading-denoised-iso-{args["ISO"]}.png', bbox_inches='tight')
        # plt.show()
        plt.close()
    return dark_shading, bad_pixels

def get_darkshading_templet(args, legal_iso=[], hint='', denoised=False):
    darkshadings = []
    ds_mean = []
    ds_std = []
    hint_dn = '_denoised' if denoised else ''
    for i, iso in enumerate(legal_iso):
        ds_path = os.path.join(args['dst_dir'], f'darkshading-iso-{iso}.npy')
        ds = np.load(ds_path)
        if denoised:
            ds = bayer2rggb(ds)
            ds = cv2.medianBlur(ds,5)
            ds = rggb2bayer(ds)
        darkshadings.append(ds-ds.mean())
        ds_mean.append(ds.mean())
        ds_std.append(ds.std())
        log(f'ISO:{iso}, mean:{ds_mean[i]:.4f}, std:{ds_std[i]:.4f}', log='./ds_templet.log')

    h,w = darkshadings[0].shape
    ds_data = np.array(darkshadings).reshape(len(legal_iso), -1)
    reg = np.polyfit(legal_iso,ds_data,deg=1)
    ds_k = reg[0].reshape(h, w)
    ds_b = reg[1].reshape(h, w)
    # ry = np.polyval(reg, legal_iso)
    print(ds_k.std(), ds_b.mean())
    # ds_b = repair_bad_pixels(ds_b, bad_points=np.load(f'{args["dst_dir"]}/bpc.npy'))
    np.save(os.path.join(args['dst_dir'], f'darkshading{hint}{hint_dn}_k.npy'), ds_k)
    np.save(os.path.join(args['dst_dir'], f'darkshading{hint}{hint_dn}_b.npy'), ds_b)
    plt.figure(figsize=(16,10))
    plt.imshow(ds_k.clip(-3*ds_k.std(),3*ds_k.std()))
    plt.colorbar()
    plt.savefig(f'{args["dst_dir"]}/darkshading{hint}{hint_dn}_k.png', bbox_inches='tight')
    plt.figure(figsize=(16,10))
    plt.imshow(ds_b.clip(-3*ds_b.std(),3*ds_b.std()))
    plt.colorbar()
    plt.savefig(f'{args["dst_dir"]}/darkshading{hint}{hint_dn}_b.png', bbox_inches='tight')
    # 保存BLE
    if os.path.exists(os.path.join(args['dst_dir'], f'darkshading{hint_dn}_BLE.pkl')):
        with open(os.path.join(args['dst_dir'], f'darkshading{hint_dn}_BLE.pkl'), 'rb') as f:
            BLE = pkl.load(f)
    else:
        BLE = {}
    for i, iso in enumerate(legal_iso):
        BLE[iso] = ds_mean[i]
    with open(os.path.join(args['dst_dir'], f'darkshading{hint_dn}_BLE.pkl'), 'wb') as f:
        pkl.dump(BLE, f)
    print(BLE)


class NLFParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/SonyA7S2/bias.yml", type=Path, help="path to config")

        return self.parser.parse_args()

if __name__ == '__main__':
    parser = NLFParser()
    parse = parser.parse()
    with open(parse.runfile, 'r', encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # get naive darkshading
    pbar = tqdm(sorted([int(iso) for iso in os.listdir(args['src_dir'])]))
    for iso in pbar:
        args['ISO'] = iso#str(iso)
        nlf = NoiseEstimation(args)
        nlf.noise_param()

    # compute linear darkshading
    if args['camera_type'] == 'SonyA7S2':
        legal_iso = sorted([int(iso) for iso in os.listdir(args['src_dir']) if 100<=int(iso)<=1600])
        get_darkshading_templet(args, legal_iso, '_lowISO')
        get_darkshading_templet(args, legal_iso, '_lowISO', denoised=True)
        legal_iso = sorted([int(iso) for iso in os.listdir(args['src_dir']) if int(iso) >1600])
        get_darkshading_templet(args, legal_iso, '_highISO')
        get_darkshading_templet(args, legal_iso, '_highISO', denoised=True)
    else:
        legal_iso = sorted([int(iso) for iso in os.listdir(args['src_dir'])])
        get_darkshading_templet(args, legal_iso)