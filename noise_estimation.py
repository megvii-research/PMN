from utils import *
from sklearn.linear_model import LinearRegression

class NoiseEstimation():
    def __init__(self, root_dir='bias-14bit', iso=6400, dst_dir='resources-14bit'):
        self.root_dir = root_dir
        self.iso = iso
        self.src_dir = os.path.join(self.root_dir, str(iso))
        self.H, self.W = 2848, 4256
        self.dst_dir = dst_dir
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(f'{self.dst_dir}/darkshading-iso-{self.iso}.npy'):
            self.darkshading, self.bad_points = self.get_darkshading(save=True)
        else:
            self.darkshading = np.load(f'{self.dst_dir}/darkshading-iso-{self.iso}.npy')
            self.bad_points = np.load(f'{self.dst_dir}/bpc-iso-{iso}.npy')

    def get_darkshading(self, save=True):
        print(f'get darkshading of ISO-{self.iso}')
        raws = np.zeros((self.H, self.W), np.float32)
        # self.src_dir = r'F:\datasets\SonyA7S2\bias-14bit\ISO-800'
        paths = os.listdir(self.src_dir)#[37:38]
        for filename in tqdm(paths):
            if filename[-4:] != '.ARW': continue
            filepath = os.path.join(self.src_dir, filename)
            raw = rawpy.imread(filepath).raw_image_visible.astype(np.float32) - 512
            raws += raw
        raw = raws / len(paths)
        mean_raw = raw[400:-400, 600:-600].mean()
        sigma_raw = raw[400:-400, 600:-600].std()
        noise_map = raw - mean_raw
        darkshading = noise_map.copy()
        # darkshading[np.abs(noise_map)>4*sigma_raw] = 0
        ds = FastGuidedFilter(darkshading/16383, np.zeros_like(darkshading), 3, eps=2e-5) * 16383
        sigma_dark = (darkshading-ds).std()
        print('mean:',noise_map.mean())
        print(f'{darkshading.std():.3e} -> {(darkshading-ds).std():.3e}')
        bad_points_map = np.abs(noise_map)>6*(darkshading-ds).std()
        bad_pixels = np.array(np.where(bad_points_map==True)).transpose()
        bpc_img = bad_points_map.astype(np.uint8) * 255
        dark_shading = darkshading + mean_raw
        dark_shading = row_denoise(filepath, self.iso, data=dark_shading)
        # dark_shading = raw.copy().clip(-6*sigma_dark, 6*sigma_dark)
        if save:
            np.save(f'{self.dst_dir}/bpc-iso-{self.iso}.npy', bad_pixels)
            cv2.imwrite(f'{self.dst_dir}/bpc-iso-{self.iso}.png', bpc_img)
            np.save(f'{self.dst_dir}/darkshading-iso-{self.iso}.npy', dark_shading)
            fig = plt.figure(figsize=(16, 10))
            # add a colorbar legend
            plt.imshow(ds.clip(-0.75,0.75))
            plt.colorbar()
            fig.savefig(f'{self.dst_dir}/darkshading-iso-{self.iso}.png', bbox_inches='tight')
            # plt.show()
            plt.close()
        return dark_shading, bad_pixels

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
            im = raw.raw_image_visible.astype(np.float32) - 512 - self.darkshading
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
        if os.path.exists(f'{self.dst_dir}/noiseparam-iso-{self.iso}.h5'):
            return
        r = 400
        raw_paths = [os.path.join(self.src_dir, name) for name in os.listdir(self.src_dir) if name[-4:]=='.ARW']
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
            raw = rawpy.imread(raw_paths[i]).raw_image_visible.astype(np.float32) - self.darkshading - 512
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
            'lam':lamda, 'wp':16383, 'bl':512,
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

    def get_real_darkshading(self, isos):
        isos = np.array(isos)
        darkshadings = []
        for iso in tqdm(isos):
            darkshadings.append(dataload(f'{self.dst_dir}/darkshading-iso-{iso}.npy')/np.log2(iso/100))
        darkshadings = np.array(darkshadings)
        plt.plot(np.log2(isos/100), darkshadings.mean(axis=(1,2)))
        # plt.show()
        real_ds = np.mean(darkshadings, axis=0)
        plt.imshow(real_ds.clip(-.75, .75))
        plt.colorbar()
        # plt.show()
    
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
        
        if dual_ISO is False:
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
                'ISO_start':params[0]['ISO'], 'ISO_end':params[-1]['ISO'],
                'Kmin':np.log(np.min(iso)), 'Kmax':np.log(np.max(iso)), 'lam':np.mean(lam), 'q':1/(2**14), 'wp':16383,
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
                    'ISO_start':params[i]['ISO'], 'ISO_end':params[i+1]['ISO'],
                    'Kmin':np.log(np.min(iso_part)), 'Kmax':np.log(np.max(iso_part)), 'lam':np.mean(lam), 'q':1/(2**14), 'wp':16383,
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


        else:
            axsigR = [None]*2
            axsigTL = [None]*2
            for f in range(2):
                axsigR[f] = fig.add_subplot(2,2,f*2+1)
                axsigTL[f] = fig.add_subplot(2,2,f*2+2)
            iso=[[],[]]
            sigR=[[],[]]
            sigTL=[[],[]]

            for param in params:
                flag = 0 if param['ISO'] <= 1600 else 1
                for i in range(len(param['sigmaR'])):
                    iso[flag].append(param['ISO']*K_ISO)
                sigR[flag].append(param['sigmaR'])
                sigTL[flag].append(param['sigmaTL'])

            for f in range(2):
                exstr = 'ISO>1600 <=> ' if f else 'ISO<=1600 <=> '
                sigR[f] = np.array(sigR[f])
                sigTL[f] = np.array(sigTL[f])
                iso[f] = np.array(iso[f])
                sigR[f] = sigR[f].flatten()
                sigTL[f] = sigTL[f].flatten()
                iso[f] = np.log(iso[f])
                sigR[f] = np.log(sigR[f])
                sigTL[f] = np.log(sigTL[f])

                regrR = LinearRegression()
                regrTL = LinearRegression()

                axsigR[f].set_title(exstr+'log(sigR) | log(K)')
                axsigTL[f].set_title(exstr+'log(sigTL) | log(K)')

                axsigR[f].scatter(iso[f], sigR[f])
                axsigTL[f].scatter(iso[f], sigTL[f])

                regrR.fit(iso[f].reshape(-1,1), sigR[f])
                regrTL.fit(iso[f].reshape(-1,1), sigTL[f])

                aR, bR = float(regrR.coef_), float(regrR.intercept_)
                axsigR[f].plot(iso[f], regrR.predict(iso[f].reshape(-1,1)),
                            c='red', label=f'k={aR:.5f}, b={bR:.5f}',linewidth = 2)
                
                aTL, bTL = float(regrTL.coef_), float(regrTL.intercept_)
                axsigTL[f].plot(iso[f], regrTL.predict(iso[f].reshape(-1,1)),
                            c='red', label=f'k={aTL:.5f}, b={bTL:.5f}',linewidth = 2)
                
                iso_range = np.array([np.min(iso[f]), np.max(iso[f])])
                stdR = np.mean((aR*iso[f]+bR-sigR[f])**2) ** 0.5
                stdTL = np.mean((aTL*iso[f]+bTL-sigTL[f])**2) ** 0.5
                print(f"stdR={stdR:.5f}, stdTL={stdTL:.5f}")
                axsigTL[f].plot(iso_range, aTL*iso_range+bTL+stdTL, c='orange', linewidth = 1)
                axsigTL[f].plot(iso_range, aTL*iso_range+bTL-stdTL, c='orange', linewidth = 1)
                axsigR[f].plot(iso_range, aR*iso_range+bR+stdR, c='orange', linewidth = 1)
                axsigR[f].plot(iso_range, aR*iso_range+bR-stdR, c='orange', linewidth = 1)

                axsigR[f].legend()
                axsigTL[f].legend()
        
        # plt.show()
        fig.savefig(f'{title}.png')
        plt.close()

        return params

    @staticmethod
    def save_params(data, save_path='iso_parts_params_SonyA7S2.txt'):
        if save_path is None:
            print("cam_noisy_params['ISO_%d_%d'] = {"%(data['ISO_start'], data['ISO_end']))
            print("    'Kmin':%.5f, 'Kmax':%.5f, 'lam':%.3f, 'q':1/(2**14), 'wp':16383, 'bl':512,"%(data['Kmin'], data['Kmax'], data['lam']))
            print("    'sigRk':%.5f,  'sigRb':%.5f,  'sigRsig':%.5f,"%(data['sigRk'], data['sigRb'], data['sigRsig']))
            print("    'sigTLk':%.5f, 'sigTLb':%.5f, 'sigTLsig':%.5f,"%(data['sigTLk'], data['sigTLb'], data['sigTLsig']))
            print("    'sigGsk':%.5f, 'sigGsb':%.5f, 'sigGssig':%.5f"%(data['sigGsk'], data['sigGsb'], data['sigGssig']))
            print("    'sigReadk':%.5f, 'sigReadb':%.5f, 'sigReadsig':%.5f,"%(data['sigReadk'], data['sigReadb'], data['sigReadsig']))
            print("    'uReadk':%.5f, 'uReadb':%.5f, 'uReadsig':%.5f}"%(data['uReadk'], data['uReadb'], data['uReadsig']))
        else:
            f=open(save_path, "a+")
            print("cam_noisy_params['ISO_%d_%d'] = {"%(data['ISO_start'], data['ISO_end']), file=f)
            print("    'Kmin':%.5f, 'Kmax':%.5f, 'lam':%.3f, 'q':1/(2**14), 'wp':16383, 'bl':512,"%(data['Kmin'], data['Kmax'], data['lam']), file=f)
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

class NLFParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--bias_dir', default='/data/SonyA7S2/bias', help="path to config")
        self.parser.add_argument('--dst_dir', default='/data/SonyA7S2/resources', help="path to config")
        self.parser.add_argument("--save_dir", "-save", default="nlf/Camera_SonyA7S2_highISO.txt", help="params save dir")
        self.parser.add_argument("--K_k", "-k", default=0.0009546, type=float, help="k of K_ISO")
        self.parser.add_argument("--K_b", "-b", default=-0.00193, type=float, help="b of K_ISO")
        return self.parser.parse_args()    

parser = NLFParser()
args = parser.parse()
bias_dir = args.bias_dir
dst_dir = args.dst_dir
iso_dir = sorted([int(name) for name in os.listdir(bias_dir)])
# get_real_darkshading(iso_dir)
print(iso_dir)
for i in range(len(iso_dir)):
    nlf = NoiseEstimation(bias_dir, iso=iso_dir[i], dst_dir=dst_dir)
#     nlf.get_darkshading()
for i in range(len(iso_dir)):
    nlf = NoiseEstimation(bias_dir, iso=iso_dir[i], dst_dir=dst_dir)
    nlf.noise_param()
# os.makedirs('nlf', exist_ok=True)
# f=open(args.save_dir, "w+")
# f.close()
# nlf.analysis_data(args.dst_dir, K_ISO=(args.K_k, args.K_b), save_dir=args.save_dir, title='Noise profile')

# with h5py.File(f'resources/noiseparam-iso-1600.h5', 'r') as f:
#     for key in f:
#         print(f'{key}, mean:{np.mean(f[key]):.3e}, std:{np.std(f[key]):.3e}')
# with h5py.File(f'resources-14bit/noiseparam-iso-1600.h5', 'r') as f:
#     for key in f:
#         print(f'{key}, mean:{np.mean(f[key]):.3e}, std:{np.std(f[key]):.3e}')

