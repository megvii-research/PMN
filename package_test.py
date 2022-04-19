import enum
from utils import *
from data_process import *
from archs import *
from base_trainer import *

def get_darkshading_templet(root_dir='/data/SonyA7S2/resources', legal_iso=[100, 125], hint='lowISO', num=None):
    darkshadings = []
    suffix1 = '' if num is None else f'-{num}'
    suffix2 = '' if num is None else f'_{num}'
    for iso in tqdm(legal_iso):
        ds_path = os.path.join(root_dir, f'darkshading-iso-{iso}{suffix1}.npy')
        ds = np.load(ds_path)
        if np.mean(ds) > 400:
            ds = ds - 512
        darkshadings.append(ds)
    ds_mean = []
    ds_std = []
    for i, iso in enumerate(legal_iso):
        ds = darkshadings[i]
        print(f'ISO:{iso}, mean:{ds.mean():.4f}, std:{ds.std():.4f}')
        ds_mean.append(ds.mean())
        ds_std.append(ds.std())
        darkshadings[i] = ds - ds.mean()
    # h,w = darkshadings[0].shape
    # ds_data = np.array(darkshadings).reshape(len(legal_iso), -1)
    # reg = np.polyfit(legal_iso,ds_data,deg=1)
    # ds_k = reg[0].reshape(h, w)
    # ds_b = reg[1].reshape(h, w)
    # np.save(os.path.join(root_dir, f'darkshading_{hint}{suffix2}_k.npy'), ds_k)
    # np.save(os.path.join(root_dir, f'darkshading_{hint}{suffix2}_b.npy'), ds_b)
    # 保存BLE
    if os.path.exists(os.path.join(root_dir, f'darkshading_BLE.pkl')):
        with open(os.path.join(root_dir, f'darkshading_BLE.pkl'), 'rb') as f:
            BLE = pkl.load(f)
    else:
        BLE = {}
    for i, iso in enumerate(legal_iso):
        BLE[iso] = ds_mean[i]
    with open(os.path.join(root_dir, f'darkshading_BLE_{num}.pkl'), 'wb') as f:
        pkl.dump(BLE, f)
    print(BLE)

def ablation_study(bias_dir='/data/SonyA7S2/bias-3355', ds_dir='/data/SonyA7S2/ds_ablation', legal_iso=[]):
    checkpoint = [1,4,9,16,25,36,49,64,81,100,150,200,250,300,350,400,1000]
    os.makedirs(ds_dir, exist_ok=True)
    for iso in legal_iso:
        p = 0
        df_dir = os.path.join(bias_dir, str(iso))
        pbar = tqdm(glob.glob(df_dir+'/*.ARW')[:400])
        ds_naive = np.zeros((2848, 4256),dtype=np.float32)
        for k, path in enumerate(pbar):
            df = dataload(path)
            ds_naive = (ds_naive*k + df) / (k+1)
            if k == checkpoint[p]-1:
                p += 1
                ds_path = os.path.join(ds_dir, f'darkshading-iso-{iso}-{k+1}.npy')
                np.save(ds_path, ds_naive)
            pbar.set_description_str(f'ISO:{iso}')

if __name__ == '__main__':
    # ablation_study(legal_iso=np.array([2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600]))
    # for camera_id in ['3355']:
    # for camera_id in ['1943', '2087', '2694', '3355']:
    for num in [1,4,9,16,25,36,49,64,81,100,150,200,250,300,350,400]:
        # num = 's2'
        # camera_id = '3355'
        get_darkshading_templet(# root_dir=f'/data/SonyA7S2/resources-{camera_id}', 
                                root_dir=f'/data/SonyA7S2/ds_ablation/', 
                                # legal_iso=np.array([50,64,80,100,125,160,200,250,320,400,500,640,800,1000,1250,1600]),
                                legal_iso=np.array([2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600]),
                                hint='highISO',num=num)
        # get_darkshading_templet(root_dir=f'/data/SonyA7S2/resources-{camera_id}', 
        #                         # root_dir=f'/data/SonyA7S2/ds_ablation/', 
        #                         legal_iso=np.array([100,200,400,800,1600]),
        #                         # legal_iso=np.array([2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600]),
        #                         hint='lowISO_s4')