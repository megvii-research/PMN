from utils import *
from data_process.process import *
import platform

SonyCCM = np.array( [[ 1.9712269,-0.6789218, -0.29230508],
                    [-0.29104823, 1.748401 , -0.45735288],
                    [ 0.02051281,-0.5380369,  1.5175241 ]])

def get_raw_with_info(path):
    raw = rawpy.imread(path)
    info = get_ISO_ExposureTime(path)
    gt_img = raw.raw_image_visible
    name = path.split('\\')[-1][:-4] if platform.system() == 'Windows' else path.split('/')[-1][:-4]
    info['name'] = name
    info['wb'], info['ccm'] = read_wb_ccm(raw)
    if info['ccm'][0,0] == 1:
        warnings.warn('Use SonyCCM')
        # print('Use SonyCCM')
        info['ccm'] = SonyCCM
    return gt_img, info

def get_basic_info(path):
    raw = rawpy.imread(path)
    info = get_ISO_ExposureTime(path)
    name = path.split('\\')[-1][:-4] if platform.system() == 'Windows' else path.split('/')[-1][:-4]
    info['name'] = name
    info['wb'], info['ccm'] = read_wb_ccm(raw)
    if info['ccm'][0,0] == 1:
        warnings.warn('Use SonyCCM')
        info['ccm'] = SonyCCM
    return info

def get_SID_info(info_dir='info', root_dir='/data/SID/Sony', mode='train'):
    root_dir = os.path.join(root_dir, 'long')
    head = []
    if 'train' in mode: head.append('0')
    if 'eval' in mode: head.append('1')
    if 'test' in mode: head.append('2')

    names = sorted([name for name in os.listdir(root_dir) if name[0] in head])
    names_short = [name for name in sorted(os.listdir(root_dir.replace('long', 'short'))) if name[0] in head]
    paths_short = []
    paths = [os.path.join(root_dir.replace('long', 'short'), names_short[0])]
    for i in range(1, len(names_short)):
        if names_short[i-1][:5] == names_short[i][:5]:
            paths.append(os.path.join(root_dir.replace('long', 'short'), names_short[i]))
        else:
            paths_short.append(paths)
            paths = []
    paths_short.append(paths)
        
    pbar = tqdm(range(len(names)))
    infos = []
    
    for i in pbar:
        path = os.path.join(root_dir, names[i])
        info = get_basic_info(path)
        info['ratio'] = np.zeros(len(paths_short[i]), dtype='int')
        for k in range(len(paths_short[i])):
            info_short = get_basic_info(paths_short[i][k])
            info['ratio'][k] = int(info['ExposureTime']/info_short['ExposureTime'])
            
        info['long'] = path
        info['short'] = paths_short[i]
        
        infos.append(info)
        pbar.set_description_str(f'Raw:{info["name"]}')

    info_path = os.path.join(info_dir, f'SID_{mode}.info')
    with open(info_path, 'wb') as out_file:
        pkl.dump(infos, out_file)
    return infos

def get_SID_info_from_txt(info_dir='infos', root_dir='/data/SID/Sony', txt='SID_Sony_paired.txt'):
    fns = read_paired_fns(txt)
    long_dir = os.path.join(root_dir, 'long')
    short_dir = os.path.join(root_dir, 'short')

    names = []
    paths = []
    paths_short = []
    for i in range(len(fns)):
        names.append(fns[0])
        paths.append(os.path.join(long_dir, fns[i][1]))
        paths_short.append(os.path.join(short_dir, fns[i][0]))

    pbar = tqdm(range(len(names)))
    infos = []
    # 第二级，文件
    for i in pbar:
        info = get_basic_info(paths[i])
        info_short = get_basic_info(paths_short[i])
        info['ratio'] = int(info['ExposureTime']/info_short['ExposureTime'])
        info['long'] = paths[i]
        info['short'] = paths_short[i]
        
        infos.append(info)
        pbar.set_description_str(f'Raw:{info["name"]}, Ratio:{fns[i][2]}')

    info_path = os.path.join(info_dir, f'SID_evaltest.info')
    with open(info_path, 'wb') as out_file:
        pkl.dump(infos, out_file)
    return infos

def get_ELD_info(info_dir='infos', root_dir='/data/ELD'):
    root_dir = os.path.join(root_dir, 'SonyA7S2')
    pbar = tqdm([f'scene-{i+1}' for i in range(10)])
    infos = []
    ratio_list = [1, 1, 10, 100, 200]
    # 第二级，文件
    for i, scene in enumerate(pbar):
        infos_scene = []
        path_scene = os.path.join(root_dir, scene)
        for k in range(16):
            path = os.path.join(path_scene, f'IMG_{k+1:04d}.ARW')
            info = get_basic_info(path)
            info['ratio'] = ratio_list[k%5]
            info['data'] = path
            infos_scene.append(info)
            pbar.set_description_str(f'{scene} - {info["name"]}')
        infos.append(infos_scene)

    info_path = os.path.join(info_dir, f'ELD_SonyA7S2.info')
    with open(info_path, 'wb') as out_file:
        pkl.dump(infos, out_file)
    return infos

def get_IMX686_info_short(
        info_dir='infos', 
        root_dir='/data/LRID', 
        subset='indoor_x5'):
    root_dir = os.path.join(root_dir, subset, '6400')
    infos = {}
    for dgain in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, dgain)
        log(f'Dgain={dgain}, Path={sub_dir}', log='worklog/IMX686_short.log')
        scenes = sorted([scene for scene in os.listdir(sub_dir)])
        pbar = tqdm(range(len(scenes)))
        dgain = int(dgain)
        infos[dgain] = []
        # 第二级，文件
        for i in pbar:
            info_scene = {'scene':scenes[i], 'dgain':dgain, 'ISO':6400, 'data':[], 'metadata':[]}
            paths = [path for path in sorted(glob.glob(os.path.join(sub_dir, scenes[i], '*.dng')))]
            for k in range(len(paths)):
                try:
                    info = get_basic_info(paths[k])
                    info_scene['data'].append(paths[k])
                    dng_items = info['name'].split('_')
                    time = '-'.join(dng_items[3:10])
                    info['time'] = time
                    info_scene['metadata'].append(info)
                except Exception as e:
                    log(f'Error: {str(e)}', log='worklog/IMX686_short.log')
                    log(f'Some Error happend, we skip the data at "{paths[k]}"...', log='worklog/IMX686_short.log')

            infos[dgain].append(info_scene)
            pbar.set_description_str(f'Dgain-{dgain:02d}, Scene-{scenes[i]}')

    info_path = os.path.join(info_dir, f'{subset}_short.info')
    with open(info_path, 'wb') as out_file:
        pkl.dump(infos, out_file)
    return infos

    nw_short.close()

def get_IMX686_info_long(
        info_dir='infos', 
        root_dir='/data/LRID', 
        subset='indoor_x5',
        GTtype='GT_align_ours'):
    sub_dir = os.path.join(root_dir, subset, 'npy', GTtype)
    log(f'Path={sub_dir}', log='worklog/IMX686_long.log')
    infos = []
    # metadata
    with open(os.path.join(root_dir, subset, f'metadata_{subset}_gt.pkl'), 'rb') as f:
        metadatas = pickle.load(f)
    # 第二级，文件
    scenes = sorted([scene[:-4] for scene in sorted(os.listdir(sub_dir))])
    pbar = tqdm(range(len(scenes)))
    for i in pbar:
        path_long = os.path.join(sub_dir, f'{scenes[i]}.npy')
        filename = f'{subset}_gt_{scenes[i]}'
        info = {'name':filename, 'data':path_long, 'ccm':metadatas[i]['ccm'], 'wb':metadatas[i]['wb']}

        infos.append(info)
        pbar.set_description_str(f'Scene-{scenes[i]}')

    info_path = os.path.join(info_dir, f"{subset}_{GTtype}.info")
    with open(info_path, 'wb') as out_file:
        pkl.dump(infos, out_file)
    return infos

class DatasetInfoParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--dstname', '-d', default='SID', type=str)
        self.parser.add_argument('--root_dir', '-r', default="E:/datasets/SID/Sony", type=str)
        self.parser.add_argument('--info_dir', '-i', default="./infos", type=str)
        self.parser.add_argument('--mode', '-m', default='test', type=str)

        return self.parser.parse_args()

if __name__ == "__main__":
    parser = DatasetInfoParser()
    args = parser.parse()
    os.makedirs(args.info_dir, exist_ok=True)
    os.makedirs('worklog', exist_ok=True)
    if args.dstname == 'ELD':
        infos = get_ELD_info(info_dir=args.info_dir, root_dir=args.root_dir)
    elif args.dstname == 'SID':
        if args.mode == 'evaltest':
            infos = get_SID_info_from_txt(info_dir=args.info_dir, root_dir=args.root_dir)
        else:
            infos = get_SID_info(info_dir=args.info_dir, root_dir=args.root_dir, mode=args.mode)
    elif args.dstname == 'LRID':
        for dir in ['indoor_x5','indoor_x3','outdoor_x3']:#, 'outdoor_x5']:
            for GTtype in ['GT_align_ours']:#,'GT_align_median','GT_align_mean','GT_ours','GT_median','GT_mean']:
                infos = get_IMX686_info_long(info_dir=args.info_dir, root_dir=args.root_dir, subset=dir, GTtype=GTtype)
        for dir in ['indoor_x5','indoor_x3','outdoor_x3']:#, 'outdoor_x5']:
            infos = get_IMX686_info_short(info_dir=args.info_dir, root_dir=args.root_dir, subset=dir)
    print('Info Example:', infos[0])
    print()
    