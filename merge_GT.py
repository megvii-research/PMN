from utils import *

# sigma是σ_read, gain是K
def VST(x, sigma, mu=0, gain=1.0, wp=1):
    # 无增益时，y = 2 * np.sqrt(x + 3.0 / 8.0 + sigma ** 2)
    y = x * wp
    y = gain * x + (gain ** 2) * 3.0 / 8.0 + sigma ** 2 - gain * mu
    y = np.sqrt(np.maximum(y, np.zeros_like(y)))
    y = y / wp
    return (2.0 / gain) * y

# sigma是σ_read, gain是K
def inverse_VST(x, sigma, gain=1.0, wp=1):
    x = x * wp
    y = (x / 2.0)**2 - 3.0/8.0 - sigma**2 / gain**2
    y_exact =  y * gain
    y_exact = y_exact / wp
    return y_exact

def bayer2gray(raw):
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], np.float32) / 16.
    gray = cv2.filter2D(raw, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return gray

def stdfilt(img, k=5):
    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img, (k, k))
    result_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    result_2 = cv2.blur(img_2, (k, k))
    result = np.sqrt(np.maximum(result_2 - result_1, 0))
    return result

def bad_point_detection(rggb):
    bad_points_map = np.zeros_like(rggb, dtype=np.uint8)
    # kernel = np.array([[0,1,0],[1,0,1],[0,1,0]], np.float32)
    rggb = rggb.astype(np.float32)
    max_nums = np.max(rggb,axis=(0,1))
    for c in range(4):
        bad_points_map_c = bad_points_map[:,:,c]
        max_num = max_nums[c]
        img = rggb[:,:,c]
        # Real-Time Photo Sensor Dead Pixel Detection for Embedded Devices
        # img_sum = cv2.filter2D(img, -1, kernel)
        # img_est = img_sum / 4
        # img_diff = np.abs(img_est - img)
        # img_avg = (img_sum + 4 * img) / 8
        # Modified Version
        img_sum = cv2.boxFilter(img, -1, (3,3), normalize=False)
        img_est = (img_sum - img) / 8
        img_diff = np.abs(img_est - img)
        img_avg = (img_sum + 7*img) / 16
        mask_high = img_diff > img_avg
        mask_low = img_diff > max_num-img_avg
        bad_points_map_c[mask_high + mask_low] = 255
    bad_points_map = rggb2bayer(bad_points_map)
    bad_points = np.array(np.where(bad_points_map>0)).transpose()
    return bad_points_map, bad_points

def robust_merge(args, ref, raw_data, scene):
    K = 0.125
    sigGs = 0.691
    raw_data_gray_vst = VST(np.array([(bayer2gray(raw) - 64).clip(0,959) for raw in raw_data]), sigGs, gain=K)
    img_mean = np.mean(raw_data, axis=0)
    img_mid = np.median(raw_data, axis=0)
    ref = repair_bad_pixels(ref, args['bad_points'], method='median')
    img_mid = repair_bad_pixels(img_mid, args['bad_points'], method='median')
    img_mean = repair_bad_pixels(img_mean, args['bad_points'], method='median')
    img_dn = GuidedFilter(bayer2rggb(ref), bayer2rggb(img_mid), d=7, eps=1e-5)
    img_dn = rggb2bayer(img_dn)

    std_map = stdfilt(VST((bayer2gray(ref) - 64).clip(0,959), sigGs, gain=K), k=5) / 3
    std_map = cv2.GaussianBlur(std_map, (5,5),0)
    img_mean_vst = np.mean(raw_data_gray_vst, axis=0)
    img_mid_vst = np.median(raw_data_gray_vst, axis=0)
    temp_map = np.abs(img_mid_vst - img_mean_vst) / 0.6
    temp_map = cv2.GaussianBlur(temp_map, (5,5),0)
    temp_map[temp_map>=1] = 1
    temp_map[temp_map<.2] = (temp_map[temp_map<.2] * 5) ** 4 / 5
    temp_map[ref>=1000] = 1
    fig = plt.figure(figsize=(16,10))
    plt.imsave(f'{args["root_dir"]}/GT/{scene}_GT{args["align_type"]}_temp.png', temp_map)
    std_map[std_map>=1] = 1
    std_map[std_map<0.2] = 0
    # std_map = np.sqrt((std_map-0.2).clip(0,1)/(1-0.2))
    # std_map = rggb2bayer(FastGuidedFilter(bayer2rggb(std_map), vst, 5, eps=1e-6))
    fig = plt.figure(figsize=(16,10))
    plt.imsave(f'{args["root_dir"]}/GT/{scene}_GT{args["align_type"]}_std.png', std_map)
    # 运动的地方和过曝区倾向于用当前帧的结果以确保正确
    img_mid = ref * temp_map + img_mid * (1-temp_map)
    # 平坦的地方和亮区用去噪帧，边缘用中值帧保持锐利
    img_dn_sharp = rggb2bayer(GuidedFilter(bayer2rggb(ref), bayer2rggb(img_mid), d=3, eps=1e-3))
    img_dn_blur = rggb2bayer(GuidedFilter(bayer2rggb(ref), bayer2rggb(img_mid), d=9, eps=5e-1))
    img_dn = img_dn_sharp * std_map + img_dn_blur * (1-std_map)
    img_merge = img_mid * std_map + img_dn * (1-std_map)
    # 细节与非细节的过渡区域用均值的结果找补细节
    std_map[temp_map>0.2] = 1
    confidece_map = std_map.copy()
    std_map[std_map==1] = 0
    std_map = std_map ** 0.25
    img_merge[std_map>0] =  (img_mean * std_map + img_merge * (1-std_map))[std_map>0]
    np.save(f'{args["root_dir"]}/npy/GT{args["align_type"]}_confidence/{scene}.npy', confidece_map)
    return img_merge, img_dn

def merge_imgs(args, scene, mode='N'):
    log(f'Start to load data for {scene}')
    ref_dir = os.path.join(args['src_dir'], scene)
    ref_path = sorted(glob.glob(os.path.join(ref_dir, '*.dng')))[-1]
    raw_ref = rawpy.imread(ref_path)
    ref = raw_ref.raw_image_visible.astype(np.float32)
    data_dir = os.path.join(args['npy_dir'], f'{scene}.npy')
    raw_data = np.load(data_dir) * 1023
    wb, ccm = read_wb_ccm(raw_ref)
    wb = list(wb)
    metadata = {"name":scene, "ccm":ccm, "wb":np.array(wb)}
    # Merge
    log(f'Merge data to GT via merge_type={args["merge_type"]}')
    bright = np.mean(raw_data, axis=(1,2))
    if args['merge_type'] == 'mean':
        img_merge = np.mean(raw_data, axis=0)
    elif args['merge_type'] == 'median':
        img_merge = np.median(raw_data, axis=0)
    elif args['merge_type'] == 'ours':
        img_merge, raw_dn = robust_merge(args, ref, raw_data, scene)
    # plot
    noisy = raw_ref.postprocess(use_camera_wb=False, user_wb=wb, half_size=False, 
                            no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    h, w, c = noisy.shape
    if args['merge_type'] == 'ours':
        raw_ref.raw_image_visible[:] = raw_dn[:]
        img_dn = raw_ref.postprocess(use_camera_wb=False, user_wb=wb, half_size=False, 
                            no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
        cv2.imwrite(f'{args["root_dir"]}/GT/{scene}_GT{args["align_type"]}_GFdn.png', img_dn[:,:,::-1])
        
    # BPC
    # 最新版本读图的时候已经BPC过了
    args['bpc'] = True
    if args['bpc'] and args['merge_type'] != 'ours':
        # bad_points_map, bad_points = bad_point_detection(bayer2rggb(img_merge))
        # np.save(f'{args["root_dir"]}/GT_bpc/{scene}.npy', bad_points)
        plt.imsave(f'{args["root_dir"]}/GT_bpc/{scene}.png', args['bad_points_map'])
        img_merge = repair_bad_pixels(img_merge, args['bad_points'], method='median')
        log(f'Fixed {len(args["bad_points"])} bad points for scene-{scene}')

    raw_ref.raw_image_visible[:] = img_merge[:]
    gt = raw_ref.postprocess(use_camera_wb=False, user_wb=wb, half_size=False, no_auto_bright=True, 
                        output_bps=8, bright=1, user_black=None, user_sat=None)

    np.save(f'{args["root_dir"]}/npy/GT{args["align_type"]}_{args["merge_type"]}/{scene}.npy', img_merge)
    # plt.figure(figsize=(16,10))
    # plt.imshow(gt)
    # plt.show()
    # plt.figure(figsize=(16,10))
    # plt.imshow(gt-noisy)
    # plt.show()
    # plt.close()

    cv2.imwrite(f'{args["root_dir"]}/GT/{scene}_GT{args["align_type"]}_{args["merge_type"]}.png', gt[:,:,::-1])
    if not os.path.exists(f'{args["root_dir"]}/GT/{scene}_Noisy.png'):
        cv2.imwrite(f'{args["root_dir"]}/GT/{scene}_Noisy.png', noisy[:,:,::-1])

    log(f'Finish merging GT, save as {args["root_dir"]}/npy/GT{args["align_type"]}_{args["merge_type"]}/{scene}.npy')
    return gt, noisy, metadata

def get_metadata(args, scene):
    ref_dir = os.path.join(args['src_dir'], scene)
    ref_path = sorted(glob.glob(os.path.join(ref_dir, '*.dng')))[-1]
    raw_ref = rawpy.imread(ref_path)
    wb, ccm = read_wb_ccm(raw_ref)
    wb = list(wb)
    metadata = {"name":scene, "ccm":ccm, "wb":np.array(wb)}
    return metadata

def process(args, replace=None):
    args['ISO'] = '100'
    # args['bpc'] = False
    args['src_dir'] = os.path.join(args['root_dir'], args['ISO'])
    pool = Pool(4)
    os.makedirs(f'{args["root_dir"]}/GT_bpc', exist_ok=True)
    os.makedirs(f'{args["root_dir"]}/GT', exist_ok=True)
    os.makedirs(f'{args["root_dir"]}/npy/GT{args["align_type"]}_confidence', exist_ok=True)
    os.makedirs(f'{args["root_dir"]}/npy/GT{args["align_type"]}_{args["merge_type"]}', exist_ok=True)
    # merge GT
    for scene in tqdm(sorted(os.listdir(args['src_dir']))[:]):
        if replace is not None:
            if scene not in replace: continue
        # elif os.path.exists(f'{args["root_dir"]}/GT/{scene}_{args["merge_type"]}_GT.png'): continue
        # gt, noisy, metadata = merge_imgs(args, scene)
        pool.apply_async(merge_imgs, (args, scene))
    pool.close()
    pool.join()
    # get metadata
    metadatas = []
    for scene in tqdm(sorted(os.listdir(args['src_dir']))[:]):
        metadatas.append(get_metadata(args, scene))
    with open(os.path.join(args['root_dir'], f'metadata_{os.path.basename(args["root_dir"])}_gt.pkl'), 'wb') as f:
        pkl.dump(metadatas, f)
        log(f'Finish dump metadata of sub_dst:"{os.path.basename(args["root_dir"])}"')

if __name__ == '__main__':
    for dir in ['outdoor_x3', 'indoor_x3', 'indoor_x5']:
        for align_type in ['_align', '']:
            for merge_type in ['ours']:#, 'mean', 'median']:
                f = open('runfiles/IMX686-bias.yml', 'r', encoding="utf-8")
                args = yaml.load(f.read())
                args['bad_points'] = np.load(f'/data/LRID/resources/bpc-iso-100.npy')
                args['bad_points_map'] = cv2.imread(f'/data/LRID/resources/bpc-iso-100.png')
                args['merge_type'] = merge_type
                args['align_type'] = align_type
                args['root_dir'] = os.path.join(args['root_dir'], dir)
                args['npy_dir'] = os.path.join(args['root_dir'], 'npy', f'GT{args["align_type"]}s')
                process(args)