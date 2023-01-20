import numpy as np
import torch
import torch.distributions as tdist
# from utils import fn_timer


def random_ccm(camera_type='IMX686'):
    """Generates random RGB -> Camera color correction matrices."""
    # # Takes a random convex combination of XYZ -> Camera CCMs.
    # xyz2cams = [[[1.0234, -0.2969, -0.2266],
    #            [-0.5625, 1.6328, -0.0469],
    #            [-0.0703, 0.2188, 0.6406]],
    #           [[0.4913, -0.0541, -0.0202],
    #            [-0.613, 1.3513, 0.2906],
    #            [-0.1564, 0.2151, 0.7183]],
    #           [[0.838, -0.263, -0.0639],
    #            [-0.2887, 1.0725, 0.2496],
    #            [-0.0627, 0.1427, 0.5438]],
    #           [[0.6596, -0.2079, -0.0562],
    #            [-0.4782, 1.3016, 0.1933],
    #            [-0.097, 0.1581, 0.5181]]]
    # num_ccms = len(xyz2cams)
    # xyz2cams = torch.FloatTensor(xyz2cams)
    # weights  = torch.FloatTensor(num_ccms, 1, 1).uniform_(1e-8, 1e8)
    # weights_sum = torch.sum(weights, dim=0)
    # xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    # rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
    #                            [0.2126729, 0.7151522, 0.0721750],
    #                            [0.0193339, 0.1191920, 0.9503041]])
    # rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    if camera_type == 'SonyA7S2':
        # SonyA7S2 ccm's inv
        rgb2cam = [[1.,0.,0.],
                [0.,1.,0.],
                [0.,0.,1.]]
    elif camera_type == 'IMX686':
        # RedMi K30 ccm's inv
        rgb2cam = [[0.61093086,0.31565922,0.07340994],
                    [0.09433191,0.7658969,0.1397712 ],
                    [0.03532438,0.3020709,0.6626047 ]]
    rgb2cam = torch.FloatTensor(rgb2cam)
    # # Normalizes each row.
    # rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    return rgb2cam


#def random_gains():
#    """Generates random gains for brightening and white balance."""
#    # RGB gain represents brightening.
#    n        = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1]))
#    rgb_gain = 1.0 / n.sample()
#
#    # Red and blue gains represent white balance.
#    red_gain  =  torch.FloatTensor(1).uniform_(1.9, 2.4)
#    blue_gain =  torch.FloatTensor(1).uniform_(1.5, 1.9)
#    return rgb_gain, red_gain, blue_gain

def random_gains(camera_type='SonyA7S2'):
    # return torch.FloatTensor(np.array([[1.],[1.],[1.]]))
    n = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1]))
    rgb_gain = 1.0 / n.sample()
    # SonyA7S2
    if camera_type == 'SonyA7S2':
        red_gain = np.random.uniform(1.75, 2.65)
        ployfit = [14.65 ,-9.63942308, 1.80288462 ]
        blue_gain= ployfit[0] + ployfit[1] * red_gain + ployfit[2] * red_gain ** 2# + np.random.uniform(0, 0.4)686
    elif camera_type == 'IMX686':
        red_gain = np.random.uniform(1.4, 2.3)
        ployfit = [6.14381188, -3.65620261, 0.70205967]
        blue_gain= ployfit[0] + ployfit[1] * red_gain + ployfit[2] * red_gain ** 2# + np.random.uniform(0, 0.4)
    else:
        raise NotImplementedError
    red_gain = torch.FloatTensor(np.array([red_gain])).view(1)
    blue_gain = torch.FloatTensor(np.array([blue_gain])).view(1)
    return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    #image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    image = torch.clamp(image, min=0.0, max=1.0)
    out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
    #out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    #image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    out   = torch.clamp(image, min=1e-8) ** 2.2
    #out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    shape = image.size()
    image = torch.reshape(image, [-1, 3])
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    out   = torch.reshape(image, shape)
    return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain, use_gpu=False):
    """Inverts gains while safely handling saturated pixels."""
    # H, W, C
    green = torch.tensor([1.0])
    if use_gpu: green = green.cuda()
    gains = torch.stack((1.0 / red_gain, green, 1.0 / blue_gain)) / rgb_gain
    gains = gains.view(1,1,3)
    #gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray  = torch.mean(image, dim=-1, keepdim=True)
    inflection = 0.9
    mask  = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0

    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    out   = image * safe_gains
    return out

def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    if image.size() == 3:
        image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
        shape = image.size()
        red   = image[0::2, 0::2, 0]
        green_red  = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 2]
        out  = torch.stack((red, green_red, blue, green_blue), dim=-1)
        out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
        out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    else: # [crops, t, h, w, c]
        shape = image.size()
        red   = image[..., 0::2, 0::2, 0]
        green_red  = image[..., 0::2, 1::2, 1]
        green_blue = image[..., 1::2, 0::2, 1]
        blue = image[..., 1::2, 1::2, 2]
        out  = torch.stack((red, green_red, blue, green_blue), dim=-1)
        # out  = torch.reshape(out, (shape[0], shape[1], shape[-3] // 2, shape[-2] // 2, 4))
        # out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

def mosaic_GBRG(image):
    """Extracts GBRG Bayer planes from an RGB image."""
    if image.size() == 3:
        image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
        shape = image.size()
        red   = image[1::2, 0::2, 0]
        green_red  = image[1::2, 1::2, 1]
        green_blue = image[0::2, 0::2, 1]
        blue = image[0::2, 1::2, 2]
        out  = torch.stack((red, green_red, green_blue, blue), dim=-1)
        out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
        out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    else: # [crops, t, h, w, c]
        shape = image.size()
        red   = image[..., 1::2, 0::2, 0]
        green_red  = image[..., 1::2, 1::2, 1]
        green_blue = image[..., 0::2, 0::2, 1]
        blue = image[..., 0::2, 1::2, 2]
        out  = torch.stack((red, green_red, green_blue, blue), dim=-1)
        # out  = torch.reshape(out, (shape[0], shape[1], shape[-3] // 2, shape[-2] // 2, 4))
        # out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

# @ fn_timer
def unprocess(image, lock_wb=False, use_gpu=False, camera_type='IMX686'):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Randomly creates image metadata.
    rgb2cam = random_ccm(camera_type)
    cam2rgb = torch.inverse(rgb2cam)
    # rgb_gain, red_gain, blue_gain = random_gains() if lock_wb is False else torch.FloatTensor(np.array([[1.],[2.],[2.]]))
    rgb_gain, red_gain, blue_gain = random_gains(camera_type) if lock_wb is False else torch.FloatTensor(np.array(lock_wb))
    if use_gpu:
        rgb_gain, red_gain, blue_gain = rgb_gain.cuda(), red_gain.cuda(), blue_gain.cuda()
    if len(image.size()) >= 4:
        res = image.clone()
        for i in range(image.size()[0]):
            temp = image[i]
            temp = inverse_smoothstep(temp)
            temp = gamma_expansion(temp)
            temp = apply_ccm(temp, rgb2cam)
            temp = safe_invert_gains(temp, rgb_gain, red_gain, blue_gain, use_gpu)
            temp = torch.clamp(temp, min=0.0, max=1.0)
            res[i]= temp.clone()
        
        metadata = {
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }
        return res, metadata
    else:
        # Approximately inverts global tone mapping.
        image = inverse_smoothstep(image)
        # Inverts gamma compression.
        image = gamma_expansion(image)
        # Inverts color correction.
        image = apply_ccm(image, rgb2cam)
        # Approximately inverts white balance and brightening.
        image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain, use_gpu)
        # Clips saturated pixels.
        image = torch.clamp(image, min=0.0, max=1.0)
        # Applies a Bayer mosaic.
        #image = mosaic(image)

        metadata = {
            'cam2rgb': cam2rgb,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }
        return image, metadata


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise     = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
    shot_noise = torch.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    n    = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26]))
    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise     = torch.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    image    = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    variance = image * shot_noise + read_noise
    n        = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise    = n.sample()
    out      = image + noise
    out      = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

if __name__ == '__main__':
    m = tdist.Poisson(torch.tensor([10.,100.,1000.]))
    for i in range(10):
        s = m.sample()
        print(s.numpy())