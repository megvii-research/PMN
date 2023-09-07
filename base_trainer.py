from utils import *
from archs import *
from losses import *
from pathlib import Path

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/SonyA7S2/Ours.yml", type=Path, help="path to config")
        self.parser.add_argument('--mode', '-m', default='evaltest', type=str, help="train or test")
        self.parser.add_argument('--debug', action='store_true', default=False, help="debug or not")
        self.parser.add_argument('--nofig', action='store_true', default=False, help="don't save_plot")
        return self.parser.parse_args()

# 不这么搞随机pytorch和numpy的联动会出bug，随机种子有问题
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

class Base_Trainer():
    def __init__(self):
        self.initialization()
    
    def get_lr_lambda_func(self):
        num_of_epochs = self.hyper['stop_epoch'] - self.hyper['last_epoch']
        step_size = self.hyper['step_size']
        T = self.hyper['T'] if 'T' in self.hyper else 1 
        if 'cos' in self.hyper['lr_scheduler'].lower():
            self.lr_lambda = lambda x: get_cos_lr(x, period=num_of_epochs//T, lr=self.hyper['learning_rate'], peak=step_size)
        elif 'multi' in self.hyper['lr_scheduler'].lower():
            self.lr_lambda = lambda x: get_multistep_lr(x, period=num_of_epochs//T, decay_base=1,
                                        milestone=[step_size, step_size*9//5], gamma=[0.5, 0.1], 
                                        lr=self.hyper['learning_rate'])
        return self.lr_lambda

    def initialization(self):
        parser = BaseParser()
        self.parser = parser.parse()
        with open(self.parser.runfile, 'r', encoding="utf-8") as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.mode = self.args['mode'] if self.parser.mode is None else self.parser.mode
        if self.parser.debug is True:
            self.args['num_workers'] = 0
            warnings.warn('You are using debug mode, only main worker(cpu) is used!!!')
        if 'clip' not in self.args['dst']: 
            self.args['dst']['clip'] = False
        self.save_plot = False if self.parser.nofig else True
        self.args['dst']['mode'] = self.mode
        self.args['dst_train']['param'] = None
        self.dst = self.args['dst']
        self.hyper = self.args['hyper']
        self.arch = self.args['arch']
        self.arch_isp = self.args['arch_isp'] if 'arch_isp' in self.args else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hostname = socket.gethostname()
        self.model_name = self.args['model_name']
        self.model_dir = self.args['checkpoint']
        self.fast_ckpt = self.args['fast_ckpt']
        self.sample_dir = os.path.join(self.args['result_dir'] ,f"samples-{self.model_name}")
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.sample_dir+'/temp', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        os.makedirs(f'./{self.fast_ckpt}', exist_ok=True)
        os.makedirs('./metrics', exist_ok=True)

class LambdaScheduler(LambdaLR):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

def get_cos_lr(step, period=1000, peak=20, lr=1e-4, ratio=0.2):
    T = step // period
    decay = 2 ** T
    step = step % period
    if step <= peak and T>0:
        mul = step / peak
    else:
        mul = (1-ratio) * (np.cos((step - peak) / (period - peak) * math.pi) * 0.5 + 0.5) + ratio
    return lr * mul / decay

def get_multistep_lr(step, period=1000, lr=1e-4, milestone=[500, 900], gamma=[0.5, 0.1], decay_base=1):
    decay = decay_base ** (step // period)
    step = step % period
    mul = 1
    for i in range(len(milestone), 0, -1):
        if step > milestone[i-1]:
            mul = gamma[i-1]
            break
    return lr * mul / decay
