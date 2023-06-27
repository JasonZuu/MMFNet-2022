import torch 
import os

from torch.serialization import save

class Base_ProcSystem():
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        pass
    
    def train(self):
        pass

    def verify(self):
        pass

    def test(self):
        pass
    def save_params(self, save_path):
        save_name, exp_name = os.path.splitext(save_path)
        torch.save(self.extractor.state_dict(), f"{save_name}_extractor{exp_name}")
        torch.save(self.frame.state_dict(), f"{save_name}_frame{exp_name}")
        torch.save(self.adaptive_module.state_dict(), f"{save_name}_adaptive_module{exp_name}")

    def load_params(self, params_path):
        """
        不需要指定每个模块的名字，只需要 {root}/{modelname}_{epoch}.pth即可
        """
        params_name, exp_name = os.path.splitext(params_path)
        self.extractor.load_state_dict(torch.load(f"{params_name}_extractor{exp_name}"))
        self.frame.load_state_dict(torch.load(f"{params_name}_frame{exp_name}"))
        adapt_P = torch.load(f"{params_name}_adaptive_module{exp_name}")
        self.adaptive_module.load_state_dict(torch.load(f"{params_name}_adaptive_module{exp_name}"))
        