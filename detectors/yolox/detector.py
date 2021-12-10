from yolox.exp import get_exp
import torch

class Yolox(object):
    def __init__(self, exp_file, weight_path):
        super(Yolox, self).__init__()
        self.exp = get_exp(exp_file)

        self.model = self.exp.get_model()
        self.model.cuda().eval()

        ckpt = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
