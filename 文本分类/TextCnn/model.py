import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCnn(nn.Module):
    def __init__(self,args):
        super(TextCnn, self).__init__()
        self.args = args
        label_num = args.label_num
        filter_num = args.filte_num
        filter_size = [int(fsz) for fsz in args.filter_size(",")]
        vocab_size = args.vocab_size
        embedding
        pass

    def forward(self, x):
        pass
