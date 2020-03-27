import argparse
import os
import sys
import torch
import torch.nn.functional as F
import data_preprocess
from model import TextCNN

parser = argparse.ArgumentParser(description="TextCNN")

parser.add_argument('-lr', type=float, default=0.001, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-filter-num', type=int, default=100, help='卷积核的个数')
parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='不同卷积核大小')
parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100,help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')

args = parser.parse_args()


def train(args):
    train_iter, dev_iter = data_preprocess.load_data(args)  #将数据集分为训练和测试
    print("-----加载完成-----")
    model = TextCNN.TextCnn(args)
    if args.cude: model.cuda()

    pass



def save(model,path,prefix,steps):
    pass


if __name__ == '__main__':
    train(args)