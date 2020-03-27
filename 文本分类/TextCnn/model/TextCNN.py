import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCnn(nn.Module):
    def __init__(self, args):
        super(TextCnn, self).__init__()
        self.args = args
        label_num = args.label_num
        filter_num = args.filte_num
        filter_size = [int(fsz) for fsz in args.filter_size(",")]
        vocab_size = args.vocab_size
        embedding_dim = args = args.embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args.static:  #是否加载预训练模型
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        '''
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        '''
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,filter_num,(fsz,embedding_dim)) for fsz in filter_size]
        )
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_size)*filter_num,label_num)
        pass

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        x = self.embedding(x) # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0),1,x.size,self.args.embedding_dim)
        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        '''
        F.relu和nn.Relu 从功能上来讲没有本质区别，F.relu是封装好的函数，nn.ReLu是一个层，依赖于nn.Module，F.relu在forward过程是不存在参数的
        '''
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item,kernal_size=(x_item.size(2),x_item.size(3))) for x_item in x]
        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0),-1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)

        #dropout层
        x = self.dropout(x)

        # 全连接层
        logits = self.linear(x)
        return logits