import jieba
from torchtext import data
import torch
import re
import os
from torchtext.vocab import Vectors

PATH = os.getcwd()

# 分词器
def tokenizer(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = pattern.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


# 去除停用词
def get_stopwords():
    stopwords = []
    file_path = PATH+"/stopwords/baidu_stopwords.txt"
    if os.path.exists(file_path):
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line[:-1]
                line = line.strip()
                stopwords.append(line)
    return stopwords


# 加载数据
def load_data(args):
    print('-----加载数据中-----')
    stop_words = get_stopwords()

    '''
    tokenize传入一个函数，表示如何将文本str变成token
    sequential表示是否切分数据，如果数据已经是序列化的了而且是数字类型的，则应该传递参数use_vocab = False和sequential = False

    '''
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer, stop_words=stop_words)
    label = data.Field(sequential=False)
    text.tokenize = tokenizer
    train, val = data.TabularDataset.splits(
        path='dataset/',
        skip_header=True,
        train='train.tsv',
        validation='validation.tsv',
        format='tsv',
        fields=[('index', None), ('label', label), ('text', text)],
    )

    if args.static:  # 由于文件夹中没有预训练模型，该路径暂时无效
        text.build_vocab(train, val, vectors=Vectors(
            name="/brucewu/projects/pytorch_tutorials/chinese_text_cnn/data/eco_article.vector"))
        args.embedding_dim = text.vocab.vectors.size()[-1]
        args.vectors = text.vocab.vectors
    else:
        text.build_vocab(train,val)
    label.build_vocab(train,val)
    train_iter, val_iter = data.Iterator.splits(
        (train, val),
        sort_key=lambda x: len(x.text),
        batch_sizes=(args.batch_size, len(val)),  # 训练集设置batch_size,验证集整个集合用于测试
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.vocab_size = len(text.vocab)
    args.label_num = len(label.vocab)
    return train_iter, val_iter
