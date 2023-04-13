import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def Tsne_graph(fname):
    f = open(fname)
    lines = f.readlines()
    label = []
    embedding = []

    print("loading data")

    for line in lines:
        data = line.split("\t")

        label.append(data[0])
        data[1] = eval(data[1].replace("\n", ""))
        embedding.append(data[1])

    print("End data formating")

    tsne = TSNE()
    tsne.fit_transform(embedding)  # 进行数据降维
    tsne = pd.DataFrame(tsne.embedding_)  # 转换数据格式
    print("data transformation")

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 不同类别用不同颜色和样式绘图
    print("plot picture")

    n1 = 0
    n2 = 0
    n3 = 0

    for i in range(len(label)):
        d = tsne.iloc[i]
        if label[i] == '0':
            s1 = plt.scatter(d[0], d[1], c='c', marker='.')
            n1 += 1
        elif label[i] == '1':
            s2 = plt.scatter(d[0], d[1], c='g', marker='.')
            n2 += 1
        elif label[i] == '2':
            s3 = plt.scatter(d[0], d[1], c='m', marker='.')
            n3 += 1

    print("End plot")

    if "normal" in fname:
        plt.xticks([])
        plt.yticks([])
        plt.legend((s1, s2, s3), ('Negative', 'Neutral', 'Positive'), loc='upper left')
        plt.savefig('bert_sup_res_' + '.png', dpi=600)

    plt.show()

Tsne_graph("./save_result_text/bert_spc_cl_normal_0.8705_0.8078.txt")