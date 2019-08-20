# !/usr/bin/python
# coding = utf-8
# 用于获取最优的K值


import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec
from matplotlib import pyplot as plt
from matplotlib import font_manager
from sklearn import cluster


my_font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc")
TaggededDocument = gensim.models.doc2vec.TaggedDocument
def GetTrainSet():
    fnames = []
    for i in range(1, 2550):
        fname = '/home/wangzeju/桌面/lunwen/safevent/' + str(i) + '.txt'
        fnames.append(fname)
    # 用来存放语料
    train_data = []
    for name in fnames:
        with open(name) as f:
            data = f.read().replace('\n', '')
            f.close()
            train_data.append(data)
    #   print('append ok!')
    return train_data


    # 分词
def CutDoc(train_data):
    stop_path = '/home/wangzeju/桌面/lunwen/stop.txt'
    with open(stop_path) as f:
        stop_list = f.readlines()
    result = []
    for each in train_data:
        each_cut = jieba.cut(each)
        each_split = ' '.join(each_cut).split()
        each_result = [word for word in each_split if word not in stop_list]
        result.append(' '.join(each_result))
    # 输出为一维数组，每个元素是一篇文档
    # print(result)
    return result


def InitData(result):
    # 每一篇文档需要一个对应的编号
    index = 0
    # 用来存放语料
    x_train = []
    # 由编号映射文档ID的字典
    doc_dict = {}
    for words in result:
        # 语料预处理
        x_train.append(TaggededDocument(words, tags=[index]))
        doc_dict[index] = str(index + 1) + '.txt'
        #        print('append ok!')
        index += 1
    # print(doc_dict)
    return x_train, doc_dict


# 第二步，初始化训练模型的参数，再保存训练结果以释放内存
def Train(x_train, size=100):
    model_dbom = gensim.models.Doc2Vec(x_train, dm=1, alpha=0.1, min_count=0, window=5, vector_size=size,
                                       min_alpha=0.025)
    # 模型的初始化，设置参数
    # 提供x_train可初始化,min_cout忽略总频率低于这个的所有单词,window预测的词与上下文词之间最大的距离, 用于预测size特征向量的维数
    model_dbom.train(x_train, total_examples=model_dbom.corpus_count, epochs=10)
    # corpus_count是文件个数  epochs 训练次数
    model_dbom.save('/home/wangzeju/桌面/lunwen/doc2vec2.model')
    return model_dbom


    # 第三步，生成文档的向量表达
def Test(result):
    model_dbom = Doc2Vec.load("/home/wangzeju/桌面/lunwen/doc2vec2.model")
    doc_id = 1
    doc_vectors = []
    # 加载训练的模型
    for text in result:
        #  print(text)
        vector = model_dbom.infer_vector([text])
        # 将该数组导入到聚类中，以无监督的方式聚类，可以得到类簇，但是每个类簇中文档的id如何确定有待解决😂️
        doc_vectors.append(vector)
        doc_id += 1
    # 返回二维数组
    return doc_vectors


def Kmeans(doc_vectors,k):
    data = doc_vectors
    clst = cluster.KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=100,tol=1e-4,random_state=0)
    clst.fit(data)
    distortion = clst.inertia_
    return distortion


def plot(sses):
    K = [k for k in range(1,201)]
    plt.figure()
    plt.plot(K,sses)
    plt.xlabel('簇数量',fontproperties=my_font)
    plt.ylabel('SSE')
    plt.show()


if __name__ == '__main__':
    # 获取预处理的语料
    train_data = GetTrainSet()
    # 中文分词，分词后的语料库，返回二维数组
    result = CutDoc(train_data)
    # 利用库初始化数据，返回（标签，文档向量）构成的二维数组，文档词典
    x_train, doc_dict = InitData(result)
    # 训练模型
    model = Train(x_train)
    # 测试，（生成文档向量)
    doc_vectors = Test(result)
    # print(doc_vectors)

    sses = []
    for k in range(1,201):
        sse = Kmeans(doc_vectors,k)
        sses.append(sse)
    plot(sses)

