# !/usr/bin/python
# coding = utf-8


import gensim
import jieba
from gensim.models.doc2vec import Doc2Vec
from gensim import corpora
from sklearn import cluster



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
    jieba.load_userdict('/home/wangzeju/桌面/lunwen/self_dict.txt')
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
    model_dbom = gensim.models.Doc2Vec(x_train, dm=0, alpha=0.1, min_count=0, window=3, vector_size=size,
                                       min_alpha=0.025)
    # 模型的初始化，设置参数
    # 提供x_train可初始化,min_cout忽略总频率低于这个的所有单词,window预测的词与上下文词之间最大的距离, 用于预测size特征向量的维数
    model_dbom.train(x_train, total_examples=model_dbom.corpus_count, epochs=10)
    # corpus_count是文件个数  epochs 训练次数
    model_dbom.save('/home/wangzeju/桌面/lunwen/doc2vec.model')
    return model_dbom


    # 第三步，生成文档的向量表达
def Test(result):
    model_dbom = Doc2Vec.load("/home/wangzeju/桌面/lunwen/doc2vec.model")
    doc_id = 1
    doc_vectors = []
    # 加载训练的模型
    for text in result:
        #  print(text)
        vector = model_dbom.infer_vector([text])
        doc_vectors.append(vector)
        doc_id += 1
    # 返回二维数组
    return doc_vectors


def Kmeans(doc_vectors,k):
    data = doc_vectors
    clst = cluster.KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=100,tol=1e-4,random_state=0)
    clst.fit(data)
    data_labels = clst.labels_
    return data_labels   # 返回一维数组，元素为每个文档的标签


# 该函数要根据预测的标签值，将全部数据划分到各个类簇中，返回类簇的集合
def GetData(data_labels):
    # 为每个文档赋予标签值
    data_get_labels = {}
    i = 1
    for data_label in data_labels:
        data_get_labels[str(i) + '.txt'] = data_label
        i += 1
    # 将文档根据标签值聚类，相同标签的存入一个列表
    clusters_data = []
    clusters0 = []
    clusters1 = []
    clusters2 = []
    clusters3 = []
    clusters4 = []
    clusters5 = []
    clusters6 = []
    clusters7 = []
    clusters8 = []
    clusters9 = []
    clusters10 = []
    clusters11 = []
    clusters12 = []
    clusters13 = []
    clusters14 = []
    clusters15 = []
    clusters16 = []
    clusters17 = []
    clusters18 = []
    clusters19 = []
    clusters20 = []
    clusters21 = []
    clusters22 = []
    clusters23 = []
    clusters24 = []
#    clusters25 = []
#    clusters26 = []
#    clusters27 = []
#    clusters28 = []
#    clusters29 = []
    for key in data_get_labels.keys():
        if data_get_labels[key] == 0:
            clusters0.append(key)
        elif data_get_labels[key] == 1:
            clusters1.append(key)
        elif data_get_labels[key] == 2:
            clusters2.append(key)
        elif data_get_labels[key] == 3:
            clusters3.append(key)
        elif data_get_labels[key] == 4:
            clusters4.append(key)
        elif data_get_labels[key] == 5:
            clusters5.append(key)
        elif data_get_labels[key] == 6:
            clusters6.append(key)
        elif data_get_labels[key] == 7:
            clusters7.append(key)
        elif data_get_labels[key] == 8:
            clusters8.append(key)
        elif data_get_labels[key] == 9:
            clusters9.append(key)
        elif data_get_labels[key] == 10:
            clusters10.append(key)
        elif data_get_labels[key] == 11:
            clusters11.append(key)
        elif data_get_labels[key] == 12:
            clusters12.append(key)
        elif data_get_labels[key] == 13:
            clusters13.append(key)
        elif data_get_labels[key] == 14:
            clusters14.append(key)
        elif data_get_labels[key] == 15:
            clusters15.append(key)
        elif data_get_labels[key] == 16:
            clusters16.append(key)
        elif data_get_labels[key] == 17:
            clusters17.append(key)
        elif data_get_labels[key] == 18:
            clusters18.append(key)
        elif data_get_labels[key] == 19:
            clusters19.append(key)
        elif data_get_labels[key] == 20:
            clusters20.append(key)
        elif data_get_labels[key] == 21:
            clusters21.append(key)
        elif data_get_labels[key] == 22:
            clusters22.append(key)
        elif data_get_labels[key] == 23:
            clusters23.append(key)
#        elif data_get_labels[key] == 24:
#            clusters24.append(key)
#        elif data_get_labels[key] == 25:
#            clusters25.append(key)
#        elif data_get_labels[key] == 26:
#            clusters26.append(key)
#        elif data_get_labels[key] == 27:
#            clusters27.append(key)
#        elif data_get_labels[key] == 28:
#            clusters28.append(key)
        else:
            clusters24.append(key)
#    print(clusters0)
#    print(clusters1)
#    print(clusters2)
#    print(clusters3)
#    print(clusters4)
#    print(clusters5)
#    print(clusters6)
#    print(clusters7)
    clusters_data.append(clusters0)
    clusters_data.append(clusters1)
    clusters_data.append(clusters2)
    clusters_data.append(clusters3)
    clusters_data.append(clusters4)
    clusters_data.append(clusters5)
    clusters_data.append(clusters6)
    clusters_data.append(clusters7)
    clusters_data.append(clusters8)
    clusters_data.append(clusters9)
    clusters_data.append(clusters10)
    clusters_data.append(clusters11)
    clusters_data.append(clusters12)
    clusters_data.append(clusters13)
    clusters_data.append(clusters14)
    clusters_data.append(clusters15)
    clusters_data.append(clusters16)
    clusters_data.append(clusters17)
    clusters_data.append(clusters18)
    clusters_data.append(clusters19)
    clusters_data.append(clusters20)
    clusters_data.append(clusters21)
    clusters_data.append(clusters22)
    clusters_data.append(clusters23)
    clusters_data.append(clusters24)
#    clusters_data.append(clusters25)
#    clusters_data.append(clusters26)
#    clusters_data.append(clusters27)
#    clusters_data.append(clusters28)
#    clusters_data.append(clusters29)
#    print(clusters_data)
    return clusters_data  # 输出为二维数组


def OpenandGetdata(train_data):
    stop_path = '/home/wangzeju/桌面/lunwen/stop.txt'
    with open(stop_path) as f:
        stop_list_ = f.readlines()
        stop_list = [i.replace('\n', '') for i in stop_list_]
    lda_data_ = []
    for each in train_data:
        each_cut = jieba.cut(each)
        each_result = [word for word in each_cut if word not in stop_list]
        #        print(each_result)
        lda_data_.append(each_result)
    # 输出为二维数组，每个元素是一篇文档
    i = 1
    lda_data = {}
    for ldadata in lda_data_:
        lda_data[str(i)+'.txt'] = ldadata
        i += 1
    # 返回词典，用于LDA分析，内容为{'xxx.txt':['','','',...],......}
#    print(lda_data)
    return lda_data

    
    # 初始化用于LDA分析的数据，将字典{'xxx.txt':['','',...]}中的数据按照簇中的索引值，将列表存入一个列表，构成二维数组
def Init_lda_data(cluster_data,lda_data):
    initldadata = []
    for data_id in cluster_data:
        initldadata.append(lda_data[data_id])
    # 返回二维数组
#    print(initldadata)
    return initldadata


    # 分词
def Lda(initldadata):
    # 加载gensim
    # 使用gensim.Dictionary从text_data中生成一个词袋（bag-of-words)
#    print(initldadata)
    dictionary = corpora.Dictionary(initldadata)
#    print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in initldadata]
#    print(corpus)
    # 加载gensim，使用LDA算法求得前五的topic，
    # 同时生成的topic在之后也会被使用到来定义文本所属主题 NUM_TOPICS = 5
    # 定义了生成的主题词的个数
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=1,id2word=dictionary)
    ldamodel.save('/home/wangzeju/桌面/lunwen/lda_model.gensim')
    topics = ldamodel.print_topics(num_words=100)
    for topic in topics:
        print(topic)
#

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
    data_labels = Kmeans(doc_vectors,25)
    clusters_data = GetData(data_labels)
#    print(len(clusters_data))
    lda_data = OpenandGetdata(train_data)
#    print(clusters_data)
#    print(lda_data)
    j = 1
    for cluster_data in clusters_data:
#        print(cluster_data)
        initldadata = Init_lda_data(cluster_data,lda_data)
        print(j)
#        print(initldadata)
        j += 1
        Lda(initldadata)
        print('----------------------------------------------------')
#        print(j)
#


