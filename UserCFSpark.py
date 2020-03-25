# import findspark
# findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
import math
import argparse

parser = argparse.ArgumentParser(description='UserCF Algorthim.')
parser.add_argument('mode', help='run mode train or test')
parser.add_argument('bid', help='business Id')
parser.add_argument('input', help='Input dataset path')
#parser.add_argument('output', help='output path')
args = parser.parse_args()


def SetLogger(sc):
    logger=sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)


def CreateSparkContext():
    sparkConf=SparkConf().setAppName("RecommendUserCF").set("spark.ui.showConsoleProgress", False)
    sc=SparkContext(conf=sparkConf)
    print("master=", sc.master)
    SetLogger(sc)
    #SetPath(sc)
    return sc

def PrepareTrainData(sc):
    rawRatingData = sc.textFile(Path)
#     header = rawRatingData.first()
#     print(header)
#     rawRatingData =  rawRatingData.filter(lambda line:line != header)
    rawRatings = rawRatingData.map(lambda line: line.split("\t"))
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[3])).sortBy(lambda x: x[2], True)
    return ratingsRDD

def PrepareTestData(sc):
    rawRatingData = sc.textFile(Path)
    #     header = rawRatingData.first()
    #     print(header)
    #     rawRatingData =  rawRatingData.filter(lambda line:line != header)
    rawRatings = rawRatingData.map(lambda line: line.split("\t"))
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1]))
    return ratingsRDD

def PrePareSimData(sc):
    rawRatingData = sc.textFile("hdfs://master:8020/lwqtest/"+args.bid+"/config/UserCF/UserCF_Sim")
    rawRatings = rawRatingData.map(lambda line: line.split(":"))
    SimRDD = rawRatings.map(lambda x: ((x[0], x[1]),float(x[2])))
    return SimRDD
def PrePareBestK(sc):
    K = sc.textFile("hdfs://master:8020/lwqtest/"+args.bid+"/config/UserCF/UserCF_K")
    K = int(K.collect()[0])
    return K

def toCSVLine(data):
    return ':'.join(str(d) for d in data)

def clacSim(list_1,dict_1,l):
    k=0
    for i in list_1:
        k = k + 1/ math.log( 1 + dict_1[i])
    k = k / math.sqrt(l)
    return k

def RecallAndPrecision(rec_list,test):
    hit = 0
    RecallAll = 0
    PrecisionAll = 0
    for user in rec_list.keys():
        if user in test:
            hit += len(list(set(rec_list[user]).intersection(set(test[user]))))
            RecallAll += len(rec_list[user])
            PrecisionAll += len(test[user])
    return (hit / (RecallAll * 1.0),hit / (PrecisionAll * 1.0))


def give_rank(item_list, rank):
    tmp_list = []
    for item in item_list:
        tmp_list.append((item, rank))
    return tmp_list

def find(user, user_list):
    for u in user_list:
        if u == user:
            return True
    return False

def RecDicToList(dic):
    list_1=list()
    for key in dic.keys():
        for value in dic[key]:
            list_1.append((key,value))
    return list_1

global Path

if __name__ == '__main__':
    print(args.mode)
    Path = args.input
    sc = CreateSparkContext()
    if args.mode == 'train':
        ratingsRDD = PrepareTrainData(sc)
        ratingsRDDLen = ratingsRDD.count()
        train = ratingsRDD.take(int(ratingsRDDLen * 8 / 10))
        train = sc.parallelize(train)
        test = ratingsRDD.subtract(train)
        train = train.map(lambda x: (x[0], x[1]))
        test = test.map(lambda x: (x[0], x[1]))
        # train = ratingsRDD.sample(False, 0.9, 10)
        # test = ratingsRDD.subtract(train)
        test = dict(test.groupByKey().map(lambda x: (x[0], list(x[1]))).collect())
        # train

        userItemRDD = train.groupByKey().map(lambda x: (x[0], list(x[1])))
        itemUserDIC = dict(ratingsRDD.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1]))).map(
            lambda x: (x[0], len(x[1]))).collect())
        puserItemRDD = userItemRDD.cartesian(userItemRDD).filter(lambda x: x[0][0] != x[1][0])

        puserItemRDD = puserItemRDD.map(
            lambda x: (x[0][0], x[1][0], list(set(x[0][1]).intersection(set(x[1][1]))), len(x[0][1]) * len(x[1][1])))
        puserItemRDD = puserItemRDD.filter(lambda x: len(x[2]) != 0 and x[3] != 0)
        userSimRDD = puserItemRDD.map(lambda x: (x[0], x[1], clacSim(x[2], itemUserDIC, x[3])))
        print(userSimRDD.count())
        lines = userSimRDD.map(toCSVLine)
        lines.saveAsTextFile("hdfs://master:8020/lwqtest/"+args.bid+"/config/UserCF/UserCF_Sim")
        userSimRDD = userSimRDD.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1])))
        #hdfs://master:8020/lwqtest/1/model/UserCF
        #/Users/liweiqiang/PycharmProjects/sparkCF
        K = [5, 10, 20, 40, 80, 160]
        N = 50
        userset = set(train.map(lambda x: x[0]).collect())
        result_recall = []
        result_precision = []
        BestRank = dict()
        for k in K:
            BestRank.clear()
            for user_id in userset:
                neighbors_list = userSimRDD.filter(lambda x: x[0] == user_id).flatMap(lambda x: x[1]).sortBy(lambda x: x[1], False).take(k)
                neighbors_list = sc.parallelize(neighbors_list)
                neigh = neighbors_list.map(lambda x: x[0]).collect()
                user_item_rec = userItemRDD.filter(lambda x: find(x[0], neigh)).union(neighbors_list).groupByKey().map(
                    lambda x: (x[0], list(x[1]))).map(lambda x: x[1])
                user_item_seen = userItemRDD.filter(lambda x: x[0] == user_id).map(lambda x: x[1]).collect()
                user_item_rec = user_item_rec.flatMap(lambda x: give_rank(x[0], x[1])).reduceByKey(
                    lambda x, y: x + y).filter(lambda x: not find(x[0], user_item_seen))
                user_item_rec = user_item_rec.sortBy(lambda x: x[1], False).map(lambda x: x[0]).take(N)
                # print(user_item_rec)
                BestRank.update({user_id: user_item_rec})
            # print(BestRank[k],'*************************************************************************')
            (recall, precision) = RecallAndPrecision(BestRank, test)
            #print(k, recall, precision)
            result_recall.append(recall)
            result_precision.append(precision)
        total = 0
        position = 0
        for i in range(len(K)):
            rp = result_recall[i] + result_precision[i]
            if total < result_recall[i] + result_precision[i]:
                total = rp
                position = i
        BestK =sc.parallelize([str(K[position])])
        BestK.coalesce(1).saveAsTextFile("hdfs://master:8020/lwqtest/"+args.bid+"/config/UserCF/UserCF_K")

    if args.mode =='test':
        userSimRDD =PrePareSimData(sc)
        userSimRDD = userSimRDD.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().map(lambda x: (x[0], list(x[1])))
        k=PrePareBestK(sc)
        ratingsRDD = PrepareTestData(sc)
        userset = set(ratingsRDD.map(lambda x: x[0]).collect())
        userItemRDD = ratingsRDD.groupByKey().map(lambda x: (x[0], list(x[1])))
        rec_list_dic = dict()
        N=50
        for user_id in userset:
            neighbors_list = userSimRDD.filter(lambda x: x[0] == user_id).flatMap(lambda x: x[1]).sortBy(lambda x: x[1],False).take(k)
            neighbors_list = sc.parallelize(neighbors_list)
            neigh = neighbors_list.map(lambda x: x[0]).collect()
            user_item_rec = userItemRDD.filter(lambda x: find(x[0], neigh)).union(neighbors_list).groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: x[1])
            user_item_seen = userItemRDD.filter(lambda x: x[0] == user_id).map(lambda x: x[1]).collect()
            user_item_rec = user_item_rec.flatMap(lambda x: give_rank(x[0], x[1])).reduceByKey(lambda x, y: x + y).filter(lambda x: not find(x[0], user_item_seen)).sortBy(lambda x: x[1], False).map(lambda x: x[0]).take(N)
            if user_item_rec:
                print(user_item_rec)
                rec_list_dic.update({user_id: user_item_rec})
        print(rec_list_dic)

        rec_list = RecDicToList(rec_list_dic)
        rec_list_rdd = sc.parallelize(rec_list)
        lines = rec_list_rdd.map(toCSVLine)
        lines.saveAsTextFile("hdfs://master:8020/lwqtest/"+args.bid+"/model/UserCF")




