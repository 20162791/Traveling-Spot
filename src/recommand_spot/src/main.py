import gensim
import os
import operator
from textRank import RawTaggerReader, TextRank

def extractReview(review_path):
    tr = TextRank(window=5, coef=1)
    print('Load...')
    stopword = set([('있', 'VV'), ('하', 'VV'), ('되', 'VV'), ('없', 'VV')])
    tr.load(RawTaggerReader(review_path), lambda w: w not in stopword and (w[1] in ('NNG', 'NNP')))
    print('Build...')
    tr.build()
    k = tr.extract(0.1)
    sorted_keyword = sorted(k, key=k.get, reverse=True)[:8]
    keyword = []
    for kw in sorted_keyword:
        temp = ''
        for k in kw:
            if len(temp) is not 0:
                temp += ' '
            if k[1] == 'VA' or k[1] == 'VV':
                temp += k[0] + '다'
            else:
                temp += k[0]
        keyword.append(temp)
    return keyword

def readKeyword(path):
    file = open(path, 'r')
    word = file.readlines()
    keyword = []
    for w in word:
        keyword.append(w.replace('\n', ''))
    return keyword

if __name__ == '__main__':

    # 사용자 리뷰 키워드 추출
    PATH_REVIEW = '../data/user_review/review.txt'
    temp = extractReview(PATH_REVIEW)
    user_keyword = []
    for t in temp:
        user_keyword += t.split(' ')
    print('User Review Keywords =', user_keyword)

    # 사용자 리뷰 키워드, 여행지 리뷰 키워드 유사도 측정
    model = gensim.models.Word2Vec.load('C:/Users/wlsdu/Desktop/TravelSpot/data/model/Word2vec_iter5.model')
    PATH = "../data/keyword/"
    spot_file_list = os.listdir(PATH)

    for uk in user_keyword:
        print('User keyword is', uk)
        similarity = {}
        for file in spot_file_list:
            spot_keyword = readKeyword(PATH+file)
            k = []
            for s in spot_keyword:
                k+=(s.split(' '))
            # print(k)
            s = []
            for sk in k:
                try:
                    r = model.wv.similarity(uk, sk)
                    s.append(r)
                except:
                    s.append(0)
            try:
                similarity[file] = max(s)
            except:
                similarity[file] = 0
        # print(similarity)
        max_spot = max(similarity.items(), key=operator.itemgetter(1))[0]
        print('가장 유사도가 높은 여행지:', max_spot, "유사도:", similarity[max_spot])

    # print(model.wv.similarity('작품', '예술'))

    pass