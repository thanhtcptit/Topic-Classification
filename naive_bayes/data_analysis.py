from os import listdir
from wordcloud import WordCloud
from utils import *
import matplotlib.pyplot as plt
from math import log
import re

topics = ['Chính_trị', 'Giáo_dục', 'Khoa_học_công_nghệ', 'Khác', 'Kinh_tế', 'Pháp_luật'
             , 'Thể_thao', 'Văn_hóa', 'Y_tế']
topic_words = {0: {}, 1 : {}, 2 : {}, 3 : {}, 4 : {}
            , 5 : {} , 6 : {}, 7 : {}, 8 : {}}
pc = {0: 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0
            , 5: 0 , 6 : 0, 7 : 0, 8 : 0}

total_words = {}
total_words_occur = {}
list_words = []

def get_text_from_topic(topic):
    text = ''
    topic_path = TAG_WORDS_DATA_DIR + topic
    for f in listdir(topic_path):
        dir = topic_path + '/' + f
        with open(dir, 'r') as file:
            content = file.read().lower()
            text += content + ' '
    return text

def analysis():
    global total_words, list_words
    for i, topic in enumerate(topics):
        topic_path = TAG_WORDS_DATA_DIR + topic + '/'
        for f in listdir(topic_path):
            pc[i] += 1
            with open(topic_path + f, 'r') as file:
                content = file.read().lower()
                content = re.sub('[":\\()<>|?*…"%^=+!“”.,&*--:0123456789‘’\']|(\\\)', '', content)
                content = list(set(content.split()))
                for w in content:
                    if w in topic_words[i].keys():
                        topic_words[i][w] += 1
                    else:
                        topic_words[i][w] = 1

                    if w in total_words.keys():
                        total_words[w] += 1
                    else:
                        total_words[w] = 1

    list_words = [(k, total_words[k]) for k in sorted(total_words,
                    key=total_words.get, reverse=True)]

def show_words_from_topics(topic):
    print ([(k, topic_words[topic][k]) for k in sorted(topic_words[topic], key=topic_words[topic].get, reverse=True)])

def stop_words():
    stop_word = [k[0] for k in list_words][:200]
    with open(STOP_WORDS, 'w') as f:
        f.write('\n'.join(stop_word))

def show_all_words():
    print(list_words)
    print('Number of words : %i' % len(list_words))
    print('Total : %i words' % sum(total_words.values()))

def tfidf():
    tfidf = {}
    N = sum(pc.values())
    max_tf = max(total_words.values())
    for w in topic_words[0]:
        tfidf[w] = topic_words[0][w] * log(N / total_words[w])
    print ([(k, tfidf[k]) for k in sorted(tfidf, key=tfidf.get, reverse=True)])


def show_wordcloud(topic, idx, show=False):
    text = get_text_from_topic(topic)
    wordcloud = WordCloud(font_path='/usr/share/stellarium/data/DejaVuSans.ttf',
                          background_color='white',
                          width=1200,
                          height=1000, max_words=100
                          ).generate(text)
    plt.figure(idx)
    plt.imshow(wordcloud)
    plt.axis('off')
    if show:
        plt.show()

if __name__ == '__main__':
    analysis()
    tfidf()