from naive_bayes.data_preprocess import train_test_split, get_data_from_dir, process_text
from math import log
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import time

class MB(object):

    def __init__(self, topics):
        self.number_of_class = len(topics)
        # Từ điển tất cả các từ
        self.total_words = {}
        # Xác suất của mỗi topic sau khi tính toán
        self.p_topics = [0 for i in range(self.number_of_class)]
        # Tổng số lượng các bài viết
        self.N = 0
        # Từ điển các từ của mỗi topic
        self.topics_words = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        self.word_in_document_count = {}
        self.topics_words_score = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        self.topics_vocabulary_size = []
        # Dữ liệu học
        self.features_train = []
        self.labels_train = []
        # Nhãn
        self.topics = topics

    # Đưa dữ liệu vào hệ thống để học
    def fit(self, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train
        self.train(features_train, labels_train, save_to_disk=True)

    # Quá trình học
    def train(self, features_train, labels_train, save_to_disk=False, alpha=1):
        self.total_words = {}
        self.p_topics = [0 for i in range(self.number_of_class)]
        self.N = 0
        self.topics_words = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        self.topics_words_score = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        self.word_in_document_count = {}

        for s, t in zip(features_train, labels_train):
            # Cắt dữ liệu vào thành từng từ, tính xác suất mỗi từ xuất hiện ở mỗi topic
            words = s.split()
            self.p_topics[t] += 1
            #words = list(set(words))
            #  Đếm số lần xuất hiện của từ trong phạm vi topic và toàn từ điển
            for w in words:
                if w in self.topics_words[t].keys():
                    self.topics_words[t][w] += 1
                    self.total_words[w] += 1
                else:
                    self.topics_words[t][w] = 1
                    if w in self.total_words.keys():
                        self.total_words[w] += 1
                    else:
                        self.total_words[w] = 1
            words = list(set(words))
            for w in words:
                if w not in self.word_in_document_count.keys():
                    self.word_in_document_count[w] = 1
                else:
                    self.word_in_document_count[w] += 1

        self.N = sum(self.p_topics)

        #self.feature_selection()
        self.calculate_term_score(alpha)
        #  Lưu điểm của từ điển ra file
        if save_to_disk:
            for i in range(self.number_of_class):
                with open(TRAINED_DIR + self.topics[i] + '_vocabulary.txt', 'w') as f:
                    f.write(str(self.p_topics[i]) + '\n')
                    for w, s in self.topics_words_score[i].items():
                        f.write(w + ' ' + str(s) + '\n')

    # Đọc dữ liệu đã đc train từ trước
    def load_pre_trained_data(self):
        for i in range(self.number_of_class):
            with open(TRAINED_DIR + self.topics[i] + '_vocabulary.txt', 'r') as f:
                self.p_topics[i] = float(f.readline())
                for line in f:
                    if len(line) < 3:
                        continue
                    w = line.split(' ')
                    self.topics_words_score[i][w[0]] = float(w[1])

    def mutual_info(self):
        ig = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}}
        N = sum(self.p_topics)
        for i in range(self.number_of_class):
            for w in self.topics_words[i].keys():
                N11 = self.topics_words[i][w]
                N10 = self.total_words[w] - N11 + 0.0001
                N01 = self.p_topics[i] - N11
                N00 = N - N10 - N01 - N11
                N1x = N10 + N11
                Nx1 = N01 + N11
                N0x = N00 + N01
                Nx0 = N00 + N10
                ig[i][w] = (N11 / N) * log((N * N11) / (N1x * Nx1), 2) + (N10 / N) * log((N * N10) / (N1x * Nx0), 2) \
                    + (N01 / N) * log((N * N01) / (N0x * Nx1), 2) + (N00 / N) * log((N * N00) / (N0x * Nx0), 2)
        return ig

    def feature_selection(self, k=1000):
        ig = self.mutual_info()

        for c, d in ig.items():
            sorted_tuple = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)][:k]
            new_dict = {}
            for t in sorted_tuple:
                new_dict[t[0]] = self.topics_words[c][t[0]]
            self.topics_words[c] = new_dict

        def calculate_term_score(self):
            # Tính tổng só lượng bài viết trong dữ liệu học
            sum_p = sum(self.p_topics.values())
            for i in range(self.number_of_class):
                self.p_topics[i] = log(self.p_topics[i] / sum_p)
            self.topics_vocabulary_size = [sum(self.topics_words[i].values()) + len(self.volcabulary) for i in
                                           self.topics_words.keys()]
            # Tinh điểm của từng từ trong từ điển
            for i in range(self.number_of_class):
                for w in self.total_words.keys():
                    if w in self.topics_words[i].keys():
                        pr = self.topics_words[i][w] + 1
                    else:
                        pr = 1
                    self.topics_words_score[i][w] = log(pr / self.topics_vocabulary_size[i])

    def calculate_term_score(self,  alpha=1):
        # Tính tổng só lượng bài viết trong dữ liệu học
        title_weight = 2
        sum_p = sum(self.p_topics)
        for i in range(self.number_of_class):
            self.p_topics[i] = log(self.p_topics[i] / sum_p)
        self.topics_vocabulary_size = [sum(self.topics_words[i].values()) for i in
                                       self.topics_words.keys()]
        # Tinh điểm của từng từ trong từ điển
        for i in range(self.number_of_class):
            for w in self.total_words.keys():
                if w in self.topics_words[i].keys():
                    pr = self.topics_words[i][w] + alpha
                    if w + '/t' in self.topics_words[i].keys() or w.find('/t') > 0:
                        pr += title_weight
                else:
                    pr = alpha
                self.topics_words_score[i][w] = log(pr / self.topics_vocabulary_size[i])

    def tfidf(self, words):
        set_words = list(set(words))
        words_score = {}
        for w in set_words:
            tf = sum([1 for i in words if i == w])
            if w in self.word_in_document_count.keys():
                idf = log(self.N / self.word_in_document_count[w])
            else:
                idf = 0
            words_score[w] = (1 + log(tf)) * idf

        sorted_dict = [(k, words_score[k]) for k in sorted(words_score, key=words_score.get, reverse=True)]
        return [k for k, _ in sorted_dict[:int(len(sorted_dict) / 2)]]

    # Dự đoán nhãn một bài viết
    def classify(self, article):
        words = article.split(' ')
        #words = list(set(words))
        #choosen_words = self.tfidf(words)
        #words = [w for w in words if w in choosen_words]
        # Xác suất bài viết thuộc các topic
        pr = self.p_topics.copy()
        for i in self.topics_words.keys():
            for w in words:
                # Log xác suất của từ xuất hiện trong topic / số từ trong từ điển
                if w in self.topics_words_score[i].keys():
                    pr[i] += self.topics_words_score[i][w]
        return np.argmax(pr), pr

    # Test trên bộ test
    def test(self, features_test, labels_test, debug=False):
        predict = []
        skip = {0 :{281, 298, 320, 333, 159, 328, 353, 358, 365, 367, 433, 559, 560, 587, 669}}
        for i, s in enumerate(features_test):
            idx, _ = self.classify(s.lower())
            if idx == 0 and labels_test[i] != 0 and debug:
                if i not in skip[idx]:
                    print (i)
                    self.log_predict_error(s.lower(), idx, labels_test[i])
                    exit()
            predict.append(idx)

        score = self.accuracy_score(labels_test, predict)
        precision = []
        recall = []
        for i in range(self.number_of_class):
            precision.append(self.precision(labels_test, predict, i))
            recall.append(self.recall(labels_test, predict, i))
        confusion_matrix = self.confusion_matrix(labels_test, predict)
        print ('Accuracy score: ', score)
        print ('Precision: ', precision)
        print ('Recall: ', recall)
        print ('F1 score: ', self.f1_score(precision, recall))
        print ('Confusion matrix :\n', confusion_matrix)

        return score, precision, recall, confusion_matrix

    def log_predict_error(self, article, predict, label):
        words = article.split(' ')
        words = list(set(words))
        sum_p = 0.
        sum_l = 0.
        words_p = []
        words_l = []
        with open(LOG, 'w') as f:
            f.write(article)
            f.write('---------------------------------\n')
            f.write('Predict: %s | Label: %s\n' % (self.topics[predict], self.topics[label]))
            for w in words:
                w_p = np.NaN
                w_l = np.NaN
                if w in self.topics_words_score[predict]:
                    w_p = self.topics_words_score[predict][w]
                    sum_p += self.topics_words_score[predict][w]
                if w in self.topics_words_score[label]:
                    w_l = self.topics_words_score[label][w]
                    sum_l += self.topics_words_score[label][w]
                if np.isnan(w_p) and np.isnan(w_l):
                    continue
                if w_p > w_l:
                    words_p.append('{} : {} | {} (*)'.format(w, round(w_p, 3), round(w_l, 3)))
                else:
                    words_l.append('{} : {} | {}'.format(w, round(w_p, 3), round(w_l, 3)))
            f.write('\n'.join(words_p))
            f.write('\n---------------------------------\n')
            f.write('\n'.join(words_l))
            f.write('\nTotal score: %f | %f' % (sum_p + self.p_topics[predict], sum_l + self.p_topics[label]))

    # Cross validate với dữ liệu đc chia thành 10 folds
    def cross_validate_test(self, folds=10, alpha=1):
        av_score = 0.0
        av_p = [0 for i in range(self.number_of_class)]
        av_r = [0 for i in range(self.number_of_class)]
        av_confusion_matrix = np.zeros((self.number_of_class, self.number_of_class))

        features_folds = []
        labels_folds = []
        for i in range(folds):
            feature, label = get_data_from_dir(CROSS_VALIDATE_DIR + str(i + 1) + '/')
            features_folds.append(feature)
            labels_folds.append(label)

        for i in range(folds):
            print('Use fold %i as test data' % (i + 1))
            cv_features_train = []
            cv_labels_train = []
            for j in range(folds):
                if i == j:
                    cv_features_test = features_folds[j]
                    cv_labels_test = labels_folds[j]
                else:
                    train_data = features_folds[j]
                    label = labels_folds[j]
                    cv_features_train += train_data
                    cv_labels_train += label
            self.train(cv_features_train, cv_labels_train, alpha)
            score, p, r , cm = self.test(cv_features_test, cv_labels_test)
            av_score += score
            av_p = [av_p[i] + p[i] for i in range(self.number_of_class)]
            av_r = [av_r[i] + r[i] for i in range(self.number_of_class)]
            av_confusion_matrix += cm

        av_score /= folds
        av_r = [round(av_r[i] / folds, 3) for i in range(self.number_of_class)]
        av_p = [round(av_p[i] / folds, 3) for i in range(self.number_of_class)]
        av_confusion_matrix /= folds
        print ('--- Average test score with fold = %d ---' % folds)
        print ('Accuracy score: ', round(av_score, 3))
        print ([j[x] / sum(j) for x in range(self.number_of_class) for j in av_confusion_matrix])
        self.show_chart([j[x] / sum(j) for x, j in zip(range(self.number_of_class), av_confusion_matrix)], 'Topic', 'Accuracy', 'Accuracy Score')
        print ('Precision: {} - Average: {}'.format(av_p, sum(av_p) / self.number_of_class))
        print ('Recall: {} - Average: {}'.format(av_r, sum(av_r) / self.number_of_class))
        f1_score = self.f1_score(av_p, av_r)
        self.show_chart(f1_score, 'Topic', 'Score', 'F1-Score')
        print ('F1 score: {} - Average: {}'.format(f1_score, sum(f1_score) / self.number_of_class))
        print ('Confusion matrix: \n', av_confusion_matrix)
        return av_score

    def accuracy_score(self, y_true, y_predict):
        count = sum([1 if y_predict[i] == y_true[i] else 0 for i in range(len(y_true))])
        return round(float(count) / len(y_true), 3)

    def precision(self, y_true, y_predict, c):
        tp = sum([1 if y_predict[i] == y_true[i] and y_predict[i] == c else 0 for i in range(len(y_true))])
        fp = sum([1 if y_predict[i] != y_true[i] and y_predict[i] == c else 0 for i in range(len(y_true))])
        return round(float(tp) / (tp + fp), 3)

    def recall(self, y_true, y_predict, c):
        tp = sum([1 if y_predict[i] == y_true[i] and y_predict[i] == c else 0 for i in range(len(y_true))])
        fn = sum([1 if y_predict[i] != y_true[i] and y_true[i] == c else 0 for i in range(len(y_true))])
        return round(float(tp) / (tp + fn), 3)

    def f1_score(self, pr, rc):
        return [round(float(2 * p * r) / (p + r), 3) for p, r in zip(pr, rc)]

    def confusion_matrix(self, y_true, y_predict):
        confution_matrix = []
        for i in range(self.number_of_class):
            matrix = [0 for i in range(self.number_of_class)]
            for yt, yp in zip(y_true, y_predict):
                if yt == yp and yt == i:
                    matrix[i] += 1
                elif yt != yp and yt == i:
                    matrix[yp] += 1
            if i == 0:
                confution_matrix = np.array(matrix)
            else:
                confution_matrix = np.vstack((confution_matrix, matrix))
        return confution_matrix

    # Đồ thị biểu diễn xác suất
    def show_chart(self, y, xlabel, ylabel, title, offset=0.05):
        x = [i for i in range(self.number_of_class)]
        plt.xticks(x, self.topics, rotation=45, fontsize=7)
        plt.bar(x, y, width=0.35)
        plt.ylim(min(y) - offset, max(y) + offset)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    # Thống kê số lượng từ
    def show_words_count(self, string):
        words = string.split(' ')
        for w in words:
            if (len(w) < 2): continue
            print ('-------------\n Word: ', w)
            for i in range(9):
                print(self.topics[i] + ' :')
                if (w in self.topics_words_score[i]):
                    print(self.topics_words_score[i][w])
                else:
                    print ('not exist')

if __name__ == '__main__':
    topics = ['Chính trị', 'Giáo dục', 'Công nghệ', 'Khác', 'Kinh tế', 'Pháp luật'
        , 'Thể thao', 'Văn hóa', 'Y tế']
    clf = MB(topics)
    clf.cross_validate_test(alpha=1)