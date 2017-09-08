import re, random, subprocess
from os import listdir, remove
from shutil import copyfile
from sklearn.model_selection import train_test_split as split
from utils import *

topics = ['Chính_trị', 'Giáo_dục', 'Khoa_học_công_nghệ', 'Khác', 'Kinh_tế', 'Pháp_luật'
             , 'Thể_thao', 'Văn_hóa', 'Y_tế']

# Tách từ sử dụng command line
def segment_words(source, des=TEMP_FILE):
    cmd = 'cd ' + TOKENIZER + ' && ./vnTokenizer.sh -i {} -o {}'
    p = subprocess.Popen(cmd.format(source, des), stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()

# Đánh dấu từ sử dụng command line
def tag_words(source, des=TEMP_FILE):
    cmd = 'cd ' + TAGGER + ' && ./vnTagger.sh -i {} -o {} -u -p'
    p = subprocess.Popen(cmd.format(source, des), stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    return output.decode('utf-8')

# Tách và đánh dầu dữ liệu đầu vào
def process_text(text):
    p = re.compile('[":()<>|?*…"%^!“”.,_&*-=+-:0123456789‘’\']|(\\\)')
    text = p.sub('', text).split('\n')
    with open(DATA_DIR + 'in.txt', 'w') as file:
        for t in text:
            if (len(t) < 3):
                continue
            t = re.sub('\s+', ' ', t)
            file.write(t + '\n')
    tag_words(source=DATA_DIR + 'in.txt')
    with open(TEMP_FILE, 'r') as file:
        text = file.read()
    return text.lower()

# Lấy dữ liệu từ thư mục
def get_data_from_dir(dir):
    with open(STOP_WORDS, 'r') as f:
        stop_words = f.read().split('\n')
    data = []
    types = []
    article = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}}
    for i, topic in enumerate(topics):
        topic_dir = dir + topic
        types += [i for _ in range(0, len(listdir(topic_dir)))]
        for f in listdir(topic_dir):
            file_dir = topic_dir + '/' + f
            with open(file_dir, 'r') as file:
                    article[i][f] = file.read()

    for i, topic in enumerate(topics):
        for s in article[i].values():
            processed_article = ''
            s = s.lower()
            s = re.sub('[":\\()<>–|?*…"%^=+!“”.,&*--:0123456789‘’\']|(\\\)', '', s)
            lines = s.split('\n')
            for i, line in enumerate(lines):
                words = line.split(' ')
                new_line = []
                for j in range(len(words)):
                    if words[j] in stop_words or len(words[j]) < 3:
                        continue
                    if i == 0:
                        new_line.append(words[j] + '/t')
                    else:
                        new_line.append(words[j])
                processed_article += ' '.join(new_line) + '\n'
            data.append(processed_article)
    return data, types

# Tách dữ liệu train và test
def train_test_split(training_dir=TAG_WORDS_DATA_DIR, test_size=0.1):
    data, types = get_data_from_dir(training_dir)
    features_train, features_test, labels_train, labels_test = split(data, types, test_size=test_size, random_state=42)
    return features_train, features_test, labels_train, labels_test

# Chia dữ liệu thành 10 fold
def cross_validate_split(data_dir, folds=10):
    for topic in topics:
        topic_dir = data_dir + topic + '/'
        files = listdir(topic_dir)
        len_files = len(files)
        fold_size = int(len_files / folds)
        for fold in range(1, folds + 1):
            sets = []
            if len_files == 0:
                break
            for i in range(fold_size):
                x = files[random.randint(0, len_files - 1)]
                sets.append(x)
                files.remove(x)
                len_files -= 1
                if len_files == 0:
                    break
            des_folder = CROSS_VALIDATE_DIR + str(fold) + '/' + topic + '/'
            for d in listdir(des_folder):
                remove(des_folder + d)
            for f in sets:
                copyfile(topic_dir + f, des_folder + f)

