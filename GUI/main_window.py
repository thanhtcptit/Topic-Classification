from tkinter import *
from tkinter import ttk
from GUI.process_window import process_window
from naive_bayes.multinomial_bayes import MB
from naive_bayes.data_preprocess import train_test_split
from utils import *

# Cửa sổ chính
class main_window():

    def __init__(self):
        self.topics = ['Chính trị', 'Giáo dục', 'Công nghệ', 'Khác', 'Kinh tế', 'Pháp luật'
        , 'Thể thao', 'Văn hóa', 'Y tế']

        ''' Kiểu đầu vào
            0: URL
            1: Local disk
            2: Text
        '''
        self.input_type = 0
        self.input_cache = 0
        self.content = ''
        # Khởi tạo multinomial bayes
        self.clf = MB(self.topics)
        #features_train, features_test, labels_train, labels_test = preprocess(TAG_WORDS_DATA_DIR, 0.1)
        self.clf.load_pre_trained_data()

        self.top = Tk()
        self.top.title('Article Classification')
        self.top.geometry('600x520')

        self.process_btn = Button(self.top, text='Process', command=self.process_btn_click)
        self.process_btn.place(x=500, y=480)

        var = IntVar()
        self.link_rb = Radiobutton(self.top, text='Link', variable=var, value=1, command=self.link_btn_click)
        self.link_rb.place(x=0, y=10)
        self.link_rb.select()
        self.text_rb = Radiobutton(self.top, text='Text', variable=var, value=2, command=self.text_btn_click)
        self.text_rb.place(x=0, y=100)
        self.text = Text(self.top, width=75)
        self.text.place(x=20, y=120)

        self.notebook = ttk.Notebook(self.top)
        self.page1 = ttk.Frame(self.notebook)
        self.url_label = Label(self.page1, text=' Website:  ')
        self.url_label.pack(side=LEFT)
        self.url_entry = Entry(self.page1, width=50)
        self.url_entry.pack(side=RIGHT)

        self.page2 = ttk.Frame(self.notebook)
        self.local_label = Label(self.page2, text=' Location:  ')
        self.local_label.pack(side=LEFT)
        self.local_entry = Entry(self.page2, width=50)
        self.local_entry.pack(side=RIGHT)

        self.notebook.add(self.page1, text='URL')
        self.notebook.add(self.page2, text='Local')
        self.notebook.place(x=25, y=35)
        self.notebook.bind('<ButtonRelease-1>', self.tab_event)

        self.link_btn_click()
        self.top.mainloop()

    # Lưu kiểu dữ liệu đầu vào
    def tab_event(self, event=None):
        if event != None:
            index = event.widget.index('@%d,%d' % (event.x, event.y))
            if event.widget.tab(index, 'text') == 'Local':
                self.input_type = 1
            else:
                self.input_type = 0
            self.input_cache = self.input_type

    # Lưu kiểu dữ liệu đầu vào
    def link_btn_click(self):
        self.text.config(state=DISABLED)
        self.text.config(background='gray')
        self.url_entry.config(state=NORMAL)
        self.local_entry.config(state=NORMAL)
        self.input_type = self.input_cache

    # Lưu kiểu dữ liệu đầu vào
    def text_btn_click(self):
        self.text.config(state=NORMAL)
        self.text.config(background='white')
        self.url_entry.config(state=DISABLED)
        self.local_entry.config(state=DISABLED)
        self.input_type = 2

    # Mở cửa sổ xử lý
    def process_btn_click(self):
        if self.input_type == 0:
            self.content = self.url_entry.get()
        elif self.input_type == 1:
            self.content = self.local_entry.get()
        else:
            self.content = self.text.get('1.0', END)
        process_window(self.input_type, self.content, self.clf)

if __name__ == '__main__':
    mw = main_window()