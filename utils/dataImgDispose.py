import os
import sys
import time

class imgDispose:
    def __init__(self, filename, count):
        self.filename = filename
        self.count = count

    def mkSubFile(self,lines, srcName, sub):
        [des_filename, extname] = os.path.splitext(srcName)
        filename = des_filename + '_' + str(sub) + extname
        print('make file: %s' % filename)
        fout = open(filename, 'w')
        try:
            # fout.writelines([head])
            fout.writelines(str(lines) + "\n")
            return sub + 1
        finally:
            fout.close()

    def splitByLineCount(self,filename, count):
        with open(filename, 'r', encoding='UTF-8') as file:
            buf = []
            for line in file:
                line = line.strip('\n').split(',')
                buf.append(line[0])
                # if len(buf) == count:
                # buf = []
            '''
            f = open("train_1_w.txt", 'w')
            new_buf = list(set(buf))
            for i in range(len(new_buf)):
                writeline = str(new_buf[i]) + "\n"
                f.write(writeline)
            f.close()
            '''
            file_list = []
            for root, dirs, files in os.walk('F:/swork/pyproject/a_pycharm_idea/MyYOLO/MyYOLO/data/train_1w'):
                for file in files:
                    if os.path.splitext(file)[1] == '.jpg':
                        file_list.append(file)
            retD = list(set(file_list).difference(set(buf)))
            print(retD)

            for i in range(len(retD)):
                print(retD[i])
                os.remove(r'F:/swork/pyproject/a_pycharm_idea/MyYOLO/MyYOLO/data/train_1w' + '/' + retD[i])

if __name__ == '__main__':
    begin = time.time()
    filename = "train_1w.txt"
    dataset = imgDispose('train_1w.txt',20)
    generator = dataset.splitByLineCount(filename, 20)
    end = time.time()
    print('time is %d seconds ' % (end - begin))