import os

dirs = r'D:\rde\enterprise_relation_extraction\dataset\ent_list'
dir_ = r'D:\rde\enterprise_relation_extraction\out_data\famous_persones.txt'


def quchong():
    gss = set()
    file1 = dir_
    # for file in os.listdir(dirs):
    # file1 = os.path.join(dirs, file)
    f = open(file1, 'r', encoding='utf8')

    for line in f.readlines():
        gss.add(line.strip())
    f.close()

    f2 = open(r'D:\rde\enterprise_relation_extraction\out_data\famous_persones2.txt', 'w', encoding='utf8')
    for gs in gss:
        f2.write(gs + '\n')
    f2.close()


def quchong_gongsi():
    files = os.listdir(r'D:\rde\data\公司列表')
    set1 = set()
    for file in files:
        print(file)
        f = open(os.path.join(r'D:\rde\data\公司列表', file), 'r', encoding='utf8')
        for line in f.readlines():
            if line:
                set1.add(line)
        f.close()
    f2 = open(r'D:\rde\data\all_ent_list.txt', 'w', encoding='utf8')
    for gs in set1:
        f2.write(gs)
    f2.close()


def rename():
    for file in os.listdir(dirs):
        file1 = os.path.join(dirs, file)
        f = open(file1, 'r', encoding='utf8')
        lines = f.readlines()
        f.close()
        new_file = file1.replace('quc_', '').replace('.txt', '') + '.txt'
        os.rename(file1, new_file)


if __name__ == '__main__':
    quchong_gongsi()
