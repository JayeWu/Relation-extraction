import re

THRESHOLD = 10


def split_sentence(content):
    sens = re.split('[。！？!?\n\r]', content)
    sens = [re.sub('[^\u4e00-\u9fa5]{20,}', '', x) for x in sens]
    sens = [x for x in sens if len(x) > THRESHOLD]
    return sens
