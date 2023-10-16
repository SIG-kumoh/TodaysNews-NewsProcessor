import re


def remove_tag(target):
    return re.sub('<.+?>', '', str(target), 0).strip()


def simply_ws(target):
    return re.sub('\s+', ' ', str(target)).strip()


def only_BMP_area(t_text: str):
    only_BMP_pattern = re.compile("[\U00010000-\U0010FFFF]+", flags=re.UNICODE)
    res = only_BMP_pattern.sub(r'', t_text)
    return res