import re

import numpy as np
import numpy.random as npr

info8 = np.iinfo(np.int8)

def clear_string(msg):
    msg = re.sub("^(b|0x?)", "", msg)  # suppression des éléments de formatage
    msg = re.sub("(\\(\\d*\\))$", "", msg)  # suppression des bases entre parenthèses
    msg = re.sub("\\s", "", msg)
    return msg


def format_message(msg, base):
    msg = msg[::-1]
    v = ' '.join(msg[i:i + 4] for i in range(0, len(msg), 4))[::-1]
    if base == 2:
        return "b{}".format(v)
    elif base == 8:
        return "0{}".format(v)
    elif base == 16:
        return "0x{}".format(v)
    else:
        return "{} ({})".format(v, base)


def format_value(value, base):
    if value < -256 or value > 255:
        return "OVF"
    value = np.uint8(value)
    if base == 2:
        v = np.binary_repr(value, width=8)
    else:
        v = np.base_repr(value, base)
    return format_message(v, base)


def choose_bin_base():
    return npr.choice((2, 8, 16))


def generate_message():
    v = npr.randint(2, size=(4,))
    return ''.join(str(e) for e in v.tostring())[:15]


def generate_divisor():
    return format_message(np.binary_repr(npr.randint(0, 100)), 2)


def get_base():
    return npr.choice((2, 8, 10, 16))


def randbyte():
    return npr.randint(info8.min, info8.max, dtype="int8").item()


def gen_float16():
    mini = -32768
    maxi = 32767
    f = (maxi - mini) * npr.random() + mini
    return np.float16(np.around(f))
