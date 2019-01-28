from lib.functions import *


class Convert:

    def __init__(self, number=None,base_src=None,base_dst=None):

       self.base_src = npr.randint(2,21) if base_src is None else base_src
       self.base_dst = npr.randint(2,21) if base_dst is None else base_dst
       self.number = format_value(npr.randint(info8.max),self.base_src) if number is None else number

    @property
    def statement(self):
        return '{} = ? ({})'.format(self.number, self.base_dst)

    @property
    def solution(self):
        try:
            n = int(clear_string(self.number), self.base_src)
            if n < info8.min or n > info8.max:
                return "OVF"
            else:
                return format_value(n, self.base_dst)
        except ValueError as e:
            return "IMP"


class Arith:

    def __init__(self,number1=None,base_src1=None,number2=None,base_src2=None,base_dst=None,op=None):
        self.base_src1 = get_base() if base_src1 is None else base_src1
        self.base_src2 = get_base() if base_src2 is None else base_src2
        self.base_dst = get_base() if base_dst is None else base_dst
        self.op = npr.choice(('+', '-', '×', '÷')) if op is None else op
        if number1 is None:
            self.n1 = randbyte()
            self.__number1 = format_value(self.n1, self.base_src1)
        else:
            self.number1 = number1
        if number2 is None:
            self.n2 = randbyte()
            self.__number2 = format_value(self.n2, self.base_src2)
        else:
            self.number2 = number2

    @property
    def number1(self):
        return self.__number1

    @number1.setter
    def number1(self,value):
        self.__number1 = value
        self.n1 = np.int8(int(clear_string(value),self.base_src1)).item()

    @property
    def number2(self):
        return self.__number2

    @number2.setter
    def number2(self,value):
        self.__number2 = value
        self.n2 = np.int8(int(clear_string(value),self.base_src2)).item()

    @property
    def statement(self):
        return "{} {} {} = ? ({})".format(self.number1, self.op, self.number2, self.base_dst)

    @property
    def solution(self):
        try:
            res = self.compute()
            if res < info8.min or res > info8.max:
                return "OVF"
            else:
                return format_value(res, self.base_dst)
        except ValueError as e:
            return "IMP"

    def compute(self):
        if self.op == '+':
            return self.n1 + self.n2
        elif self.op == '-':
            return self.n1 - self.n2
        elif self.op == '×':
            return self.n1 * self.n2
        elif self.op == '÷':
            if self.n1 < 0 or self.n2 <= 0:
                raise ValueError
            else:
                return self.n1 / self.n2
        else:
            raise ValueError


class DecimalToFloat16:

    def __init__(self,src=None,base_dst=None):
        self.src = gen_float16() if src is None else src
        self.base_dst = choose_bin_base() if base_dst is None else base_dst
        self.source = np.float16(src)
        self.base_dst = base_dst

    @property
    def statement(self):
        return "{} => ? ({})".format(self.source, self.base_dst)

    @property
    def solution(self):
        s = bin(self.source.view("H"))[2:].zfill(16)
        bits = int(s, 2)
        if self.base_dst == 2:
            pad = 16
        elif self.base_dst == 8:
            pad = 6
        else:
            pad = 0
        v = np.base_repr(bits, self.base_dst).zfill(pad)
        v = v[::-1]
        v = ' '.join(v[i:i + 4] for i in range(0, len(v), 4))
        if self.base_dst == 2:
            return "b{}".format(v[::-1])
        elif self.base_dst == 8:
            return "0{}".format(v[::-1])
        else:
            return "0x{}".format(v[::-1])


class Float16ToDecimal:
    def __init__(self,src=None,base_src=None):
        self.base_src = choose_bin_base() if base_src is None else base_src
        if src is None:
            tmp = DecimalToFloat16(base_dst=self.base_src)
            self.src = tmp.solution
            self.val = tmp.source
            del tmp
        else:
            self.src = clear_string(src)
            self.val = self.float_from_unsigned16(self.src, self.base_src)

    @property
    def statement(self):
        return "{} => ? (10)".format(self.src)

    @property
    def solution(self):
        return self.val

    @property
    def polynom(self):
        if np.isneginf(self.val):
            return "-INF"
        if np.isposinf(self.val):
            return "+INF"
        if np.isnan(self.val):
            return "NaN"
        if self.val == 0:
            return "0"
        signum = "-(" if self.val < 0 else "+("
        v = np.abs(self.val)
        m = int(np.log2(v))
        if 2 ** m > v:
            m = m - 1
        s = ""
        for i in range(m, m - 10, -1):
            p = 2 ** i
            if v / p >= 1:
                s = s + "+2^{}".format(i)
                v = v - p
        return signum + s[1:] + ")"

    @staticmethod
    def float_from_unsigned16(src, base_src):
        n = int(src, base_src)
        assert 0 <= n < 2 ** 16
        sign = n >> 15
        exp = (n >> 10) & 0b011111
        fraction = n & (2 ** 10 - 1)
        if exp == 0:
            if fraction == 0:
                return -0.0 if sign else 0.0
            else:
                return (-1) ** sign * fraction / 2 ** 10 * 2 ** (-14)  # subnormal
        elif exp == 0b11111:
            if fraction == 0:
                return float('-inf') if sign else float('inf')
            else:
                return float('nan')
        return (-1) ** sign * (1 + fraction / 2 ** 10) * 2 ** (exp - 15)


class HammingMessage:
    def __init__(self,msg=None,base_src=None, base_dst=None,encoded=None):
        self.base_src = choose_bin_base() if base_src is None else base_src
        self.base_dst = choose_bin_base() if base_dst is None else base_dst
        self.encoded = npr.choice((True,False)) if encoded is None else encoded
        if msg is None:
            self.bin_msg = generate_message()
            tmp = np.base_repr(int(self.bin_msg, 2), self.base_src)
            self.msg = format_message(tmp, self.base_src)
        else:
            self.msg = msg
            self.bin_msg = np.binary_repr(int(clear_string(msg), self.base_src))

    @property
    def statement(self):
        return "{} à {} = ? ({})".format(self.msg, "encoder" if self.encoded else "décoder", self.base_dst)

    @property
    def solution(self):
        if self.encoded:
            return format_message(self.encode_message(), self.base_dst)
        else:
            return format_message(self.decode_message(), self.base_dst)

    def encode_message(self):
        msg = [int(i) for i in self.bin_msg[::-1]]
        bits = self.generate_code(msg)
        v = ''.join(str(i) for i in bits)
        v = v[::-1]
        return np.base_repr(int(v, 2), self.base_dst)

    def decode_message(self):
        power = self.parity_count()
        error = self.error_pos
        msg = [str(i) for i in reversed(self.bin_msg)]
        try:
            if error != 0:
                c = msg[error - 1]
                d = '0' if c == '1' else '1'
                msg[error - 1] = d
            for i in range(power - 1, -1, -1):
                del msg[2 ** i - 1]
        except IndexError as e:
            return "Erreur - " + str(e)
        rslt = ''.join(i for i in msg[::-1])
        return np.base_repr(int(rslt, 2), self.base_dst)

    def generate_code(self, msg):
        i = 0
        parity = 0
        while i < len(msg):  # recherche nb bits à ajouter
            if 2 ** parity == i + parity + 1:
                parity = parity + 1
            else:
                i = i + 1
        bits = np.zeros(len(msg) + parity, dtype=int)  # tableau de hamming
        i, j, k = 1, 0, 0
        while i <= len(bits):
            if 2 ** j == i:
                bits[i - 1] = 2  # remplissage des inconnus
                j = j + 1
            else:
                bits[k + j] = msg[k]
                k = k + 1
            i = i + 1
        i = 0
        while i < parity:
            bits[2 ** i - 1] = self.parity(bits, i)
            i = i + 1
        return bits

    def parity_count(self):
        parity = 0
        while 2 ** parity <= len(self.bin_msg):
            parity = parity + 1
        return parity

    @property
    def error_pos(self):
        parity = self.parity_count()
        msg = [int(i) for i in reversed(self.bin_msg)]
        par = np.zeros(parity, dtype=int)
        syndrome = ""
        power = 0
        while power < parity:
            for i in range(len(msg)):
                k = i + 1  # extraction des bits de parité
                s = np.binary_repr(k)
                bit = int((int(s) / 10 ** power) % 10)
                if bit == 1 and msg[i] == 1:
                    par[power] = int((par[power] + 1) % 2)
            syndrome = str(par[power]) + syndrome
            power = power + 1
        return int(syndrome, 2)

    @staticmethod
    def parity(bits, power):
        parity = 0
        for i in range(len(bits)):
            if bits[i] != 2:
                # si la case ne contient pas 2, on récupère l'index en binaire
                s = np.binary_repr(i + 1)
                x = int((int(s) / 10 ** power) % 10)
                if x == 1 and bits[i] == 1:
                    parity = int((parity + 1) % 2)
        return parity


class CrcMessage:
    def __init__(self,msg=None,base_src=None,div=None,base_dst=None,encoded=None):
        self.base_src = choose_bin_base() if base_src is None else base_src
        self.div = generate_divisor() if div is None else div
        self.base_dst = choose_bin_base() if base_dst is None else base_dst
        self.encoded = npr.choice((True,False)) if encoded is None else encoded
        if msg is None:
            self.bin_msg = generate_message()
            tmp = np.base_repr(int(self.bin_msg, 2), self.base_src)
            self.msg = format_message(tmp, self.base_src)
        else:
            self.msg = msg
            self.bin_msg = np.binary_repr(int(clear_string(self.msg), self.base_src))

    @property
    def statement(self):
        return "{} à {} [diviseur : {}]".format(self.msg, "encoder" if self.encoded else "décoder", self.div)

    @property
    def solution(self):
        div = np.binary_repr(int(clear_string(self.div), 2))
        r = self.compute_crc(self.bin_msg, div)[len(self.bin_msg) + 1:]
        v = np.base_repr(int(r, 2), self.base_dst)
        return format_message(v, self.base_dst)

    @staticmethod
    def compute_crc(dividend, divisor):
        def divide(d, r):
            curr = 0
            while not (len(r) - curr < len(d)):
                for i in range(len(d)):
                    r[curr + i] = np.bitwise_xor(r[curr + i], d[i])
                while r[curr] == 0 and curr != len(r):
                    curr = curr + 1
            return r

        divis = [int(i) for i in divisor]
        # computation
        div = [int(i) for i in dividend]
        for i in range(len(divis)):
            div.append(0)
        rem = list(div)
        rem = divide(divis, rem)
        crc = np.bitwise_xor(div, rem)
        return ''.join(str(i) for i in crc)


class Ca2:

    def __init__(self, number=None,
                 base_src=None,
                 base_dst=None):

        self.number = format_value(npr.randint(info8.min, info8.max), base_src) if number is None else number
        self.base_src = choose_bin_base() if base_src is None else base_src
        self.base_dst = choose_bin_base() if base_dst is None else base_dst

    @property
    def statement(self):
        return "Ca2 de {} = ? ({})".format(self.number, self.base_dst)

    @property
    def solution(self):
        n = int(clear_string(self.number), self.base_src)
        return format_value(-n, self.base_dst)
