import binascii


class Cipher:

    @staticmethod
    def str2HexStr(string):
        return binascii.b2a_hex((u"%s" % string).encode("utf8")).decode()

    @staticmethod
    def hexStr2Str(string):
        return binascii.a2b_hex(string).decode("utf8")
