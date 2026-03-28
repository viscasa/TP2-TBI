import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Bit-level compression menggunakan Elias-Gamma Coding.

    Elias-Gamma encoding untuk bilangan positif N:
      1. Hitung floor(log2(N)) = L
      2. Tulis L buah '0' (unary prefix)
      3. Tulis representasi binary dari N dalam (L+1) bit

    Contoh:
      1  → '1'          (L=0: 0 nol + '1')
      2  → '010'        (L=1: 1 nol + '10')
      3  → '011'        (L=1: 1 nol + '11')
      5  → '00101'      (L=2: 2 nol + '101')
      13 → '0001101'    (L=3: 3 nol + '1101')

    Sama seperti VBEPostings, postings_list diconvert ke gap-based list dulu.
    TF list di-encode langsung (tanpa gap).

    CATATAN: Elias-Gamma hanya bisa encode bilangan >= 1.
    Untuk menangani angka 0 (yang mungkin muncul), kita tambahkan 1 saat encode
    dan kurangi 1 saat decode.
    """

    @staticmethod
    def _elias_gamma_encode_number(n):
        """
        Encode satu bilangan non-negatif n menjadi string of bits.
        Kita tambah 1 ke n agar bisa encode angka 0.
        """
        n = n + 1  # shift supaya n >= 1
        if n == 1:
            return '1'
        L = n.bit_length() - 1  # floor(log2(n))
        # L buah '0', lalu binary representation dari n dalam (L+1) bit
        return '0' * L + bin(n)[2:]

    @staticmethod
    def _elias_gamma_encode_list(numbers):
        """
        Encode list of non-negative integers menjadi bytes menggunakan
        Elias-Gamma coding.
        """
        # Bangun bitstring
        bits = []
        for n in numbers:
            bits.append(EliasGammaPostings._elias_gamma_encode_number(n))
        bitstring = ''.join(bits)

        # Pad bitstring ke kelipatan 8 dan simpan panjang padding
        pad_length = (8 - len(bitstring) % 8) % 8
        bitstring += '0' * pad_length

        # Convert bitstring ke bytes
        byte_list = []
        for i in range(0, len(bitstring), 8):
            byte_list.append(int(bitstring[i:i+8], 2))

        # Prepend pad_length sebagai byte pertama
        result = bytes([pad_length]) + bytes(byte_list)
        return result

    @staticmethod
    def _elias_gamma_decode_list(encoded_bytes):
        """
        Decode bytes yang di-encode dengan Elias-Gamma kembali menjadi
        list of non-negative integers.
        """
        if len(encoded_bytes) == 0:
            return []

        pad_length = encoded_bytes[0]
        data_bytes = encoded_bytes[1:]

        # Convert bytes ke bitstring
        bitstring = ''
        for b in data_bytes:
            bitstring += format(b, '08b')

        # Hapus padding di akhir
        if pad_length > 0:
            bitstring = bitstring[:-pad_length]

        # Decode bitstring
        numbers = []
        i = 0
        while i < len(bitstring):
            # Hitung jumlah leading zeros
            L = 0
            while i < len(bitstring) and bitstring[i] == '0':
                L += 1
                i += 1

            # Baca (L+1) bit untuk bilangan binary
            if i + L + 1 > len(bitstring):
                break  # data habis (seharusnya tidak terjadi pada data valid)

            binary_str = bitstring[i:i + L + 1]
            n = int(binary_str, 2)
            numbers.append(n - 1)  # un-shift (karena encode menambah 1)
            i += L + 1

        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes dengan Elias-Gamma.
        Diubah dulu ke gap-based list.
        """
        # Convert ke gap-based list
        gap_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_list.append(postings_list[i] - postings_list[i - 1])
        return EliasGammaPostings._elias_gamma_encode_list(gap_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode Elias-Gamma encoded postings list.
        Hasil decode masih gap-based, jadi harus di-reconstruct.
        """
        gap_list = EliasGammaPostings._elias_gamma_decode_list(encoded_postings_list)
        # Reconstruct dari gap ke absolute
        result = [gap_list[0]]
        for i in range(1, len(gap_list)):
            result.append(result[-1] + gap_list[i])
        return result

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode term frequency list dengan Elias-Gamma (tanpa gap).
        """
        return EliasGammaPostings._elias_gamma_encode_list(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode Elias-Gamma encoded term frequency list.
        """
        return EliasGammaPostings._elias_gamma_decode_list(encoded_tf_list)


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list    : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, \
            f"hasil decoding POSTINGS tidak sama! got {decoded_posting_list}"
        assert decoded_tf_list == tf_list, \
            f"hasil decoding TF tidak sama! got {decoded_tf_list}"
        print("✓ Encode/Decode PASSED")
        print()

    # Perbandingan ukuran
    print("=" * 50)
    print("PERBANDINGAN UKURAN KOMPRESI")
    print("=" * 50)
    print(f"{'Method':<25} {'Postings (bytes)':<20} {'TF (bytes)':<15}")
    print("-" * 50)
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        enc_p = Postings.encode(postings_list)
        enc_t = Postings.encode_tf(tf_list)
        print(f"{Postings.__name__:<25} {len(enc_p):<20} {len(enc_t):<15}")
