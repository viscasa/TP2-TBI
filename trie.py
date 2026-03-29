class TrieNode:
    """ Node dasar penyusun struktur Tree Trie. """
    def __init__(self):
        self.children = {}
        self.term_id = -1
        # Andaikan bukan node akhir, maka term_id = -1


class Trie:
    """ Implementasi sederhana dari Prefix-Tree (Trie) """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, term, term_id):
        """ Mensisipkan kata baru pada susunan node karakter Trie. """
        curr = self.root
        for char in term:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.term_id = term_id

    def search(self, term):
        """ Mencari kecocokan kata persis di dalam node. Mengembalikan ID term. """
        curr = self.root
        for char in term:
            if char not in curr.children:
                return -1
            curr = curr.children[char]
        return curr.term_id

    def get_all_with_prefix(self, prefix):
        """ 
        Mencari seluruh kata (dan ID-nya) yang diawali dengan huruf prefix tertentu.
        Akan berguna pada fasa UI/Auto-complete.
        """
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                # Tidak ada satupun kata yang memiliki prefix ini
                return []
            curr = curr.children[char]
        
        results = []
        self._dfs_collect(curr, prefix, results)
        return results

    def _dfs_collect(self, node, current_string, results):
        """ Helper rekursif pencarian ke dalam cabang Leaf Tree Trie. """
        if node.term_id != -1:
            results.append((current_string, node.term_id))
        
        # Penelusuran pre-order berdasarkan abjad agar hasil urut leksikografis
        for char in sorted(node.children.keys()):
            child_node = node.children[char]
            self._dfs_collect(child_node, current_string + char, results)


class TrieIdMap:
    """
    Penyesuaian antar-muka untuk kelas util.IdMap(). Mengelompokkan utilitas IdMap 
    tapi berlandaskan struktur Trie berdasar arahan PDF dosen.
    """
    def __init__(self):
        self.trie = Trie()
        self.id_to_str = []

    def __len__(self):
        """ Mengembalikan banyaknya term disimpan. """
        return len(self.id_to_str)

    def __get_str(self, i):
        """ Mengembalikan term string berdasarkan termid-nya. """
        return self.id_to_str[i]

    def __get_id(self, s):
        """ 
        Pencarian berbasis Trie dengan fallback pemendahan item jika 
        s belum ada di dictionary.
        """
        result_id = self.trie.search(s)
        if result_id == -1:
            # Term baru
            new_id = len(self.id_to_str)
            self.id_to_str.append(s)
            self.trie.insert(s, new_id)
            return new_id
        return result_id

    def __getitem__(self, key):
        """ Overriding perlakuan standard [] """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError


if __name__ == '__main__':
    print("=== Demo Trie-Based Dictionary ===")
    t_map = TrieIdMap()
    
    # 1. Menambahkan kata-kata ke dalam Trie
    words_to_insert = ["halo", "semua", "selamat", "pagi", "halyard", "halcyon", "hello"]
    print(f"1. Meng-insert kata-kata berikut ke dalam Trie:\n   {words_to_insert}")
    
    for word in words_to_insert:
        _ = t_map[word] # __getitem__ akan otomatis insert ke Trie jika belum ada
        
    print(f"2. Total term unik di dalam dictionary: {len(t_map)}")
    print("\n3. Mendemonstrasikan ID Mapping (seperti dictionary biasa):")
    print(f"   - ID untuk kata 'semua'   : {t_map['semua']}")
    print(f"   - Kata untuk ID {t_map['pagi']}         : {t_map[t_map['pagi']]}")
    
    print("\n=== Demo Auto-Complete (Prefix Search) ===")
    prefixes_to_test = ["hal", "se", "p", "z"]
    
    for prefix in prefixes_to_test:
        results = t_map.trie.get_all_with_prefix(prefix)
        # result adalah list of tuple (word, term_id)
        words_only = [res[0] for res in results]
        print(f"Mencari prefix '{prefix}'{' '*(4-len(prefix))} -> {words_only}")
        
    print("\nTest fungsionalitas TrieIdMap berhasil dilalui dengan baik.")
