# -----------------------------------------------------------
# Pytorch version for data creation:
# Reader for comments location data, also produces batches for NN training
#
# ------------------------------------------------------------
from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)

import numpy as np

from collections import namedtuple
import gensim
import collections

from torch.utils.data import DataLoader, Dataset

EMPTY_LINE = "EMPTY"
COMMENT_ONLY_LINE = "COMMENTEMPTY"

UNKNOWN_WORD = "-unk-"
special_symbols = [UNKNOWN_WORD]
UNKNOWN = 0
AVG_WEMBED = 1
sembed_options = { "avg_wembed": AVG_WEMBED}

CODE_EMBEDDINGS_DATA = "~/Desktop/data/code-big-vectors-negative300.bin" #change to embedding directory
PRE_EMBED_DIM = 300
MIN_WORD_FREQ = 0


def read_w2v_models():
    """
    Reads pretrained w2v embeddings.
    There are two types of embeddings -- trained on code or on English comments. The
    function reads both.

    Returns
    -------
    w2v_models : two gensim based w2v models, first code, the other comment
    w2v_dims : dimensions of the two models above.

    """

    print("Loading pre-trained embeddings\n")
    print("Reading word embeddings")
    print("\t sources: " + CODE_EMBEDDINGS_DATA)
    code_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(CODE_EMBEDDINGS_DATA, binary=True)
    code_w2v_dim = code_w2v_model.vector_size
    print("..done.")
    w2v_models = [code_w2v_model]
    w2v_dims = [code_w2v_dim]
    return w2v_models, w2v_dims


def read_comment_location_file(c_file, max_stmt_len, skip_empty_stmt):
    """
    This method reads a comment file. It creates the loc sequence for each file. The return value is a list of file sequences.

    Parameters
    ----------
    c_file : string
         a file containing comment location information. The format is
         fileid\tloc\tlabel\tgramarlabel\tstmt\tmodulebackground\tlibrarybackground
         The label indicates whether a comment appears before this statement. The file
         may also contain background knowledge. Side features are present in separate files.

    skip_empty_stmt : boolean
         ignore lines where the stmt text matches exactly the empty line marker

    Return
    ------
    file_seqs: list of file_seq.
         Each file_seq is a list of line_of_codes

    all_words: counter
         code and background tokens to their counts across the dataset

    """

    print("Reading comment locations...")

    curfile = ""
    cur_file_conts = file_sequence()

    all_words = collections.Counter()  # vocab of code
    file_seqs = []

    f_ch = open(c_file, "r")
    for line in f_ch:
        fileid, loc, blk_bd, label, _, _, code, _, _ = line.strip().split("\t")
        # blk_bds for hierarchical LSTM model   # 1 ==begin, 2==  mid  3 == end of block.  there is no system of blocks,  bd value  always -1

        # skip lines which are blank because they had a comment
        if code == COMMENT_ONLY_LINE:
            continue

        # skip all empty lines if the marker is set
        if code == EMPTY_LINE and skip_empty_stmt:
            continue

        # clean and truncate the LOC
        code_toks = code.split()[0: min(len(code.split()), max_stmt_len)]
        for w in code_toks:
            all_words[w] += 1
        code = " ".join(code_toks).strip()

        cloc = line_of_code(fileid, loc, blk_bd, code, label)

        # store the current files contents before moving to next file
        if fileid != curfile:
            if cur_file_conts.num_locs() > 0:
                file_seqs.append(cur_file_conts)
                cur_file_conts = file_sequence()
            curfile = fileid

        cur_file_conts.add_loc(cloc)

    if cur_file_conts.num_locs() > 0:
        file_seqs.append(cur_file_conts)
    f_ch.close()
    print("\tThere were " + str(len(file_seqs)) + " file sequences")
    return file_seqs[0:200], all_words   #just to be able to run on local: get first 200 files


def get_restricted_vocabulary(word_counts, vocab_size, min_freq, add_special):
    """ Create vocabulary of a certain max size

    Parameters
    ----------
    word_counts: Counter
    vocab_size: int
    add_special: boolean
         if true add special tokens into the vocab
    min_freq: int
         must be at least this frequent

    Return
    ------
    words: list of vocab words
    """

    non_special_size = min(vocab_size, len(word_counts))
    if add_special and non_special_size == vocab_size:
        non_special_size -= len(special_symbols)
    word_counts_sorted = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    words = [k for (k, v) in word_counts_sorted if k not in special_symbols and v >= min_freq][0:non_special_size]

    if add_special:
        for s in special_symbols:
            words.insert(0, s)
    return words


def assign_vocab_ids(word_list):
    """
    Given a word list, assign ids to it and return the id dictionaries
    """
    word_ids = [x for x in range(len(word_list))]
    word_to_id = dict(zip(word_list, word_ids))
    id_to_word = dict(zip(word_ids, word_list))
    return word_to_id, id_to_word


def code_each_word(wlist, vocab_to_id, ignore_unknown, ignore_special):
    """
    :return list of wordids
    """
    ret = []
    for w in wlist:
        if ignore_special and w in special_symbols:
            continue
        if w in vocab_to_id:
            ret.append(vocab_to_id[w])
        elif not ignore_unknown:
            ret.append(vocab_to_id[UNKNOWN_WORD])
    return ret


def code_text(text, vocab_to_id, ignore_unknown, ignore_special):
    """
    Convert a text into list of ids

    Parameters
    ----------
    ignore_unknown : boolean
         skip tokens which are not in the vocabulary (dont add UNK)
    ignore_special : boolean
         skip special tokens
    """
    return code_each_word(text.split(), vocab_to_id, ignore_unknown, ignore_special)


def get_avg_embedding(wids, id_to_vocab, w2v_model, w2v_dim):
    """
    Get average embedding. If w2v_model is None, then the embedding is
    all zeroes. Otherwise, average for the words that are in the model vocab.
    """

    sum_embed = np.zeros(w2v_dim, dtype=np.float32)
    if w2v_model == None:
        return sum_embed
    found_wc = 0
    for wid in wids:
        tk = id_to_vocab[wid]
        if tk != UNKNOWN_WORD and tk in w2v_model.vocab:
            sum_embed += w2v_model.wv[tk]
            found_wc += 1
    if found_wc > 0:
        return sum_embed / found_wc
    return sum_embed


def get_embedding_vectors(w2v_model, w2v_dim, vocab_to_id):
    """
    Get all embeddings in a [nwords * dimsize] matrix. If word not
    found, have random embedding.

    """

    emb = np.random.uniform(-0.001, 0.001, size=(len(vocab_to_id), w2v_dim))
    for w, i in vocab_to_id.items():
        if w in w2v_model.vocab:
            emb[i] = w2v_model.wv[w]

    # print(str(emb[0][0:25]))
    # print(str(emb[1][0:25]))
    # print(str(emb[2][0:25]))
    return emb



class line_of_code(object):
    """
    All information pertaining to a loc of code. The loc may be a
    loc, or a syntactically defined code loc
    """

    def __init__(self, fileid, locid, blk_bd, code, label):
        """
        Parameters
        ----------
        fileid : int
        locid : int
        blk_bd: int loc boundary (-1 for NA, 1 for start, 2 mid and 3 end of a block
        code : string rep of source code
        label : int 0/1, whether a label appears before this code loc
        """
        self.fileid = int(fileid)
        self.locid = int(locid)
        self.blk_bd = int(blk_bd)
        self.code = code
        self.label = int(label)

    def __str__(self):
        return str(self.fileid) + "\t" + str(self.locid) + "\t" + str(self.blk_bd) + "\t" + str(
            self.label) + "\t" + str(self.code)


class line_of_code_features(object):
    """
    Features and embeddings for a loc of code
    """

    def __init__(self, coded, lc_embedded):
        self.coded = coded
        self.embedded = lc_embedded


class file_sequence(object):
    """
    All the information attached to one file of source code. A source
    file has a list of line_of_codes
    """

    def __init__(self):
        self.clocs = []

    def add_loc(self, loc):
        self.clocs.append(loc)

    def num_locs(self):
        return len(self.clocs)

    def get_loc(self, bid):
        return self.clocs[bid]

    def get_all_locs(self):
        return self.clocs

    def __str__(self):
        return str([cb.__str__() for cb in self.clocs])

class block_dataset(Dataset):
    def __init__(self,cfile,   max_len_stmt,  skip_empty_line,   word_vsize):
        file_seqs, all_words = read_comment_location_file(cfile, max_len_stmt, skip_empty_line)
        w2v_models, w2v_dims = read_w2v_models()
        max_blks, max_len_blk = 80, 50
        (wordv, all_wordv, all_word_w2i, all_word_i2w) = create_vocabularies(all_words, word_vsize)
        cloc_id_to_features = {}
        ignore_unknown, ignore_special = True, True

        for file_num in range(len(file_seqs)):
            file_seq = file_seqs[file_num]

            for lc_no in range(file_seq.num_locs()):
                lc = file_seq.get_loc(lc_no)
                lc_code = lc.code

                lc_coded = code_text(lc_code, all_word_w2i, ignore_unknown, ignore_special)
                lc_embedded = get_avg_embedding(lc_coded, all_word_i2w, w2v_models[0], w2v_dims[0])

                lc_features = line_of_code_features(lc_coded, lc_embedded)
                cloc_id_to_features[str(lc.fileid) + "#" + str(lc.locid)] = lc_features

        self.file_seqs = file_seqs
        self.cloc_id_to_features = cloc_id_to_features
        self.wordv = wordv

        self.all_wordv = all_wordv
        self.all_word_w2i = all_word_w2i
        self.all_word_i2w = all_word_i2w
        self.all_word_vocab_size = len(self.all_word_w2i)

        self.max_len_lc = max_len_stmt
        self.lc_embed_size = w2v_dims[0]
        self.w2v_models = w2v_models
        self.w2v_dims = w2v_dims

        data_by_seq_len = []
        cur_seq = []

        count_true_num_blks = 0
        Lc_and_Features = namedtuple('Lc_and_Features', ['lc', 'lc_feat'])

        for file_seq in self.file_seqs:
            if len(data_by_seq_len) == 0 or data_by_seq_len[-1] != []:
                data_by_seq_len.append([])

            for lcno in range(file_seq.num_locs()):
                lc = file_seq.get_loc(lcno)
                lc_features = self.cloc_id_to_features[str(lc.fileid) + "#" + str(lc.locid)]
                lc_bd = lc.blk_bd
                if lc_bd == 1:
                    if len(cur_seq) > 0:
                        data_by_seq_len[-1].append(cur_seq)
                        cur_seq = []
                    if len(data_by_seq_len[-1]) >= max_blks:  # if exceed max_blks, create new file for the rest of the file.
                        data_by_seq_len.append([])
                    count_true_num_blks += 1
                    cur_seq.append(Lc_and_Features(lc, lc_features))

                elif lc_bd == 2 or lc_bd == 3:
                    if len(cur_seq) < max_len_blk:
                        cur_seq.append(Lc_and_Features(lc, lc_features))

            if len(cur_seq) > 0:
                data_by_seq_len[-1].append(cur_seq)
                cur_seq = []

        total_sequences = len(data_by_seq_len)
        print(total_sequences)

        count_blks = 0
        for i in range(len(data_by_seq_len)):
            for j in range(len(data_by_seq_len[i])):
                count_blks += 1
            # print("num_blocks read = " + str(count_blks))
        assert count_true_num_blks == count_blks, "Number of blks read into data %d does not match true number of blocks %d" % (count_blks, count_true_num_blks)


        self.data = np.zeros((total_sequences, max_blks, max_len_blk, self.max_len_lc),   dtype=np.int32)  # AVRage embeddding
        self.data_wts = np.zeros((total_sequences, max_blks, max_len_blk, self.max_len_lc), dtype=np.float32)

        self.targets = np.zeros((total_sequences, max_blks), dtype=np.int32)
        self.target_weights = np.zeros((total_sequences, max_blks), dtype=np.float32)

        for i in range(total_sequences):
            for k in range(max_blks):
                for j in range(max_len_blk):
                    if i >= total_sequences or k >= len(data_by_seq_len[i]) or j >= len(data_by_seq_len[i][k]):
                        wid_datum = np.array([self.all_word_w2i[UNKNOWN_WORD]])
                    else:
                        wid_datum = np.array(data_by_seq_len[i][k][j].lc_feat.coded)
                        self.targets[i][k] = data_by_seq_len[i][k][0].lc.label
                        self.target_weights[i][k] = 1.0

                    self.data[i][k][j][0:wid_datum.shape[0]] = wid_datum
                    self.data_wts[i][k][j][0:wid_datum.shape[0]] = np.ones(wid_datum.shape[0])
                #    print(self.data[i][k][j])

        self.pretrained_vectors = get_embedding_vectors(self.w2v_models[0], self.w2v_dims[0], self.all_word_w2i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.data_wts[item], self.targets[item], self.target_weights[item], self.pretrained_vectors

def create_vocabularies(all_words, word_vsize):
    """
    Create a vocabulary that combines the loc terms and the background terms
    """

    print("\tCreating vocabularies for code and background..")
    wordv = get_restricted_vocabulary(all_words, word_vsize, MIN_WORD_FREQ, True)

    all_wordv = []
    all_wordv.extend(wordv)
    all_word_w2i, all_word_i2w = assign_vocab_ids(all_wordv)

    return (wordv, all_wordv, all_word_w2i, all_word_i2w)



def main():
    #Example only for training dataset:
    cfile = input("Enter data.txt path for training: ")
    #cfile = "/home/ipek/Desktop/code/cpred/split30_word_data/train/data.txt"

    skip_empty_line = False
    word_vsize = 5000
    max_blks, max_len_blk, max_len_stmt = 80, 50, 30  #I.e. each file can include at most 80 blocks, each block can have at most 50 line of code and each line of code has at most 30 words
    batch_sizes = [8]

    for batch_size in batch_sizes:
        dataset = block_dataset(cfile,max_len_stmt, skip_empty_line,word_vsize)
        loader = DataLoader(dataset=dataset, batch_size= int(batch_size),shuffle=True)
        print(loader)
        for idx, (da,da1,da2,da3,_) in enumerate(loader):
            print(da.shape)
            print(str(da))

            print(da1.shape)
            print(str(da1))

            print(da2.shape)
            print(str(da2))

            print(da3.shape)
            print(str(da3))
            break

if __name__ == "__main__":
    main()
