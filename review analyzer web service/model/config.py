import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = "../aspect-extraction-master/data/glove.840B.300d.txt"
    filename_glove_small = "../aspect-extraction-master/data/glove.6B.300d.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.840B.{}d-copy.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    # filename_dev = "data/coNLL/eng/eng.testa.iob"
    # filename_test = "data/coNLL/eng/eng.testb.iob"
    # filename_train = "data/coNLL/eng/eng.train.iob"

    #filename_dev = filename_test = filename_train = "data/test.txt" # test
    filename_train = "../aspect-extraction-master/data/Laptop_Train_v2.iob"
    filename_dev = "../aspect-extraction-master/data/Laptop_Validate_v2.iob"
    filename_test = "../aspect-extraction-master/data/Laptops_Test_Data_phaseB.iob"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words_res_2016.txt"
    filename_tags = "data/tags_res_2016.txt"
    filename_chars = "data/chars_res_2016.txt"
    #
    plt_acc = []
    plt_rec = []
    plt_prec = []
    plt_f1 = []
    # training
    conv2_dim        = 306
    mlp_size         = 300
    train_embeddings = False
    nepochs          = 200
    dropout          = 0.5
    dropout_conv     = 0.6
    batch_size       = 30
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 1
    clip             = 1 # if negative, no clipping
    nepoch_no_imprv  = 25
    WINDOW_LEN = 5
    DIM = 306
    conv2_filter_size = 5
    FILTER_SIZE = [2,3]
    NUMBER_OF_FEATURE_MAPS = [100,50]
    conv = True
    stride = 1
    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU
