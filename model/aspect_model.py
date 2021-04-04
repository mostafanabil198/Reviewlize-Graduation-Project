import numpy as np
import os
import tensorflow as tf



from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from sklearn.metrics import precision_recall_fscore_support


class ASPECTModel(BaseModel):
    """Specialized class of Model for Aspect"""

    def __init__(self, config):
        
        super(ASPECTModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.max_length = 0


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.dropout_conv = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout_conv")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None, dropout_conv=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        
        word_ids, sequence_lengths,max_length = pad_sequences(words, 0)
        self.max_length = max_length

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }


        if labels is not None:
            labels, _,_ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        if dropout_conv is not None:
            feed[self.dropout_conv] = dropout_conv

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            
            _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        print("print word embedding in add_word_embeddings_op()")
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

        print(tf.Print(self.word_embeddings,[self.word_embeddings]))

        if self.config.conv:
            # word_embeddings = (batch_size * sentence_length * 300)
            # ksizes: size of sliding window = (1 * 3 * 300)
            # stride: movement of the window: = (1 * 1 * 300 * 1)
            # temp_size = (batch_size * 65 * (window_size * 300) ) => merge words in window size together
            self.temp = tf.squeeze(tf.extract_image_patches(self.word_embeddings[:,:,:,tf.newaxis], ksizes=[1, self.config.WINDOW_LEN, self.config.DIM, 1], strides=[1, self.config.stride, self.config.DIM, 1], rates=[1, 1, 1, 1], padding='SAME'))
            self.image_patches = tf.reshape(self.temp, (-1, tf.shape(self.word_ids)[1], self.config.WINDOW_LEN, self.config.DIM))
            # convert every word as a single sample and each one consist of the words in the window size affecting it
            self.image_patches_reshaped = tf.reshape(self.image_patches, (-1, self.config.WINDOW_LEN, self.config.DIM))[:,:,:,tf.newaxis]

    

    def add_logits_op_conv(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """

        with tf.name_scope("conv-maxpool"):

                 # ( (BATCH_SIZE*WORDS), WINDOW_LEN, DIM, 1 )
            pooled_out = []
            for i,filter_size in enumerate(self.config.FILTER_SIZE):
                # filter_width * filter_height * in channels * num_of_out_filters
                filter_shape = [filter_size, self.config.DIM, 1, self.config.NUMBER_OF_FEATURE_MAPS[i]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    #W = tf.get_variable(shape = filter_shape, initializer = tf.truncated_normal_initializer(stddev=0.001), name="W"+str(i))
                b = tf.Variable(tf.constant(0.1, shape=[self.config.NUMBER_OF_FEATURE_MAPS[i]]), name="b")

                conv = tf.nn.conv2d(self.image_patches_reshaped,filter=W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                    #print(tf.Print(conv,[conv]))
    # conv = tf.squeeze(conv) # ( (BATCH_SIZE*WORDS), WINDOW_LEN-FILTER_SIZE + 1, NUMBER_OF_FEATURE_MAPS)
                    #conv = tf.nn.bias_add(conv,b)
                    #conv = tf.nn.relu(conv)
                    #conv_non_linear = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # ( (BATCH_SIZE*WORDS), WINDOW_LEN-FILTER_SIZE + 1, 1, NUMBER_OF_FEATURE_MAPS)

                pooled = tf.nn.max_pool(conv,ksize=[1, (self.config.WINDOW_LEN-filter_size+1), 1, 1],strides=[1, 1, 1, 1],padding='VALID',data_format = 'NHWC',name="pool")
                pooled = tf.squeeze(pooled) # ( (BATCH_SIZE*WORDS), NUMBER_OF_FEATURE_MAPS)
                self.output = tf.reshape(pooled, (-1, tf.shape(self.word_ids)[1], self.config.NUMBER_OF_FEATURE_MAPS[i]))
                pooled_out.append(self.output)
        
        # summing filters together
        self.h_pool = tf.concat(pooled_out, 2)

        with tf.name_scope("size_calc"):
            size=0
            for i in range(len(self.config.FILTER_SIZE)):
                size += self.config.NUMBER_OF_FEATURE_MAPS[i]


        with tf.variable_scope("proj"):
            
            dense_input = tf.reshape(self.h_pool, (-1, size))
            #dense_input = tf.nn.dropout(dense_input, self.dropout_conv)
            
            output = tf.contrib.layers.fully_connected(
            dense_input,
            self.config.mlp_size,
            activation_fn=tf.nn.relu,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1227),
            weights_regularizer=tf.contrib.layers.l2_regularizer(0.001),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="input1"
            )

            

            output = tf.nn.dropout(output, self.dropout_conv)

            output = tf.contrib.layers.fully_connected(
            output,
            self.config.ntags,
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1227),
            weights_regularizer=tf.contrib.layers.l2_regularizer(0.001),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="input2"
            )

        self.logits = tf.reshape(output, (-1, tf.shape(self.word_ids)[1], self.config.ntags))

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                tf.int32)
           

    def add_loss_op(self):
        """Defines the loss"""
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # Aspect specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op_conv()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0,dropout_conv=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout, self.config.dropout_conv)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        gold = []
        pred = []
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
               
                
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                label_pred_arr = []
                for i in lab_pred:
                    label_pred_arr.append(i)
                gold+=lab
                pred+=label_pred_arr
                accs    += [a==b for (a, b) in zip(lab, label_pred_arr)]
                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags,message = "gold standard"))
                lab_pred_chunks = set(get_chunks(label_pred_arr,
                                                 self.config.vocab_tags,message = "prediction"))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        score = precision_recall_fscore_support(gold, pred, average='macro')
        #score_bi = precision_recall_fscore_support(gold, pred, labels =[1,2], average='macro')
        return {"acc": 100*acc, "f1": 100*f1, "partial_matching_stat": score}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
