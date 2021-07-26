from model.data_utils import CoNLLDataset
from model.aspect_model import ASPECTModel
from model.config import Config
import matplotlib.pyplot as plt

def main():
    # create instance of config
    config = Config()

    # build model
    model = ASPECTModel(config)
    model.build()
    # model.restore_session("results/test/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)
    epocs = list(range(1, len(config.plt_acc)+1))
    plt.plot(epocs, config.plt_acc, 'b')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()
    plt.plot(epocs, config.plt_rec, 'b')
    plt.xlabel("epochs")
    plt.ylabel("recall")
    plt.show()
    plt.plot(epocs, config.plt_prec, 'b')
    plt.xlabel("epochs")
    plt.ylabel("precision")
    plt.show()
    plt.plot(epocs, config.plt_f1, 'b')
    plt.xlabel("epochs")
    plt.ylabel("F1-measure")
    plt.show()

if __name__ == "__main__":
    main()
