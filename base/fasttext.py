
"""
    Note: Before training fasttext, you should go to https://github.com/facebookresearch/fastText
install fastText as instruction in README.md.
"""
import fasttext
import os
dataset = 'movie'
root = "data"
root2 = 'checkpoint'

def data_preprocess(input_file_content,  output_file):
    """
    Only used for convert train and test data to fasttext format.
    :return: no return
    """
    with open(input_file_content, 'r', encoding='utf8') as f1:
        with open(output_file, 'w', encoding='utf8') as f2:
            for line in f1:
                label, sent = line.strip().split("\t")
                head="__label__"
                label=head+label
                f2.write(f"{label.strip()}\t{sent}\n")


if __name__ == "__main__":
    data_preprocess(os.path.join(root, dataset, "train.tsv"), os.path.join(root, dataset, "fasttext_train.train"))
    data_preprocess(os.path.join(root, dataset, "dev.tsv"),  os.path.join(root, dataset, "fasttext_dev.txt"))
    data_preprocess(os.path.join(root, dataset, "test.tsv"),  os.path.join(root, dataset, "fasttext_test.valid"))

    # model_save = os.path.join(root2, dataset, "fasttext_model")

    # train_file = os.path.join(root, dataset, "fasttext_train.txt")
    # test_file = os.path.join(root, dataset, "fasttext_test.txt")
    # classifier = fasttext.train_supervised(input=train_file, dim=100, epoch=20,
    #                                        lr=0.1, wordNgrams=1, bucket=2000000)
    #
    # result = classifier.test(test_file)
    # print(result)