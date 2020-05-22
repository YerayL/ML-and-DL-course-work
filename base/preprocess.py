import sys
import os
import re

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


re_punct = r"[\[\]\{\}\:\;\'\"\,\<\.\>\/\?\\\|\`\!\@\#\$\%\^\&\*\(\)\-\_\=\+]"
root = "data"

def prep_sst_2(data="../data/SST-2"):
    eng = English()
    filename = ["train.tsv", "dev.tsv", "test.tsv"]

    for name in filename:
        target_file = open(os.path.join(root, "SST-2", name), "w+")
        with open(os.path.join(data, name)) as f:
            l = f.readline()
            print(l)
            for line in f:
                if not "test" in name:
                    sent, label = line.strip().split("\t")
                else:
                    idx, sent = line.strip().split("\t")
                    label = "0"
                doc = eng(sent)
                token_filtered = [tok for tok in doc if not re.match(re_punct, tok.text)]
                #if not "test" in name:
                #    token_filtered_nonstop = [tok for tok in token_filtered if not tok.is_stop]
                #    if token_filtered_nonstop:
                #        token_filtered = token_filtered_nonstop
                token_filtered = [tok.text.lower() for tok in token_filtered]
                if not token_filtered:
                    print(line)
                    continue
                target_file.write(f"{label.strip()}\t{' '.join(token_filtered)}\n")
        target_file.flush()
        target_file.close()


def prep_sst_5(data="../data/SST-5"):
    eng = English()
    filename = ["sst_train.txt", "sst_dev.txt", "sst_test.txt"]
    outname = ["train.tsv", "dev.tsv", "test.tsv"]

    for idx, name in enumerate(filename):
        target_file = open(os.path.join(root, "SST-5", outname[idx]), "w+")
        with open(os.path.join(data, name)) as f:
            # l = f.readline()
            # print(l)
            for line in f:
                label, sent = line.strip().split("\t")
                label = re.search("\d+$", label).group()
                label = int(label) - 1
                doc = eng(sent)
                token_filtered = [tok for tok in doc if not re.match(re_punct, tok.text)]
                if not "test" in name:
                    token_filtered_nonstop = [tok for tok in token_filtered if not tok.is_stop]
                    if token_filtered_nonstop:
                        token_filtered = token_filtered_nonstop
                token_filtered = [tok.text for tok in token_filtered]
                if not token_filtered:
                    print(line)
                    continue
                target_file.write(f"{label}\t{' '.join(token_filtered)}\n")
        target_file.flush()
        target_file.close()


prep_sst_2()
prep_sst_5()
