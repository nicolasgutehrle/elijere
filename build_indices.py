# %%

from src.model import ELIJERE
from src.utils import loadCorpus, prepare_corpus
from collections import Counter
import json 

# %%


config_project = {
    "project_name": "Q5",
    "lg": "en",
    "spacy_model": "en_core_web_lg",
    "anchor_textvalue": ['lemma', 'pos'],
    "removePROPN": True,
    "support": 0,
    "corpus_param" : {
        # size of the train set. 1 means the whole corpus
        "train_size": 1,
        # size of the dev set
        "dev_size": 0,
        # removes Other labels
        "clean": True
        # "maxsize": 200000
    }
}

# %%

# clean = True if you want to removes sentences annotated as Other
clean = config_project['corpus_param']['clean']
corpus = loadCorpus(f"projects/{config_project['project_name']}", clean=clean)
len(corpus), corpus[0]

# %%

config_project['corpus_param']['corpus'] =  corpus
data = prepare_corpus(**config_project['corpus_param'])


try:
    print('Train size :',len(data['X_train']), 'Dev size :', len(data['X_dev']), 'Test size :',len(data['X_test']))
except:
    print('Train size :',len(data['X_train']))

c = Counter([y for y in data['y_train']])
print('Classes distribution : ', dict(c.most_common()))

# %%

elijere = ELIJERE()

elijere_parameters = {
    "data": data,
    "savepath": f"projects/{config_project['project_name']}",
    "anchor_textvalue": config_project['anchor_textvalue'],
    "support": config_project['support'],
    "removePROPN": config_project['removePROPN']
}

elijere.fit(**elijere_parameters)

# %%

with open(f"projects/{config_project['project_name']}/model/elijere_config.json", 'w', encoding='utf-8') as f:
    json.dump(config_project, f, indent=4)