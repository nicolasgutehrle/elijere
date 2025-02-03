# %%
from src.dares import DARES
from collections import Counter
import json 

# %%

dares_config = {
    "project_name": "Q5",
    "lg": "en",
    "spacy_model": "en_core_web_lg",
    "n_core": 6,
    "dares_parameters": {
        "item_limit": 1,
        "item_save_step": 20,
        "items_per_pages": 10,
        "source_doc": 'wikipedia',
        "score_cutoff": 95,
        "getOther": True,
        "maxsizesent": True,
        "removeNoMatch": True
    },
    "entities":[
        {
            # specifiy by what Item type / Property must search in the WhatLinksHere pages (e.g. Q5 ('human'))
            "type": 'Q5',
            # you can provide a label for the Item / Property (e.g. 'human')
            "label": "human",
            # indicate the set of relations you want to collect from Wikidata / Wikipedia
            "props":{
                # PXX are the identifier of a Property on Wikidata
                # you define the label (e.g. placeOfBirth)
                "P19": {
                    "label": 'placeOfBirth',
                    "source": "Person",
                    "target": "Location"
                },
                "P119": {
                    "label": 'placeOfBurial',
                    "source": "Person",
                    "target": "Location"
                },
                "P569": {
                    "label": 'dateOfBirth',
                    "source": "Person",
                    "target": "Time"
                },
                "P570": {
                    "label": 'dateOfDeath',
                    "source": "Person",
                    "target": "Time"
                },
                "P509": {
                    "label": 'causeOfDeath',
                    "source": "Person",
                    "target": "Misc"
                },
                "P26": {
                    "label": 'spouse',
                    "source": "Person",
                    "target": "Person"
                },
                "P106": {
                    "label": 'occupation',
                    "source": "Person",
                    "target": "Misc"
                },
                "P69": {
                    "label": 'educatedAt',
                    "source": "Person",
                    "target": "Location"
                }
            }      
        }]
}


# %%

wp = DARES(**dares_config)

# collects url links to wikidata items
wp.collect_Wikidata_links()

# collects data about entities
wp.processListEntities()

# extracting shortest dependency path
corpus = wp.extract_sdp()

with open(f'projects/{dares_config['project_name']}/dares_config.json', 'w', encoding='utf-8') as f:
    json.dump(dares_config, f, indent=4)