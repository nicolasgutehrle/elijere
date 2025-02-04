# ELIJERE : Extensible, Lightweight and Interpretable Joint Extraction of Relations and Entities

**NOTE : this package is still under development**

ELIJERE is a method for the task of Joint Extraction of Relations and Entities. Unlike most methods for this task, which relies on deep-learning and Transformers architectures, this method relies on two linguistics resources : 
* a Syntactic Index, which describes how a relation is expressed syntactically and what kind of entities are involved
* a Lexical Index, which describes how a relation is expressed lexically 

Both these resources are built from lexico-syntactic patterns. 
For this package, we propose a script to build the DARES dataset, which consists of annotated sentences expressing relations between entities that are collected from Wikipedia and distantly annotated using Wikipedia. Here, the lexico-syntactic patterns correspond to the Shortest Dependency Path between two entities. 
However, these resources could potentially be built from any dataset of annotated sentences expressing a relation between entities (this should be implemented in future versions). 

## Install

To install this package, run the following command:

```
pip install dist/elijere-0.0.1.tar.gz 
```
Note : this might change in future version, if the code is released to PyPI

This tool relies on spaCy to perform the linguistic analysis of the sentences, and for applying the rules. spaCy will be installed when installing the package, but you will also have to download the model(s) you want to use for processing the sentences. 

### Building the DARES dataset

The parameters for building the DARES dataset are stored in a dictionnary at the top of the ```python building_dares_dataset.py```. Please check it and modifiy it accordingly before running the script. Once it is done, you can run the script with the following command :

```
python build_dares_dataset.py
```
This will build a ```projects``` directory, which will contain a folder named as the "project_name" parameter in the dictionnary. This folder is then structured as follows:
* **dares_config.json** : the same dictionnary as at the top of the ```python build_dares_dataset.py``` file, containing the parameters for building the DARES dataset
* **whatlinkshere** : contains a list of URL for downloading Wikidata Items from the What Links Here page
* **wikidatalinks** : list of Items corresponding to a Property or a Type in Wikidata, which will be used to access related Wikipedia pages
* **entity_data** : contains a separate .json file containing data collected from Wikipedia for each Item in the **wikidatalinks** folder .
* **corpus** : contains a separate .json file for each Item in the **entity_data** folder, containing sentences processed by spaCy. These files are used for building the Indices.

### Building the linguistic resources for the ELIJERE method

Similarly, the parameters for building the linguistic resources are stored in a dictionnary at the top of the ```python building_indices.py```. Please check it and modifiy it accordingly before running the script. Once it is done, you can run the script with the following command :

```
python building_indices.py
```

This will build a ```model``` folder structured as follows: 
* **elijere_config.json** : the same dictionnary as at the top of the ```python building_indices.py``` file, containing the parameters for building the indices
* **semanticIndex** (this name will change in future versions): 
    * **index** : the Lexical Index, stored in the .csv format
    * **params** : the parameters for building the Lexical Index

* **syntacticIndex** : 
    * **index** : the Syntactic Index, stored in the .json format
    * **params** : the parameters for building the Syntactic Index


## Running

From there, you can import the extractor and process sentences. An example is shown below: 
```
from elijere.model import ELIJERE

# instantiating the model
elijere = ELIJERE()
# loading the indices from the project (must point to the folder containing the 'model' folder)
elijere.load_model('projects/Q5')

# example sentence to process
text = 'George Washington was born on February 22, 1732'

# extracts relations from single sentence
facts =elijere.extractFacts(text)
print(facts)
```


### Package structure

This package is structured as follows
* **dist** contains the files for installing the package
* **doc** contains the documentation of the package, as generated with Sphinx
* **projects** contains the projects, ie DARES dataset and Indices, built by running the scripts
* **src** contains the package **elijere**, which is structured as follows:
    * **dares**: module for building the DARES dataset
    * **model**: module for building the Indices and implementing the ELIJERE method
    * **processor**: module for processing the sentences of the DARES dataset with spaCy and extract the SDPs
    * **utils**: module containing sets of utility functions

## License and reference

This tool is under XXX Licence. 

If you use this tool, please cite :

```
ENTER CITATION
```