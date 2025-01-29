import sys

sys.path.append('..')

import json
from glob import glob
import networkx as nx
from typing import List 
import pandas as pd 
from collections import defaultdict
from networkx.classes.graph import Graph
import matplotlib.pyplot as plt 
from itertools import combinations
from functools import partial
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from networkx.classes.reportviews import NodeView, EdgeView
from spacy.tokens import Doc
import copy
from functools import reduce
import numpy as np 
# from nervaluate import Evaluator


def saveWhatLinksHere(entity_type:str, savepath:str, list_urls:List[str]) -> None:
    """
    Save results of getWhatLinksHere to disk

    :param savepath: Path to save folder
    :type savepath: str
    :param list_urls: List of URLs to save
    :type list_urls: List[str]
    """

    os.makedirs(f"{savepath}/whatlinkshere/", exist_ok=True)
    with open(f"{savepath}/whatlinkshere/{entity_type}-whatlinkshere.json", 'w', encoding='utf-8') as f:
        json.dump(
            {
                "type": entity_type,
                "urls": list_urls
            }, f, indent=4
        )
  
    # with open(f"{savepath}/whatlinkshere/{entity_type}-whatlinkshere.txt", 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(list_urls))
    

def loadWhatLinksHereLinks(entity_type:str, folderpath: str = "") -> List[str]:
    """
    Loads list of URL from text file if exists, otherwise returns empty list

    :param folderpath: Path to folder containing 'whatlinkshere.txt' file, which list of URL, defaults to None
    :type folderpath: str, optional
    :return: List of URL to What Links Here pages
    :rtype: List[str]
    """
    def process(ent:str):

        if not folderpath:
            return {
                "type": entity_type,
                "urls": []
            }
        elif not os.path.exists(f"{folderpath}/whatlinkshere/{ent}-whatlinkshere.json"):
            return {
                "type": entity_type,
                "urls": []
            }
        else:
            with open(f"{folderpath}/whatlinkshere/{ent}-whatlinkshere.json", 'r', encoding='utf-8') as f:
                return json.load(f)
            
    if isinstance(entity_type, dict):
        return [process(x) for x in entity_type.keys()]
    else:
        return process(entity_type)


        # with open(f"{folderpath}/whatlinkshere/{entity_type}-whatlinkshere.txt", 'r', encoding='utf-8') as f:
        #     file = f.read()
        #     return file.split('\n')

def saveWikidataLinks(savepath:str, data:dict) -> None:
    """
    Saves results from getWikidataLinks

    :param savepath: Path to folder to save 'wikidata_entities.txt' file
    :type savepath: str
    :param list_urls: List of URL to save to disk
    :type list_urls: List[str]
    """
    os.makedirs(f"{savepath}/wikidatalinks/", exist_ok=True)

    if isinstance(data, list):
        for x in data:
            with open(f"{savepath}/wikidatalinks/{x['type']}-wikidatalinks.json", 'w', encoding='utf-8') as f:
                json.dump(x, f, indent=4)

    else:
        with open(f"{savepath}/wikidatalinks/{data['type']}-wikidatalinks.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

def loadWikidataLinks(entity_type:str, folderpath: str = "") -> List[str]:
    """
    Load wikidatalinks file

    :param folderpath: Path to folder containing 'wikidatalinks.txt' file, which list of URL, defaults to None
    :type folderpath: str, optional
    :return: List of entity Ids
    :rtype: List[str]
    """
    def process(ent:str):
        if not folderpath:
            return []
        elif not os.path.exists(f"{folderpath}/wikidatalinks"):
            return []
        else:
            with open(f"{folderpath}/wikidatalinks/{ent}-wikidatalinks.json", 'r', encoding='utf-8') as f:
                return json.load(f)
            
    if isinstance(entity_type, dict):
        return [process(x) for x in entity_type.keys()]
    else:
        return [process(entity_type)]
            
def saveEntitiesData(savepath: str, entityData:dict) -> None:
    """
    Save entity data as obtained by the WikidataParser

    :param savepath: Path to project where to save data
    :type savepath: str
    :param entityData: Dictionnary to save
    :type entityData: dict
    """

    os.makedirs(f"{savepath}/entity_data/{entityData['type']}", exist_ok=True)
    with open(f"{savepath}/entity_data/{entityData['type']}/{entityData['id']}.json", 'w', encoding='utf-8') as f:
        json.dump(entityData, f, indent=4)

def loadEntitiesData(savepath: str) -> List[dict]:
    """
    Helper function to load entity data files on disk

    :param savepath: Path to folder containing the files 
    :type savepath: str
    :return: List of entity data dictionaries
    :rtype: List[dict]
    """

    list_entities_data = []

    for doc in glob(f"{savepath}/entity_data/**/**.json", recursive=True):
        with open(doc, encoding='utf-8') as f:
            list_entities_data.append(json.load(f))
    return list_entities_data

def saveDocument(savepath:str, data:dict) -> None:

    os.makedirs(f"{savepath}/corpus", exist_ok=True)

    save_data = copy.deepcopy(data)
    for dict_sent in save_data['content']:

        dict_sent['graph'] = nx.node_link_data(dict_sent['graph'])
        for dict_rel in dict_sent['props']:
            for sdp in dict_rel['sdpgraphs']:
            # try:
                if not isinstance(sdp['sdpgraph'], str):
                    sdp['sdpgraph'] = nx.node_link_data(sdp['sdpgraph'])

    with open(f"{savepath}/corpus/graph_{save_data['id']}.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4) 

def saveCorpus(savepath: str, data:List[dict]) -> None:
    """
    Save corpus as obtained by Processor

    :param savepath: Path to project where to save data
    :type savepath: str
    :param data: Data to save
    :type data: dict
    """

    os.makedirs(f"{savepath}/corpus", exist_ok=True)
    saveFunc = saveDocument
    for d in data:
         saveFunc(savepath=f"{savepath}/corpus", data=d)

def loadDocument(savepath: str, clean:bool=True) -> dict:

    with open(savepath, 'r', encoding='utf-8') as f:
        dict_ent = json.load(f)
            # print(dict_ent)

    for dict_sent in dict_ent['content']:

        dict_sent['graph'] = nx.node_link_graph(dict_sent['graph'], directed=True, multigraph=False)
        # list to pop some graphs if the sdpgraph is SYNTACTIC-ERROR

        for dict_rel in dict_sent['props']:
            list_pop = []

            for i, prop in enumerate(dict_rel['sdpgraphs']):
                # if not isinstance(dict_rel['sdpgraph'], str):
                if not prop['sdpgraph'] == 'SYNTACTIC-ERROR':
                    prop['sdpgraph'] = nx.node_link_graph(prop['sdpgraph'], directed=True, multigraph=False)
                else:
                    list_pop.append(i)
            # print(list_pop)
            if clean:
                dict_rel['sdpgraphs'] = [x for i, x in enumerate(dict_rel['sdpgraphs']) if i not in list_pop]

    dict_ent['content'] = list(filter(lambda x: x['props'], dict_ent['content']))
    return dict_ent
       
def loadCorpus(savepath: str, clean=True) -> List[dict]:
    """
    Helper function to load a corpus as obtained by the Processor class

    :param savepath: Path to project where the corpus is stored
    :type savepath: str
    :param clean: Whether to remove graph with SYNTACTIC-ERRORS, defaults to True
    :type clean: bool, optional
    :return: Loaded corpus
    :rtype: List[dict]
    """
    list_graphs = []
    append_func = list_graphs.append 

    for doc in glob(f"{savepath}/corpus/**.json"):
        # print(doc)

        dict_ent = loadDocument(savepath=doc, clean=clean)
        append_func(dict_ent)

    list_graphs = list(filter(lambda x: x['content'], list_graphs))
    # list_graphs = filter()
    return list_graphs

def save_dataset(savepath:str, dataset:dict) -> None:

    os.makedirs(f"{savepath}/dataset", exist_ok=True)

    save_data = copy.deepcopy(dataset)
    for k, v in save_data.items():
        if k.startswith('X'):
            for prop in v:
                # print(prop)
                prop['sdpgraph'] = nx.node_link_data(prop['sdpgraph'])

    with open(f"{savepath}/dataset/dataset.json", 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4)

def load_dataset(savepath:str) -> dict:

    with open(f"{savepath}/dataset/dataset.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for k, v in dataset.items():
        if k.startswith('X'):
            for prop in v:
                # print(prop)
                prop['sdpgraph'] = nx.node_link_graph(prop['sdpgraph'], directed=True, multigraph=False)


    return dataset

def prepare_corpus(corpus: List[dict], train_size:float=1, dev_size:float=0, clean:bool=True, defaultclass:str='Other', maxsize:int=0, savepath:str = '') -> dict:
    """
    Selects graphs associated with properties from corpus

    :param corpus: Corpus of data
    :type corpus: List[dict]
    :return: List of dictionnaries containing the graph and data
    :rtype: List[dict]
    """
    # maxsize = int(len(list_graphs) / 4)
    # list_graphs = list_graphs[:maxsize]
    # len(list_graphs)
    list_prop = [z for x in corpus for y in x['content'] for z in y['props']]
    list_graphs = []

    for x in corpus:
        for y in x['content']:
            main_graph = y['graph']
            for p in y['props']:
                # print(p)
                for sdp in p['sdpgraphs']:
                    sdp['prop'] = p['prop']
                    sdp['source'] = p['source']
                    sdp['target'] = p['target']
                    sdp['sent'] = p['sent']
                    sdp['sent_graph'] = main_graph
                    try:
                        sdp['source_type'] = p['source_type']
                        sdp['target_type'] = p['target_type']
                    except:
                        pass
                    list_graphs.append(sdp)
    # for x in list_prop:
    #     for sdp in x['sdpgraphs']:
    #         sdp['prop'] = x['prop']
    #         sdp['source'] = x['source']
    #         sdp['target'] = x['target']
    #         sdp['sent'] = x['sent']

    #         try:
    #             sdp['source_type'] = x['source_type']
    #             sdp['target_type'] = x['target_type']
    #         except:
    #             pass
    #         list_graphs.append(sdp)

    if maxsize:
        list_graphs = list_graphs[:maxsize]
    
    if train_size == 1:
        X_train = list_graphs
        y_train = [x['prop'] for x in list_graphs]

        if clean:
            X_train = list(filter(lambda x: x['prop'] != defaultclass, X_train))
            y_train = list(filter(lambda x: x != defaultclass, y_train))

        data = {
            "X_train": X_train,
            "y_train": y_train,
        }

    elif train_size == .5:
        X_dev, X_test, y_dev, y_test = divide_train_dev_text(list_graphs=list_graphs, train_size=.5)

        data = {
            "X_dev": X_dev,
            "y_dev": y_dev,
            "X_test": X_test,
            "y_test": y_test
        }

    else:

        X_train, X_dev, y_train, y_dev = divide_train_dev_text(list_graphs=list_graphs, train_size=train_size)
        X_dev, X_test, y_dev, y_test = divide_train_dev_text(list_graphs=X_dev, train_size=dev_size)

        if clean:
            X_train = list(filter(lambda x: x['prop'] != defaultclass, X_train))
            y_train = list(filter(lambda x: x != defaultclass, y_train))

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_dev": X_dev,
            "y_dev": y_dev,
            "X_test": X_test,
            "y_test": y_test
        }


    if savepath:
        save_dataset(savepath=savepath, dataset=data)


    return data

def getNodeText(graph: Graph, node:int, textvalue:str = '', cleanPropn:bool = True) -> str:
        """
        Returns node as str, as a combination of different textual values
        (text, lemma, pos, dep). Textvalue can be a str or a list of these
        options. If cleanPropn is True, remove the text or lemma of any node
        with the PROPN pos tag

        :param graph: Graph to process 
        :type graph: Graph
        :param node: Node to get the text from 
        :type node: int
        :param textvalue: Textual value to represent the node, defaults to 'text'
        :type textvalue: str, optional
        :param cleanPropn: Whether to remove PROPN tags, defaults to True
        :type cleanPropn: bool, optional
        :raises Exception: Raise exception if 
        :return: Textual value of the node
        :rtype: str
        """
        # TODO : what does the Exception do ? 

        # needs to combine the different text values provided by the user
        if isinstance(textvalue, list):
            if cleanPropn:
                if graph.nodes[node]['pos'] == 'PROPN':

                    token =  '_'.join([graph.nodes[node][x] for x in textvalue if x not in ('text', 'lemma')])
            
                else:                        
                    token =  '_'.join([graph.nodes[node][x] for x in textvalue])

            else:
                token =  '_'.join([graph.nodes[node][x] for x in textvalue])

        else:
            if cleanPropn:
                if textvalue in ('text', 'lemma'):
                    raise Exception("Cannot clean PROPN tag and keep text or lemma. Either set cleanPropn to False, or use pos or dep as textvalue")

                else:
                    token = graph.nodes[node][textvalue]
            else:
                token = graph.nodes[node][textvalue]
        return token


def vectorize_data(data, we):
    vec_data = []
    for doc in data:
        graph = doc['sdpgraph']
        vec_graph = []
        for node in graph:
            word = getNodeText(graph, node, textvalue='text', cleanPropn=False)
            try:
                vec = we.word_vec(word)
                vec_graph.append(vec)
            except:
                pass
        if vec_graph:
            vec_graph = reduce(lambda x, y: x + y, vec_graph)
            # print(vec_graph.shape)

            vec_data.append(vec_graph)
        else:
            vec_data.append(np.zeros(300))
        # break
        # print()
    vec_data = np.asarray(vec_data)
    return vec_data


def load_dictrel(savepath: str) -> dict:
    """
    Simple helper function to load dictionnary containing properties and their names.

    :param savepath: Path to the project where dict_rel is saved
    :type savepath: str
    :return: dict_rel
    :rtype: dict
    """
    
    with open(f'{savepath}/dict_rel.json', encoding='utf-8') as f:
        dict_rel = json.load(f)
    return dict_rel

def divide_train_dev_text(list_graphs:List[dict], train_size:float = .8) -> tuple:
    """
    Helper function to apply train_test_split to list_graphs

    :param list_graphs: List of dict of graphs to process
    :type list_graphs: List[dict]
    :param train_size: Size for train set, defaults to .8
    :type train_size: float, optional
    :return: Output of train_test_split
    :rtype: tuple
    """
    return train_test_split(list_graphs, [x['prop'] for x in list_graphs], train_size=train_size, random_state=42)


def getConfusionMatrix(y_true:List[str], y_pred:List[str], plot:bool=True) -> None:
    """
    Helper function to generate confusion matrix graph from predictions and true labels

    :param y_true: List of true labels
    :type y_true: List[str]
    :param y_pred: List of predictions
    :type y_pred: List[str]
    """
    labels = list(set(y_true))
    conf_m = confusion_matrix(y_true, y_pred, labels=labels, normalize='all')
    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_m, display_labels=labels)
        fig, ax = plt.subplots(figsize=(20,20))
        disp.plot(ax=ax)
    else:
        return pd.DataFrame(conf_m, index=labels, columns=labels)


def vizGraph(graph: Graph, label:str = '') -> None:
    """
    Simple function to vizualise a graph, with either text, lemma or pos as node labels

    :param graph: Graph to visualize
    :type graph: Graph
    :param label: Textual values for nodes, defaults to 'text'
    :type label: str, optional
    """

    plt.figure(figsize=(15, 15))

    pos = nx.spring_layout(graph)

    if label:
        node_labels = nx.get_node_attributes(graph, label) 
        nx.draw(graph, pos, with_labels = True, labels=node_labels)

    else:
        # node_labels = graph.nodes()
        nx.draw(graph, pos, with_labels = True)

    edge_labels = nx.get_edge_attributes(graph,'dep')


    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

def nodeEq(node1: NodeView, node2: NodeView) -> bool:
    """
    Checks equality between two nodes, for isomorphism check

    :param node1: First node to check
    :type node1: NodeView
    :param node2: Second node to check
    :type node2: NodeView
    :return: Whether the two nodes are equal
    :rtype: bool
    """

    if node1['pos'] == node2['pos']:
        if node1['dep'] == node2['dep']:
            return True
        else:
            return False 
    else:
        return False


def edgeEq(edge1: EdgeView, edge2: EdgeView) -> bool:
    """
    Checks edge equality, for isomorphism check

    :param edge1: First edge to check
    :type edge1: EdgeView
    :param edge2: Second edge to check
    :type edge2: EdgeView
    :return: Whether the two edges are equal
    :rtype: bool
    """
    
    # print(edge1 == edge2)
    if edge1 == edge2:
        return True
    return False


def node_subst_cost(node1, node2):
    # check if the nodes are equal, if yes then apply no cost, else apply 1
    if node1['pos'] == node2['pos']:
        # if node1['dep'] == node2['dep']:
        #     return 0
        # return 1
        return 0
    return 1

def node_del_cost(node):
    return 1  # here you apply the cost for node deletion

def node_ins_cost(node):
    return 1  # here you apply the cost for node insertion

# arguments for edges
def edge_subst_cost(edge1, edge2):
    # check if the edges are equal, if yes then apply no cost, else apply 3
    if edge1==edge2:
        return 0
    return 1
    # return 0

def edge_del_cost(node):
    # return 0
    return 1  # here you apply the cost for edge deletion

def edge_ins_cost(node):
    # return 0
    return 1  # here you apply the cost for edge insertion

def balanceRelationDataset(corpus: List[dict], upper_limit:int) -> List[dict]:
    """
    Simple helper function to balance the relation dataset from the bottom. Makes sure there are no more instances than the upper limit for each relation.

    :param corpus: Dataset to process
    :type corpus: List[dict]
    :param upper_limit: Upper maximum of relation in the dataset
    :type upper_limit: int
    :return: Balanced dataset
    :rtype: List[dict]
    """

    tmp_dict = defaultdict(int)

    for doc in corpus:
        for cont in doc['content']:
            count = Counter(prop['prop'] for prop in cont['props'])
            for k,v in count.most_common():
                if tmp_dict[k] < upper_limit: 
                    tmp_dict[k]+=v
                else:
                    cont['props'] = [x for x in cont['props'] if x['prop'] != k]

    for c in corpus:
        c['content'] = [x for x in c['content'] if x['props']]
    corpus = [x for x in corpus if x['content']]
    return corpus 


def getRelationNames(dict_rel:dict) -> dict:
    """
    Helper function to get mapping between properties and their labels

    :param dict_rel: original dict_rel 
    :type dict_rel: dict
    :return: dictionnary containing the mapping
    :rtype: dict
    """
    tmp_dict = dict_rel[0]['props']
    func_update = tmp_dict.update
    for d in dict_rel[1:]:
        func_update(d['props'])
    tmp_dict['Other'] = 'Other'
    return tmp_dict

def doc2graph(doc: Doc) -> dict:
    """
    Creates directed graph from Doc, where Token objects are
    node, and dependency labels are added to edges.

    :param doc: spaCy document to transform into graph
    :type doc: Doc
    :return: Dictionary containing the doc as graph and its metadata
    :rtype: dict
    """
    # dictionary to store the graph and its metadata
    dict_graph = {}

    graph = nx.DiGraph()
    attrs = {}

    for token in doc:

        # add edge between token and its head
        if token != token.head:

            graph.add_edge(token.head.i, token.i, dep = token.dep_)
        
        # add ROOT token to the head key in the dictionary
        if token.dep_ == 'ROOT':
            dict_graph['head'] = token.i

        # token metadata
        attrs[token.i] = {
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'dep': token.dep_,
            'char_idx': token.idx
            # 'morph': str(token.morph)
        }

    nx.set_node_attributes(graph, attrs)

    dict_graph['graph'] = graph

    return dict_graph

def showEval(eval_dict:dict, key:str) -> pd.DataFrame:
    eval_res = [v for k, v in eval_dict[key].items()]
    dev = [x['dev'] for x in eval_res]
    test = [x['test'] for x in eval_res]

    df_dev = pd.DataFrame.from_dict(dev)
    df_dev = df_dev.iloc[:,:-1]
    df_dev.index = list(eval_dict[key].keys())
    df_dev.columns = pd.MultiIndex.from_product([["Dev"], df_dev.columns])    

    df_test = pd.DataFrame.from_dict(test)
    df_test = df_test.iloc[:,:-1]
    df_test.index = list(eval_dict[key].keys())
    df_test.columns = pd.MultiIndex.from_product([["Test"], df_test.columns])    

    evaluation = pd.merge(df_dev, df_test, left_index=True, right_index=True)
    evaluation.loc[len(evaluation.index)] = evaluation.mean()


    # evaluation.columns = ['P', 'R', 'F1', 'P', 'R', 'F1']

    # evaluation.columns = pd.MultiIndex.from_product([["Dev", "Test"], evaluation.columns])    
    return evaluation

def getEnt(x, dict_ent=None):
    ner = []
    # for x in data:
    if x['prop'] != 'Other':
        sourceNode = x['sourceNode']
        sourceType = x['source_type']

        if dict_ent:
            sourceType = dict_ent[sourceType]

        start = sourceNode[0]
        end = sourceNode[-1] + 1

        ner.append(
            {
                        "label": sourceType,
                        "start": start,
                        "end": end
                    }
        )

        targetNode = x['targetNode']
        targetType = x['target_type']

        if dict_ent:
            targetType = dict_ent[targetType]


        start = targetNode[0]
        end = targetNode[-1] + 1

        ner.append(
            {
                        "label": targetType,
                        "start": start,
                        "end": end
                    }
        )

    return ner 

def evaluate_models(models:dict, evaldata:dict,  ner_tags, list_thresh:List[float], dict_rel=None, dict_ent=None,):

    def evaluation(X_data, y_data):

        true_ner = [getEnt(x, dict_ent) for x in X_data]
        # print(y_data)
        if dict_rel:
            true_rel = [dict_rel[x] for x in y_data]
        else:
            true_rel = [x for x in y_data]
        # print(true_rel)
        # print(len(y_data), len(true_rel))
        # print(thresh)
        rel_predictions = []
        total_ner_predictions = []

        for x in X_data:
            rel_pred = func_classify(x['sdpgraph'], thresh=thresh)
            # print(rel_pred)
            rel_predictions.append(rel_pred)
            ner_predictions = []

            if rel_pred['pred'] != 'Other':
                ner_pred = func_ner(x['sent_graph'], rel_pred['pred'], rel_pred['candidate'])
                for pred in ner_pred:

                    if dict_ent:
                        label = dict_ent[pred['pred']]
                    else:
                        label = pred['pred']

                    ner_predictions.append(
                        {
                            "label": label,
                            "start": int(pred['start']),
                            "end": int(pred['end'])
                        }
                    )

            total_ner_predictions.append(ner_predictions)

        # pred_y = [x['pred'] for x in rel_predictions]
        rel_eval = model.evaluate(true_rel, [x['pred'] for x in rel_predictions])

        ner_evaluator = Evaluator(true_ner, total_ner_predictions, tags=ner_tags)
        ner_eval, ner_type_eval = ner_evaluator.evaluate()

        return rel_predictions, total_ner_predictions, true_rel, true_ner, rel_eval, ner_eval, ner_type_eval

    
    eval_dict = {}

    for modelname, model in models.items():
        func_classify = model.predict
        func_ner = model.NERclassification 

        thresh_dict = {}
        for thresh in list_thresh:

            dev_rel_predictions, dev_total_ner_predictions, dev_true_rel, dev_true_ner, dev_rel_eval, dev_ner_eval, dev_ner_type_eval = evaluation(evaldata['X_dev'], evaldata['y_dev'])

            test_rel_predictions, test_total_ner_predictions, test_true_rel, test_true_ner, test_rel_eval, test_ner_eval, test_ner_type_eval = evaluation(evaldata['X_test'], evaldata['y_test'])

            thresh_dict[thresh] = {
                "devPred": dev_rel_predictions,
                'trueDev': dev_true_rel,
                "dev": dev_rel_eval,
                'devNer': dev_ner_eval,
                'devNerType': dev_ner_type_eval,
                'devNerPred': dev_total_ner_predictions,
                'devNerTrue': dev_true_ner,

                "testPred": test_rel_predictions,
                'trueTest': test_true_rel,
                "test": test_rel_eval,
                'testNer': test_ner_eval,
                'testNerType': test_ner_type_eval,
                'testNerPred': test_total_ner_predictions,
                'testNerTrue': test_true_ner,
            }
        eval_dict[modelname] = thresh_dict
    return eval_dict
    # return {"dev": df_test, "test": df_test}

def nereval2df(data_eval, name):
    index_list = ['ent_type', 'partial', 'strict', 'exact']
    df = pd.DataFrame.from_dict([{
        'P': data_eval[i]['precision'], 
        'R': data_eval[i]['recall'],
        'F1': data_eval[i]['f1']}
        for i in index_list])
    
    df.index = pd.MultiIndex.from_tuples([(name, x) for x in index_list], names=['model', 'eval']) 
    return df

def nerevaltype2df(data_eval, name):
    df = pd.DataFrame.from_dict(data_eval)
    index_list = ['ent_type', 'partial', 'strict', 'exact']

    for k in data_eval.keys():
        for i in index_list:
            df.loc[i][k] = data_eval[k][i]['f1']
    df.index = pd.MultiIndex.from_tuples([(name, x) for x in index_list], names=['model', 'eval']) 
    df = df.round(decimals=3)
    return df
# def getAllSubgraphs(list_graphs: List[dict]):
#     """
#     Simple helper function to retrieve SDPs graphs for each 
#     document graph
#     """
#     all_subgraphs = [x for y in list_graphs for x in y['sdps']]
#     for i, x in enumerate(all_subgraphs):
#         x['i'] = i 
    
#     return all_subgraphs


# def loadGraphIndex(path: str) -> dict:
#     """
#     """
#     with open(path, encoding='utf-8') as f:
#         tmp_dict = json.load(f)

#     graph_index = defaultdict(list)
#     for k, v in tmp_dict.items():
#         # print(k)
#         if k.startswith('group'):
#             # print(k, type(k))
#             # break
#             graph_index[k] = nx.node_link_graph(v, directed=True, multigraph=False)
    
#         else:
#         # if isinstance(k, int):
#             for x in v:
#                 graph_index[int(k)].append(
#                     {
#                         "group": x['group'],
#                         "graph": nx.node_link_graph(x['graph'], directed=True, multigraph=False)
#                     }

#                 )

#     return graph_index

# def loadVectors(path:str) -> pd.DataFrame:
#     """
#     """
#     lexicon_tfidf = pd.read_csv(f'{path}/lexicon_tfidf.csv', index_col=0, engine='c')
#     pattern_vectors = pd.read_csv(f'{path}/pattern_vectors.csv', index_col=0, engine='c')

#     return lexicon_tfidf, pattern_vectors


# def areGraphIdentical(graph1: Graph, graph2: Graph, term_textvalue: str = 'text') -> bool:
#     """
#     Checkss if two graphs selected by the key argument are identical in terms of nodes and 
#     edges. Discard the source and target nodes to evaluate equality.
#     Can also select lemma instead of original text to compare
#     """

#     if nx.is_isomorphic(graph1, graph2, edge_match=lambda x, y: x == y):
#         set1 = [token[1][term_textvalue] for token in graph1.nodes(data=True)]
#         set2 = [token[1][term_textvalue] for token in graph2.nodes(data=True)]

#         if set1 == set2:
#             return True

#         return False
#     return False


    # sum of the terms 
    # return terms.sum(axis=1)

# def loadListGraphsTrainTest(savepath: str):
#     """
#     Simple helper function to read savepath folder and load
#     documents
#     """
#     list_graphs = []
#     append_func = list_graphs.append
#     for doc in glob(f"{savepath}/**.json"):
#         with open(doc, encoding='utf-8') as f:
#             dict_graph = json.load(f)
#             dict_graph['graph'] = nx.node_link_graph(dict_graph['graph'], directed=True, multigraph=False)
#             try:
#                 if not isinstance(dict_graph['sdpgraph'], str):
#                     dict_graph['sdpgraph'] = nx.node_link_graph(dict_graph['sdpgraph'], directed=True, multigraph=False)
#             except:
#                 pass

#             append_func(dict_graph)

#     # if exclude_list:
#     #     list_graphs = [x for x in list_graphs if x['propName'] not in exclude_list]


#     return list_graphs

# def getTrueGraph(dict_graph, graphkey='sdpgraph', anchor_textvalue=['text', 'pos']):
#     """
#     Helper function to get the Grounf Truth graph, and its anchor
#     By default will take the graph at 'sdpgraph' key, otherwise will
#     take the 'graph' key
#     """
#     try:
#         trueGraph = dict_graph[graphkey]
#         dict_graph = analyseGraph(dict_graph, graphkey=graphkey, anchor_textvalue=anchor_textvalue)

#     except:
#         # for graphs labelled as Other
#         trueGraph = dict_graph['graph']
#         dict_graph = analyseGraph(dict_graph, graphkey='graph', anchor_textvalue=anchor_textvalue)

#     return trueGraph

# def getDepStr(graph, dep, node1, node2): 
#     """
#     """
#     if nx.has_path(graph, node2, node1):
#         return f"<-{dep['dep']}"
#     else:
#         return f"{dep['dep']}->"

# def getGraphPaths(graph, anchor=None):
#     """
#     Get all longest possible paths in graph
#     """
#     if anchor:
#         all_paths = nx.single_source_shortest_path(graph, anchor)
#         all_paths = list(all_paths.values())


#     else:
#         all_paths = []
#         # gets all possible patgs between pairs of node
#         for node1, node2 in combinations(graph.nodes(), 2):

#             paths1 = nx.all_simple_paths(graph, node1, node2)
#             paths1 = list(paths1)

#             if paths1:
#                 paths1.sort()
#                 all_paths.extend(paths1)
            
#             paths2 = nx.all_simple_paths(graph, node2, node1)
#             paths2 = list(paths2)

#             if paths2:
#                 paths2.sort()
#                 all_paths.extend(paths2)
    
#     # filters so as to only keep the longest possible paths
#     list_subsets = [path1 for path1, path2 in combinations(all_paths, 2) if set(path1).issubset(set(path2))]
#     list_subsets = list_subsets + [path1 for path1, path2 in combinations(all_paths[::-1], 2) if set(path1).issubset(set(path2))]
#     filtered_paths = [x for x in all_paths if x not in list_subsets]

#     filtered_paths = [list(graph.subgraph(x).nodes(data=True)) for x in filtered_paths]
    
#     return filtered_paths

