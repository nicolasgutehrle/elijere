
import sys 
sys.path.append('..')

from typing import List, Callable
# from multiprocessing.dummy import Pool

import pandas as pd
from networkx.classes.graph import Graph
from networkx.classes.reportviews import NodeView

from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
from collections import Counter
from networkx.algorithms.isomorphism import DiGraphMatcher

from .utils import nodeEq, edgeEq, getRelationNames, getNodeText, vizGraph, doc2graph, node_subst_cost, node_del_cost, node_ins_cost, edge_subst_cost, edge_del_cost, edge_ins_cost
# getGraphPaths

# from nltk import ngrams

from functools import partial

from itertools import groupby
import os
import json
from scipy import stats

import copy
from glob import glob 

from itertools import combinations, groupby

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import Levenshtein as lvhst

from functools import reduce

import numpy as np


class SyntacticIndex:


    def __init__(self, syntacticIndexPath:str = '') -> None:
        """
        Constructor for the IndexModel class

        :param semanticIndexPath: Path to project storing the semanticIndex, defaults to ''
        :type semanticIndexPath: str, optional
        :param syntacticIndexPath: Path to project storing the syntacticIndex, defaults to ''
        :type syntacticIndexPath: str, optional
        """
        # if not specified, starts with empty index
        if syntacticIndexPath:
            self.syntacticIndex, self.syntacticIndexParams = self.loadSyntacticIndex(savepath=syntacticIndexPath)
        else:
            self.syntacticIndex = {}
            self.syntacticIndexParams = {}


    def __analyseGraph(self, dict_graph: dict, graphkey:str='graph', anchor_textvalue:str='') -> dict:
        """
        Analyse given graph by finding its anchor node, its corresponding textual representation
        Add them to the metadata of the graph

        :param dict_graph: Dictionnary containing data about graph and graph itself
        :type dict_graph: dict
        :param graphkey: Keys containing the graph to analyse, defaults to 'graph'
        :type graphkey: str, optional
        :param anchor_textvalue: Textua value or values for the anchor node of the graph, defaults to 'text'
        :type anchor_textvalue: str, optional
        :return: Updated dict_graph
        :rtype: dict
        """
        try:
            graph = dict_graph[graphkey]
        except:
            graph = dict_graph['graph']

        anchor, anchortext = self.getGraphAnchor(graph, anchor_textvalue=anchor_textvalue)


        dict_graph['anchor'] = anchor
        dict_graph['anchortext'] = anchortext

        return dict_graph
    


    def getGraphAnalysis(self, list_graphs: List[dict], anchor_textvalue:str, graphkey:str='graph', removeNoAnchor:bool = True) -> List[dict]:
        """
        Finds anchor node and related anchor text for each graph in the list
        anchor_textvalue specifies to textvalue to use to textually represent the anchor node
        graphkey specifies which graph (graph or sdpgraph to use) for analysis

        :param list_graphs: List of dictionnaries containing data about graph and graph itself
        :type list_graphs: List[dict]
        :param anchor_textvalue: Textual value of the anchor node
        :type anchor_textvalue: str
        :param graphkey: Key where the graph is stored, defaults to 'graph'
        :type graphkey: str, optional
        :param removeNoAnchor: Whether to remove graph where there is NO-ANCHOR found, defaults to True
        :type removeNoAnchor: bool, optional
        :return: Updated list of dictionnaries
        :rtype: List[dict]
        """

        # if user does not provide specific text value for the anchor node,
        # it takes the default textvalue 
        list_dicts = map(partial(self.__analyseGraph, graphkey=graphkey, anchor_textvalue = anchor_textvalue), list_graphs)

        # sorts alphabetically by the anchor text
        list_dicts = list(list_dicts)
        list_dicts.sort(key=lambda x: x['anchortext'])

        if removeNoAnchor:
            # makes sure to remove anchor which are only PROPN, without text
            list_dicts = list(filter(lambda x: x['anchortext'] != 'NO_ANCHOR', list_dicts))
            list_dicts = list(filter(lambda x: x['anchortext'] != 'pROPN', list_dicts))
        
        return list_dicts

    

    def getGraphAnchor(self, graph: Graph, anchor_textvalue:str='') -> tuple:
        """
        Returns the anchor / predicate of a graph which is the node without any in edges.
        If there is no such node in the graph, returns 'NO_ANCHOR', which most likely indicates
        an error in the original analysis

        :param graph: Graph to analyse
        :type graph: Graph
        :param anchor_textvalue: Textual value or values for the graph anchor node, defaults to ''
        :type anchor_textvalue: str, optional
        :return: Tuple containing the anchor node and its textual value
        :rtype: tuple
        """
 
        # measure the in-degree of each, then sort them in increasing order
        list_degree = list(graph.in_degree(graph.nodes()))
        list_degree.sort(key=lambda x: x[1])
        
        if list_degree:
            # take the node with the smallest in-degree, i.e. 0
            anchor_node = list_degree[0][0]
            # if this node is a PROPN, need to find another anchor,
            # for now, other possible anchor is ROOT
            if graph.nodes[anchor_node]['pos'] == 'PROPN':
                depnode = [x[0] for x in graph.nodes(data=True) if x[1]['dep'] in ('ROOT')]
                if depnode:
                    anchor_node = depnode[0]
                else:
                    anchor_node = 'NO_ANCHOR'
        else:
            anchor_node = 'NO_ANCHOR'

        if anchor_node != 'NO_ANCHOR':
            anchortext = getNodeText(graph, anchor_node, textvalue=anchor_textvalue)
            anchortext = f"{anchortext[0].lower()}{anchortext[1:]}"

        else:
            anchortext = anchor_node
        return anchor_node, anchortext

    def trainSyntacticIndex(self, list_graphs: List[dict], anchor_textvalue: str, graphkey:str='graph', propkey:str='prop', dict_rel:dict = {}, support:int=0, savepath:str='') -> None:
        """
        Learns Syntactic Index from SDP graphs

        :param list_graphs: List of dict containing the SDP graphs to process with their metadata
        :type list_graphs: List[dict]
        :param anchor_textvalue: Text value to use for the anchor node
        :type anchor_textvalue: str
        :param graphkey: Key in the list_graph containing the graph to process, defaults to 'graph'
        :type graphkey: str, optional
        :param propkey: Key in list_graphs containing the possible labels, defaults to 'propName'
        :type propkey: str, optional
        :param savepath: Path where to save syntactic index, defaults to None
        :type savepath: str, optional
        :return: Syntactic index as a dictionnary
        :rtype: dict
        """

        syntactic_index = {}

        # gets graph anchor
        graph_analysis = self.getGraphAnalysis(list_graphs, anchor_textvalue, graphkey, removeNoAnchor=True)

        # groups graph by their anchor
        for anchor, anchor_group in groupby(graph_analysis, lambda x: x['anchortext']):
                
            anchor_group = list(anchor_group)
            anchor_group.sort(key=lambda x: x['sdpgraph'].size())

            list_candidates = []
            candidate_append = list_candidates.append

            while True:
                # keeps running until anchor_group is empty
                if not anchor_group:
                    break
                else:
                    candidate = anchor_group.pop(0)
                    candidate_graph = candidate['sdpgraph']

                    # keeps every graph identical to the candidate graph
                    identicals = filter(lambda x: nx.is_isomorphic(candidate_graph, x['sdpgraph'], node_match=nodeEq, edge_match=edgeEq), anchor_group)
                    identicals = list(identicals)

                    # gets count of prop
                    props = [candidate[propkey]] + list(map(lambda x: x[propkey], identicals))

                    # dict_rel replaces property ID by given label
                    if dict_rel:
                        tmp_dict = getRelationNames(dict_rel=dict_rel)

                        props = [
                            { "name": tmp_dict[k], "support": v}
                            for k, v in dict(Counter(props)).items() if v >= support 
                        ]
                    else:
                    
                        props = [
                            { "name": k, "support": v }
                            for k, v in dict(Counter(props)).items() if v >= support   
                        ]

                    # ambiguous if multiple prop possible
                    ambiguous = 1 if len(props) > 1 else 0 

                    # get source and target entities types and nodes
                    if props:

                        try:                    
                            source_types = [candidate['source_type']] + list(map(lambda x: x['source_type'], identicals))
                        except:
                            source_types = []

                        try:
                            source_nodes = [candidate['sourceNodeRoot']]

                        except:
                            source_nodes = []

                        try: 
                            target_types = [candidate['target_type']] + list(map(lambda x: x['target_type'], identicals))
                        except:
                            target_types = []
                        
                        try:
                            target_nodes = [candidate['targetNodeRoot']]
                        except:
                            target_nodes = []

                        # learns NER rule :keeps track of entity types given relation used for NER in 
                        # case of multiple entity types for a pattern
                        ner_rules = {}
                        for p, s, t in zip(props, source_types, target_types):
                            ner_rules[p['name']] = {
                                'source_type': s,
                                'target_type': t
                            }
                        # add new value to the index
                        if source_nodes != target_nodes:
                            candidate_append(
                                {
                                    'graph': candidate_graph,
                                    'size': candidate_graph.size(),
                                    'props': props,
                                    "ambiguous": ambiguous,
                                    'source_types': list(set(source_types)),
                                    'source_nodes': list(set(source_nodes)),
                                    'target_types': list(set(target_types)),
                                    'target_nodes': list(set(target_nodes)),
                                    "ner_rules": ner_rules
                                }
                            )

                    # remove any graph identical to the candidate
                    anchor_group = list(filter(lambda x: x not in identicals, anchor_group))
            
            # add identifier to graph
            if list_candidates:
                for i, c in enumerate(list_candidates):
                    c['i'] = i + 1

                syntactic_index[anchor] = list_candidates
                
        if savepath:
            self.saveSyntacticIndex(savepath, syntactic_index, anchor_textvalue, graphkey, propkey)

        self.syntacticIndex = syntactic_index
        self.syntacticIndexParams = {
            "anchor_textvalue": anchor_textvalue,
            "graphkey": graphkey,
            "propkey": propkey
        }

    def saveSyntacticIndex(self, savepath: str, syntacticIndex: dict, anchor_textvalue: str, graphkey: str, propkey:str):
        """
        Function to save syntactic Index on disk in a "syntacticIndex" folder. This folder contains a JSON file for each entry in the index, named according to the anchor key.

        :param savepath: Path where to 
        :type savepath: str
        :param syntacticIndex: Syntactic Index to save on disk
        :type syntacticIndex: List[dict]
        :param anchor_textvalue: Textual value used for anchor node
        :type anchor_textvalue: str
        :param graphkey: Key in original dictionaries containing the graph to process
        :type graphkey: str
        :param propkey: Key in original dictionaries containing the properties
        :type propkey: str
        """

        savepath = f"{savepath}/model/syntacticIndex"
        params = {
            "anchor_textvalue": anchor_textvalue,
            "graphkey": graphkey,
            # "prop_rename": prop_rename,
            "propkey": propkey,
            # "getExtensions": getExtensions
        }

        os.makedirs(f"{savepath}/index", exist_ok=True)
        os.makedirs(f"{savepath}/params", exist_ok=True)

        copy_index = copy.deepcopy(syntacticIndex)
        for k, v in copy_index.items():
            for graph in v:
                graph['graph'] = nx.node_link_data(graph['graph'])
                
                # for path in graph['paths']:
                #     path['path'] = nx.node_link_data(path['path'])
                
                # if 'extensions' in graph.keys():

                #     for ext in graph['extensions']:
                #         ext['graph'] = nx.node_link_data(ext['graph'])
                #         for path in ext['paths']:
                #             path['path'] = nx.node_link_data(path['path'])
            # with open(f"{savepath}/index/{k}.json", 'w', encoding='utf-8') as f:
            #     json.dump(v, f, indent=4)

        with open(f"{savepath}/index/syntacticIndex.json", 'w', encoding='utf-8') as f:
            json.dump(copy_index, f, indent=4)

        # print(copy_index)     
        with open(f"{savepath}/params/syntacticIndexParams.json", 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4)

        # with open(f"{savepath}/syntacticIndex.json", 'w', encoding='utf-8') as f:
        #     json.dump(copy_index, f, indent=4)


    def loadSyntacticIndex(self, savepath: str) -> tuple:
        """
        Loads syntactic index stored at path

        :param savepath: path to folder which stores the syntactic index
        :type savepath: str
        :return: tuple containing the syntactic index and its parameters
        :rtype: tuple
        """
        savepath = f"{savepath}/model/syntacticIndex"

        with open (f"{savepath}/params/syntacticIndexParams.json", encoding='utf-8') as f:

            syntacticIndexParams = json.load(f)

        # syntacticIndex = {}

        # for file in glob(f"{savepath}/index/**.json"):

        #     anchor = os.path.basename(file)[:-5]
        #     # print(anchor)
        #     anchor = anchor.encode(encoding = 'UTF-8', errors = 'strict')

        #     with open(file, encoding='utf-8') as f:

        #         anchorDict = json.load(f)

        #         for graph in anchorDict:

        #             graph['graph'] = nx.node_link_graph(graph['graph'], directed=True, multigraph=False)
                    
        #             # for path in graph['paths']:
        #             #     path['path'] = nx.node_link_graph(path['path'], directed=True, multigraph=False)
                
        #         syntacticIndex[anchor] = anchorDict                

        with open (f"{savepath}/index/syntacticIndex.json", encoding='utf-8') as f:

            syntacticIndex = json.load(f)

        for v in syntacticIndex.values():
            for graph in v:
                graph['graph'] = nx.node_link_graph(graph['graph'], directed=True, multigraph=False)
                
                # for path in graph['paths']:
                #     path['path'] = nx.node_link_graph(path['path'], directed=True, multigraph=False)
                    
        return syntacticIndex, syntacticIndexParams


class SemanticIndex():

    def __init__(self, semanticIndexPath:str = '') -> None:
        """
        Constructor for the IndexModel class

        :param semanticIndexPath: Path to project storing the semanticIndex, defaults to ''
        :type semanticIndexPath: str, optional
        :param syntacticIndexPath: Path to project storing the syntacticIndex, defaults to ''
        :type syntacticIndexPath: str, optional
        """
        # if not specified, starts with empty index
        if semanticIndexPath:
            self.semanticIndex, self.semanticIndexParams = self.loadSemanticIndex(savepath=semanticIndexPath)
        else:
            self.semanticIndex = pd.DataFrame()
            self.semanticIndexParams = {}


    def __getTermFrequency(self, graph: Graph, textvalue: str = 'text', pos_filter: List[str] = []) -> dict:
        """
        Returns the frequency of each node label in a graph. The label
        can either be text, pos or lemma.

        :param graph: Graph to process
        :type graph: Graph
        :param textvalue: Textual value to use for tokens, defaults to 'text'
        :type textvalue: str, optional
        :param pos_filter: List of POS tags to specifiy which token to keep, defaults to None
        :type pos_filter: List[str], optional
        :return: Dictionnary count of each token in the graph
        :rtype: dict
        """

        def getTokens(list_nodes: NodeView, textvalue: str, pos_filter: List[str]=[]) -> List[str]:
            """
            Returns string representation of nodes in list_nodes. The string representation depends on textvalue, which can be a list of values. 
            Those values are the attributes of the node. POS filter allows to keep nodes based on their POS value
            """        

            # only one textvalue is given
            if not isinstance(textvalue, list):
                if pos_filter:
                    filtered_tokens = filter(lambda x: x[1]['pos'] in pos_filter, list_nodes)
                    tokens = map(lambda x: x[1][textvalue], filtered_tokens)
                else:
                    tokens = map(lambda x: x[1][textvalue], list_nodes)

                tokens = list(tokens)
            # multiple text values given 
            else:
                # multiple values for a token requested
                if pos_filter:
                    filtered_tokens = filter(lambda x: x[1]['pos'] in pos_filter, list_nodes)
                    tokens = [[token[1][tv] for tv in textvalue] for token in filtered_tokens]

                else:
                    tokens = [[token[1][tv] for tv in textvalue] for token in list_nodes]
                
                tokens = ['_'.join(token) for token in tokens]

            return tokens


        list_nodes = graph.nodes(data=True)
        tokens = getTokens(list_nodes, textvalue=textvalue, pos_filter=pos_filter )

        # counts frequency of each token representation
        return dict(Counter(tokens))    
        
    def getTermVectors(self, graph: Graph) -> pd.DataFrame:
        """
        Returns the sum of the vectors of each term in the graph

        :param graph: Graph to analyse
        :type graph: Graph
        :return: Vectors from semantic index of nodes in Graph
        :rtype: pd.DataFrame
        """
        
        textvalue = self.semanticIndexParams['textvalue']
        if isinstance(textvalue, list):
            list_terms = [[token[1][tv] for tv in textvalue] for token in graph.nodes(data=True)]
            list_terms = ['_'.join(token) for token in list_terms]

        else:
        # takes terms in graph with given text value (text, lemma, pos)
            list_terms =  [token[1][textvalue] for token in graph.nodes(data=True)]

        # only keeps the terms present in the vocabulary
        list_terms = list(set(list_terms).intersection(set(self.semanticIndex.index)))
        terms = self.semanticIndex.loc[list_terms]
        return terms

    # def trainSemanticIndex(self, list_graphs: List[dict], textvalue: str, pos_filter:List[str]=[], ngram_size:int=0, dict_rel: dict = {}, removePROPN: bool = True, min_weight:float=0 ,savepath: str='') -> None:


    def trainSemanticIndex(self, list_graphs: List[dict], textvalue: str, pos_filter:List[str]=[], dict_rel: dict = {}, 
                           removePROPN: bool = True, min_weight:float=0 ,savepath: str='') -> None:
        """
        Calculates an ESA matrix where rows are words and columns are concepts / classes.
        Textvalue can be a string or a list of string to select the string representation of nodes. POS_filter is a list of 
        POS tags to keep certain tokens.

        :param list_graphs: List of dict containing the SDP graphs to process with their metadata
        :type list_graphs: List[dict]
        :param textvalue: Text value to use for nodes
        :type textvalue: str
        :param pos_filter: List of POS tags to select what nodes to keep. If None, keeps all the tags, defaults to None
        :type pos_filter: List[str], optional
        :param dict_rel: Dictionary of properties, i.e. the columns of the matrix
        :type dict_rel: dict
        :param removePROPN: Whether PROPN tags are removed or not
        :type removePROPN: bool
        :param savepath: Path to save semantic index, defaults to None
        :type savepath: str, optional
        """
        tfidf = TfidfTransformer()

        # needed to select correct graphs
        selected_graphs = []
        for graph in list_graphs:
            try:
                # for graphs associated with any category other than Other
                g = graph['sdpgraph']
            except:
                # for graphs associated with Other category
                g = graph['graph']
            selected_graphs.append(g)

        # gets frequency of each term in the corpus
        dict_freqs = map(partial(self.__getTermFrequency, textvalue=textvalue, pos_filter=pos_filter), selected_graphs)
        # keeps the value in the index to use as column names
        index = map(lambda x: x['prop'], list_graphs)
        # creates frequency matrix
        df_tf = pd.DataFrame.from_dict(dict_freqs)
        df_tf.fillna(0, inplace=True)

        df_tf['CONCEPT-INDEX'] = list(index)
        # groups each doc by its class, then sums up the value of the tokens
        # the matrix is transposed so as to have a token x concept shape
        df_tf = df_tf.groupby('CONCEPT-INDEX').sum().T

        vec_freq = tfidf.fit_transform(df_tf)
        semantic_index = pd.DataFrame(vec_freq.todense(), columns=df_tf.columns, index=df_tf.index)
        # needed to rename classes to explicit name
        if dict_rel:
            tmp_dict = getRelationNames(dict_rel=dict_rel)
            # print(tmp_dict)
            s = semantic_index.columns.to_series()
            semantic_index.columns = s.map(tmp_dict).fillna(s)

        # removes any token that is PROPER NOUN
        if removePROPN:
            semantic_index.drop(semantic_index[semantic_index.index.str.endswith(('PROPN'))].index, axis=0, inplace=True)

        semantic_index[semantic_index < min_weight] = 0

        if savepath:
            self.saveSemanticIndex(savepath, semantic_index, textvalue, pos_filter, dict_rel, removePROPN) 


        self.semanticIndex = semantic_index
        self.semanticIndexParams = {
            "textvalue":  textvalue,
            "pos_filter": pos_filter,
            # "ngram_size": ngram_size,
            "dict_rel": dict_rel,
            "removePROPN": removePROPN
        }

    def saveSemanticIndex(self, savepath: str, semantic_index: pd.DataFrame, textvalue: str, pos_filter: List[str], dict_rel:dict, removePROPN:bool) -> None:
        """
        Save semantic index on disk as a "semanticIndex" folder contianing the matrix as a CSV

        :param savepath: Folder where to save semantic index
        :type savepath: str
        :param semantic_index: Semantic index to save
        :type semantic_index: pd.DataFrame
        :param textvalue: Text value used for the nodes
        :type textvalue: str
        :param pos_filter: List of POS tags to select tokens to keep
        :type pos_filter: List[str]
        :param dict_rel: Dictionary of properties, i.e. the columns of the matrix
        :type dict_rel: dict
        :param removePROPN: Whether PROPN tags are removed or not
        :type removePROPN: bool
        """

        savepath = f"{savepath}/model/semanticIndex"
        params = {
            "textvalue": textvalue,
            "pos_filter": pos_filter,
            "dict_rel": dict_rel,
            "removePROPN": removePROPN
        }

        os.makedirs(f"{savepath}/index", exist_ok=True)
        os.makedirs(f"{savepath}/params", exist_ok=True)

        semantic_index.to_csv(f"{savepath}/index/semanticIndex.csv")
        
        with open(f"{savepath}/params/semanticIndexParams.json", 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4)

    def loadSemanticIndex(self, savepath: str) -> tuple:
        """
        Loads semantic index stored at path

        :param savepath: path where semantic index folder is stored
        :type savepath: str
        :return: tuple containing the index and its parameter
        :rtype: tuple
        """

        semanticIndex = pd.read_csv(f"{savepath}/model/semanticIndex/index/semanticIndex.csv", index_col=0)
        with open (f"{savepath}/model/semanticIndex/params/semanticIndexParams.json", encoding='utf-8') as f:

            semanticIndexParams = json.load(f)
        
        return semanticIndex, semanticIndexParams
    



class IndexModel:

    def __init__(self, extractor, classifier, we=None) -> None:
        """
        Constructor for the IndexModel class

        :param semanticIndexPath: Path to project storing the semanticIndex, defaults to ''
        :type semanticIndexPath: str, optional
        :param syntacticIndexPath: Path to project storing the syntacticIndex, defaults to ''
        :type syntacticIndexPath: str, optional
        """
        self.extractor = extractor
        self.classifier = classifier
        self.we = we
        if self.we:
            self.mlClassifier = True
        else:
            self.mlClassifier = False

    def semanticClassification(self, candidate_graph: Graph, possible_labels: List[str] = [], thresh: float=.0, defaultPred:str='Other') -> dict:
        """
        Classify graph using the semantic index : sums up the vector representation of each node in the graph that is found in the semantic index,
        and takes the highest value. If possible_labels is provided, it will only consider these labels

        :param candidate_graph: Candidate graph to classify
        :type candidate_graph: Graph
        :param semantic_index: Semantic index to use
        :type semantic_index: pd.DataFrame
        :param textvalue: Text value of the tokens to use. Should be the same as used in the semantic index parameters
        :type textvalue: List[str]
        :param possible_labels: Possible labels associated with this graph, so as to narrow down the results, defaults to None
        :type possible_labels: List[str], optional
        :param thresh: Minimum threshold for a prediction, defaults to .7
        :type thresh: float, optional
        :param defaultPred: Default result if no other, defaults to 'Other'
        :type defaultPred: str, optional
        :return: Dictionary containing the prediction, its semantic score and the rule leading to that prediction
        :rtype: dict
        """

        # finds vector for each node in the candidate graphs
        terms = self.classifier.getTermVectors(candidate_graph)

        # calculates semantic score, i.e. harmonic mean
        terms = terms.apply(stats.hmean, axis=0)

        # keep only semantic scores for possible labels
        if possible_labels:
            terms = terms.loc[possible_labels]

        # get highest semantic score
        prediction = terms.index[terms.argmax()]
        score = terms[prediction]
        
        # checks if semantic score is higher than
        # semantic threshold
        if score > thresh:
            return {
                "prediction": prediction,
                "score": score,
                "rule": 'semantic'
            }
        else:
            return {
                "prediction": defaultPred,
                "score": score,
                "rule": 'tooWeak'
            }


    def NERclassification(self, sent_graph: nx.Graph, pred:str, candidate: dict) -> List[dict]:
        """
        Categorize and finds boundaries of Source and Target entities

        :param sent_graph: Sentence dependency graph
        :type sent_graph: nx.Graph
        :param pred: Label predicted during Relation Classification step
        :type pred: str
        :param candidate: Extracted relation
        :type candidate: dict
        :return: List of dictionaries, containing the types and boundaries of Source and Target entities
        :rtype: List[dict]
        """

        def checkEdge(edgeValue:str) -> bool:
            """
            Checks if edge label is a correct dependency role

            :param edgeValue: Edge label
            :type edgeValue: str
            :return: True if edge label is ok, else False
            :rtype: bool
            """
            if edgeValue.startswith('acl'):
                return False
            elif edgeValue in ('appos', 'conj', 'det'):
                return False
            return True

        def checkNode(node: dict) -> bool:
            """
            Checks if node POS tag is correct

            :param node: Node to verify
            :type node: dict
            :return: True if node POS tag is ok, else False
            :rtype: bool
            """
            if node['pos'] in ('PUNCT', 'PRON', 'ADP', 'CCONJ', 'ADV', 'AUX', 'DET'):
                return False
            return True

        def checkNodeContinuity(x:int, y:int) -> bool:
            """
            Checks if x and y are continuous nodes

            :param x: Nodes in graph
            :type x: int
            :param y: Other node in graph
            :type y: int
            :return: True if x and y are continuous
            :rtype: bool
            """
            if x > y:
                if x - y != 1:
                    return False
                return True 
            else: 
                if y - x != 1:
                    return False
                return True 

        def filterNodes(nodeList:List[int]) -> List[int]:
            """
            Filter list of nodes by their continuity

            :param nodeList: List of nodes to check
            :type nodeList: List[int]
            :return: Filtered list of nodes
            :rtype: List[int]
            """

            # checks list of nodes by their continuity
            ent_nodes = []
            for a, b in zip(nodeList[:-1], nodeList[1:]):
                if checkNodeContinuity(a, b):
                    ent_nodes.append(a)
                else:
                    break
            # does the same, in reverse order
            if ent_nodes:
                if checkNodeContinuity(ent_nodes[-1], nodeList[-1]):
                    ent_nodes.append(nodeList[-1])
            return ent_nodes

        def extendEntity(node:int, graph:nx.Graph) -> List[int]:
            """
            Determines the boundaries of an entity

            :param node: Root node of the entity in the graph
            :type node: int
            :param graph: Sentence dependency graph
            :type graph: nx.Graph
            :return: list of nodes belonging to the entity
            :rtype: List[int]
            """

            # filter nodes and edgesaccording to their pos tag and dependency role
            node_neighbours = list(graph.neighbors(node))
            node_neighbours = [i for i in node_neighbours if checkEdge(graph.edges[(node, i)]['dep'])]
            node_neighbours = [i for i in node_neighbours if checkNode(graph.nodes[i])]
            tmp_nodes = [node] + node_neighbours
            tmp_nodes.sort()

            node_index = tmp_nodes.index(node)

            # filter nodes by their continuity
            ent_nodes = filterNodes(tmp_nodes[node_index:])
            ent_nodes += filterNodes(tmp_nodes[:node_index][::-1])

            # if no more neighbour node, takes the root node as boundaries
            if not ent_nodes:
                ent_nodes = [node]
            return ent_nodes 

        def getEntStartEnd(ent_nodes, graph):

            ent_char_start = graph.nodes[ent_nodes[0]]['char_idx']
            ent_char_end = graph.nodes[ent_nodes[-1]]['char_idx']
            ent_char_end += len(graph.nodes[ent_nodes[-1]]['text'])

            return ent_nodes[0],ent_nodes[-1] + 1, ent_char_start, ent_char_end

        # finds type and boundaries of the Source entity
        source_node = candidate['source_nodes'][0]
        source_node = extendEntity(source_node, sent_graph)
        source_start, source_end, source_char_start, source_char_end = getEntStartEnd(source_node, sent_graph)

        if len(candidate['source_types']) == 1:
            source_type = candidate['source_types'][0]
        else:
            source_type = candidate['ner_rules'][pred]['source_type']

        # finds type and boundaries of the Target entity
        target_node = candidate['target_nodes'][0]
        target_node = extendEntity(target_node, sent_graph)

        target_start, target_end, target_char_start, target_char_end = getEntStartEnd(target_node, sent_graph)

        if len(candidate['target_types']) == 1:
            target_type = candidate['target_types'][0]
        else:
            target_type = candidate['ner_rules'][pred]['target_type']

        return [
            {
                "pred": source_type,
                "root_node": candidate['source_nodes'][0],
                "start": source_start,
                "end": source_end,
                "char_start": source_char_start,
                "char_end": source_char_end
            },
            {
                "pred": target_type,
                "root_node": candidate['target_nodes'][0],
                "start": target_start,
                "end": target_end,
                "char_start": target_char_start,
                "char_end": target_char_end
            }
        ]
    

        # return source_type, source_node, target_type, target_node
    
    def vectorize(self, graph):
        vec_graph = []
        for node in graph:
            word = getNodeText(graph, node, textvalue='text', cleanPropn=False)

            try:
                vec = self.we.word_vec(word)
                vec_graph.append(vec)
            except:
                pass
        return vec_graph

    def MLsemanticClassication(self, graph: Graph, possible_labels: List[str] = [], thresh: float = 0, defaultPred = 'Other') -> dict:
        """
        Semantic classification with a ML classifier. Requires word embeddings to encode the nodes

        :param graph: Candidate graph to categorize
        :type graph: Graph
        :param possible_labels: Possible labels of candidate graphs, defaults to []
        :type possible_labels: List[str], optional
        :param thresh: Semantic Threshold, defaults to 0
        :type thresh: float, optional
        :param defaultPred: Default prediction, defaults to 'Other'
        :type defaultPred: str, optional
        :return: Predicted label
        :rtype: dict
        """
        # get word embedding vector for each node in graph
        vec_graph = self.vectorize(graph)

        if vec_graph:
            # sums the vectors and feed the classifier for prediction
            vec_graph = reduce(lambda x, y: x + y, vec_graph)
            pred = self.classifier.predict_proba([vec_graph])[0]

            # filter probabilities for possible label 
            if possible_labels:
                pred = [x for x, y in zip(pred, self.classifier.classes_) if y in possible_labels]
                label = possible_labels[np.argmax(pred)]
            else:
                label = self.classifier.classes_[np.argmax(pred)]

            # gets highest probability
            score = pred[np.argmax(pred)]
            if score > thresh:
                return {
                    "prediction": label,
                    "score": score,
                    "rule": 'semantic'
                }

            else:
                return {
                "prediction": defaultPred,
                "score": score,
                "rule": 'tooWeak'
            }

        else:
            return {
                "prediction": defaultPred,
                "score": 0.0,
                "rule": 'NoPrediction'
            }

    def matchPattern(self, searchGraph:Graph, candidate:dict, nodeMatch:Callable, edgeMatch:Callable) -> dict:
        """
        Searches if candidate matches the pattern 'searchGraph'.

        :param searchGraph: Pattern that is test
        :type searchGraph: Graph
        :param candidate: Dictionary containing the candidate and its metadata
        :type candidate: dict
        :param nodeMatch: Function to determine if two nodes are equal
        :type nodeMatch: Callable
        :param edgeMatch: Function to determine if two edges are equal
        :type edgeMatch: Callable
        :return: Dictionary of matching nodes if there is a match, otherwise None
        :rtype: dict
        """
            
        # search in this subgraph if any possible pattern matches
        matcher = DiGraphMatcher(searchGraph, candidate['graph'], node_match=nodeMatch, edge_match=edgeMatch)

        if matcher.is_isomorphic():
            possibles_labels = candidate["props"]
            ner_rules = candidate['ner_rules']
            # extracts the corresponding subgraph
            # where there has been a match
            # candidates_nodes = list(matcher.mapping.keys())
            candidates_nodes = list(searchGraph.nodes())


            source_nodes, target_nodes = [], []

            for k, v in matcher.mapping.items():

                if v in candidate['source_nodes']:
                    source_nodes.append(k)
                elif v in candidate['target_nodes']:
                    target_nodes.append(k)


            source_types = [x for x in candidate['source_types']]       
            target_types = [x for x in candidate['target_types']]       


            # candidates.append(
            return    {
                "nodes": candidates_nodes,
                "labels": possibles_labels,
                "graph": candidate['graph'],
                'source_types': source_types,
                'source_nodes': source_nodes,
                'target_types': target_types,
                'target_nodes': target_nodes,
                'ner_rules': ner_rules
                }
                # )
        else:
            # Test to match graphs only by their POS, and ignoring their edges
            matcher = DiGraphMatcher(searchGraph, candidate['graph'], node_match=lambda x,y: x['pos'] == y['pos'], edge_match=lambda x, y: True)

            if matcher.is_isomorphic():
                possibles_labels = candidate["props"]
                ner_rules = candidate['ner_rules']
                # extracts the corresponding subgraph
                # where there has been a match
                # candidates_nodes = list(matcher.mapping.keys())
                candidates_nodes = list(searchGraph.nodes())

                source_nodes, target_nodes = [], []

                for k, v in matcher.mapping.items():
                    if v in candidate['source_nodes']:
                        source_nodes.append(k)
                    elif v in candidate['target_nodes']:
                        target_nodes.append(k)

                source_types = [x for x in candidate['source_types']]       
                target_types = [x for x in candidate['target_types']]       


                # candidates.append(
                return    {
                    "nodes": candidates_nodes,
                    "labels": possibles_labels,
                    "graph": candidate['graph'],
                    'source_types': source_types,
                    'source_nodes': source_nodes,
                    'target_types': target_types,
                    'target_nodes': target_nodes,
                    'ner_rules': ner_rules
                    }
            else:
                return None


    def predict(self, graph: Graph, thresh: float = 0) -> dict:
        """
        Classify candidate graph using the semantic and syntactic indexes

        :param graph: Graph to analyse
        :type graph: Graph
        :param thresh: Threshold for semantic prediction, defaults to 0
        :type thresh: float, optional
        :return: Dictionnary containing the prediction, the rule leading to it and its score
        :rtype: dict
        """
        pred = 'Other'
        score = 0
        rule = 'noAnchorMatch'
        anchor, anchortext = None, None

        graph_size = graph.size()
        anchor, anchortext = self.extractor.getGraphAnchor(graph=graph, anchor_textvalue=self.extractor.syntacticIndexParams['anchor_textvalue'])
        
        # gets possible patterns correspoding to this anchor
        if anchortext in self.extractor.syntacticIndex.keys():
            rule = 'noPatternMatch'

            # gets syntactic patterns from Syntactic Index
            possible_patterns = self.extractor.syntacticIndex[anchortext]
            possible_patterns = filter(lambda x: x['size'] == graph_size, possible_patterns)

            # finds if patterns match subgraph in candidate graph
            candidates = map(lambda x: self.matchPattern(graph, x, nodeEq, edgeEq), possible_patterns)
            candidates= filter(lambda x: x, candidates)

            predictions = []
            # categorize each found subgraph
            for candidate in candidates:
                c_graph = candidate['graph']

                # possible labels associated with candidate graph
                possible_labels = [x['name'] for x in candidate['labels']]

                # semantic predictions depends on the classifier 
                if not self.mlClassifier:

                    semantic_class = self.semanticClassification(c_graph, possible_labels=possible_labels, thresh=thresh)
                else:
                    semantic_class = self.MLsemanticClassication(c_graph, possible_labels=possible_labels, thresh=thresh)
                
                pred = semantic_class['prediction']
                score = semantic_class['score']
                rule = semantic_class['rule']

                predictions.append({
                    "pred": pred,
                    "score": score,
                    "rule": rule,
                    "anchor": anchor,
                    "anchortext": anchortext,
                    "candidate": candidate

                }
                )
            if predictions:
                # returns the predictions with the highest confident score
                predictions.sort(key=lambda x: x['score'], reverse=True)
                return predictions[0]

            
        return {
            "pred": pred,
            "score": score,
            "rule": rule,
            "anchor": anchor,
            "anchortext": anchortext,

        }
    

    # def extractCandidatesFromGraph(self, graph, fuzzyMatch:bool = False):

    def extractCandidatesFromGraph(self, graph:nx.Graph ) -> List[dict]:
        """
        Finds every node that can be considered as anchor / predicate, i.e. are found in the graph index returns every graph associated with every predicate

        :param graph: Sentence graph to search for candidates
        :type graph: nx.Graph
        :return: List of all found candidates in graph
        :rtype: List[dict]
        """
        def getCandidates(searchGraph, patternDict, nodeMatch, edgeMatch):
            """
            Search pattern in graph. If found, returns the corresponding nodes
            with their possible labels / classes
            nodeMatch and EdgeMatch are the function to define if two graphs are 
            isomorphic
            """     
            # search in this subgraph if any possible pattern matches
            matcher = DiGraphMatcher(searchGraph, patternDict['graph'], node_match=nodeMatch, edge_match=edgeMatch)

            if matcher.subgraph_is_isomorphic():

                possibles_labels = patternDict["props"]
                
                # extracts the corresponding subgraph where there has been a match
                candidates_nodes = list(matcher.mapping.keys())

                return    {
                    "candidates_nodes": candidates_nodes,
                    "possibles_labels": possibles_labels
                    }
            

            else:
                # more flexible matching, which ignores dependency matching between graphs
                matcher = DiGraphMatcher(searchGraph, patternDict['graph'], node_match=lambda x,y: x['pos'] == y['pos'], edge_match=lambda x, y: True)

                if matcher.subgraph_is_isomorphic():

                    possibles_labels = patternDict["props"]
                    
                    # extracts the corresponding subgraph
                    # where there has been a match
                    candidates_nodes = list(matcher.mapping.keys())

                    # candidates.append(
                    return    {
                        "candidates_nodes": candidates_nodes,
                        "possibles_labels": possibles_labels
                        }
                else:
                    return None
                # return None


        def filterCandidates(candidates: List[dict]) -> List[dict]:
            """
            Keep only non-overlaping paths

            :param candidates: List of candidate graphs
            :type candidates: List[dict]
            :return: List of longest possible candidate graphs
            :rtype: List[dict]
            """
            set_paths = []
            for c in candidates:
                c['candidates_nodes'].sort()
                # print(c)
                if c['candidates_nodes'] not in set_paths:
                    set_paths.append(c['candidates_nodes'])

            set_paths.sort(key=lambda x: len(x))
            not_allowed_paths = []

            # does not allow paths that are subsets of other paths
            if len(set_paths) > 1:
                for x, y in combinations(set_paths, 2):
                    if set(x).issubset(set(y)):
                        not_allowed_paths.append(x)

            candidates = filter(lambda x: x['candidates_nodes'] not in not_allowed_paths, candidates)

            # removes duplicate candidates
            filter_candidates = []
            for x in candidates:
                if x not in filter_candidates:
                    filter_candidates.append(x)

            # gathers labels for the same path
            filter_candidates.sort(key=lambda x: x['candidates_nodes'])

            final_candidates = []
            for key, group in groupby(filter_candidates, key=lambda x: x['candidates_nodes']):
                labels = []
                for g in group:
                    labels.extend(g['possibles_labels'])

                final_candidates.append(
                    {
                        "nodes": key,
                        "labels": labels
                    }
                )

            return final_candidates


        def getCandidatesFromNode(node: Graph.nodes) -> List[dict]:
            """            
            Extracts all possible candidate graphs where node is the anchor

            :param node: the node that must serves as anchor
            :type node: Graph.nodes
            :return: List of candidate patterns where node is the anchor node
            :rtype: List[dict]
            """
            def process(node_text:str):
                all_candidates = []                
                # subgraph where current node is the anchor node might be a subgraph of
                # main graph, might also be the main graph itself

                # all possible patterns associated with this anchor
                possible_patterns = self.extractor.syntacticIndex[node_text]

                candidates = map(lambda x: getCandidates(searchGraph=graph, patternDict= x, 
                                                         nodeMatch = nodeEq, edgeMatch = edgeEq), possible_patterns)  
                candidates = filter(lambda x: x, candidates)
                candidates = list(candidates)

                # only keep longest possible candidates
                candidates = filterCandidates(candidates)
                for c in candidates:

                    c['graph'] = nx.subgraph(graph, c['nodes'])
                    all_candidates.append(
                        {
                            "anchorNode": node,
                            "anchorText": node_text,
                            "candidate": c
                        }
                    )

                return all_candidates

            node_text = getNodeText(graph, node, textvalue=self.extractor.syntacticIndexParams['anchor_textvalue'])

            if node_text in self.extractor.syntacticIndex.keys():
                return process(node_text)

            else:
                if fuzzyMatch:
                    lvsht_dist = map(lambda x:(x, lvhst.distance(node_text, x)), list(self.extractor.syntacticIndex.keys()))
                    lvsht_dist = filter(lambda x: x[1] == 1, lvsht_dist)
                    candidates = []
                    for c in lvsht_dist:
                        # print(node_text, c)
                        
                        candidates.extend(process(c[0]))
                    return candidates

                else:
                    return []

        all_candidates = map(getCandidatesFromNode, graph.nodes())
        # flattens the list
        all_candidates = (x for y in all_candidates for x in y)

        return all_candidates

    def extractFacts(self, doc, thresh=0) -> List[dict]:
        """
        Extract relations and entities from spaCy Doc 

        :param doc: sentence to extract relations and entities from
        :type doc: Doc
        :param thresh: Semantic threshold, defaults to 0
        :type thresh: int, optional
        :return: List of relations and entities extracted from Doc
        :rtype: List[dict]
        """

        # converts doct to graph
        dict_graph = doc2graph(doc)
        # extracts subgraphs from doc dependency graph

        candidates = self.extractCandidatesFromGraph(dict_graph['graph'])
        candidates = list(candidates)

        all_preds = []
        for c in candidates:
            # categorize each candidate graph
            c_graph = c['candidate']['graph']
            prediction = self.predict(c_graph, thresh=thresh)

            # finds entities boundaries and types, if graph not Other
            if prediction['pred'] != 'Other':
                ent_pred = self.NERclassification(dict_graph['graph'], prediction['pred'], prediction['candidate'])
                all_preds.append({"fact": prediction, "ner": ent_pred})
        
        return all_preds


    # def extractFacts(self, doc, thresh=0, fuzzyMatch=False):

    #     dict_graph = doc2graph(doc)
    #     candidates = self.extractCandidatesFromGraph(dict_graph['graph'], fuzzyMatch=fuzzyMatch)
    #     candidates = list(candidates)

    #     all_preds = []
    #     for c in candidates:
    #         # print('cc', c)
    #         c_graph = c['candidate']['graph']
    #         prediction = self.predict(c_graph, thresh=thresh)

    #         if prediction['pred'] != 'Other':

    #             ent_pred = self.NERclassification(dict_graph['graph'], prediction['pred'], prediction['candidate'])
            
    #             all_preds.append({"fact": prediction, "ner": ent_pred})
        
    #     return all_preds

    def evaluate(self, y_true:List[str], y_pred:List[str], ignoreOther:bool = True) -> dict:
        """
        Helper function to evaluate model in terms of Precision, Recall and F1

        :param y_true: True labels
        :type y_true: List[str]
        :param y_pred: Predicted labels
        :type y_pred: List[str]
        :param ignoreOther: Whether to ignore Other labels, defaults to True
        :type ignoreOther: bool, optional
        :return: Dictionary with P,R,F1 scores and report
        :rtype: dict
        """
        # print(len(y_true), len(y_pred))
        if ignoreOther:
            i_other = [i for i, x in enumerate(y_true) if x == 'Other']
            y_true = [x for i, x in enumerate(y_true) if i not in i_other]
            y_pred = [x for i, x in enumerate(y_pred) if i not in i_other]
        
        # print('True :', len(y_true), 'Pred :',len(y_pred))

        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')


        report = classification_report(y_true, y_pred, digits=3)

        return {
            'P': p,
            "R": r,
            "F1": f1,
            "report": report,
            # "confusion": ax
        }


        
