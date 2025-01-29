# %%
import sys 

sys.path.append('..')

import re
from typing import List 
from multiprocessing.dummy import Pool

from functools import partial

from spaczz.matcher import FuzzyMatcher
from spacy.util import filter_spans
from spacy.language import Language
from spacy.tokens import Doc

import json
import networkx as nx
import pandas as pd
from itertools import groupby

from utils.utils import saveCorpus, saveDocument, doc2graph

# print('TP')

class TextProcessor:
        # re_brackets = re.compile(r'(\(|\[)+[^\)\]]*(\)|\])+')
    re_doublespaces = re.compile(r' {2,}')
    re_spacecomma = re.compile(r' ,')
    re_start_trailings = re.compile(r'^(,? |-|")')
    re_end_trailings = re.compile(r'( |")$')
    re_start_non_alpha = re.compile(r'^[^a-zA-Z]+')
    re_non_alpha = re.compile(r'["«»\[\]]')
    re_phonetic = re.compile(r'[/\[][^/\]]*[/\]]( ?Écouter)?')

    def __init__(self, nlp: Language) -> None:
        self.nlp = nlp

    def cleanText(self, text: str) -> str:
        """
        Cleaning pipeline : removes double spaces, space with commas, trailing punctuations, non-alphanumeric and phonetic transcriptions

        :param text: Text to clean
        :type text: str
        :return: Cleaned text
        :rtype: str
        """

        text = self.re_phonetic.sub('', text)
        # text = re_brackets.sub('', text)
        text = self.re_doublespaces.sub(' ', text)
        text = self.re_spacecomma.sub(',', text)
        text = self.re_start_trailings.sub('', text)
        text = self.re_end_trailings.sub('', text)
        text = self.re_start_non_alpha.sub('', text)
        text = self.re_non_alpha.sub('', text)
        # if not text:
        #     return None
        return text


    def groupbyPropBySent(self, entityData: dict, savepath:str = '') -> dict:
        """
        Loops over propertie key in entityData and groups them by
        their common sentence

        :param entityData: Dictionary containing data about an entity
        :type entityData: dict
        :param savepath: Path where to save files
        :type savepath: str
        :return: Dictionnary for that entity where properites are grouped by sentences
        :rtype: dict
        """
        # loops over each example of properties and groups them by their common sentence
        all_prop = [sent for prop in entityData['properties'] for sent in prop['sents']]
        all_prop.sort(key=lambda x: x['sent'])
        all_prop = groupby(all_prop, lambda x: x['sent']) 

        sentProp = []
        sentProp_append = sentProp.append
        for sent_i, data in enumerate(all_prop):
            sent, group = data[0], data[1]

            sentProp_append(
                {
                    'sent': sent,
                    'sent_i': sent_i,
                    'props': list(group)
                }
            )
        groupedSentProp = {
            "id": entityData['id'],
            "content": sentProp
        }
        if savepath:
            id = groupedSentProp['id']
            with open(f"{savepath}/{id}.json", 'w', encoding='utf-8') as f:
                json.dump(groupedSentProp, f, indent=4)

        return groupedSentProp
    
    def multi_groupbyPropBySent(self, list_entities_data: List[dict], n_core:int=4, savepath:str='') -> List[dict]:
        """
        Applies groupbyPropSent in parallel processing

        :param list_entities_data: List of dictionnaries containing entities data
        :type list_entities_data: List[dict]
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: _description_, defaults to ''
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities with updated data
        :rtype: List[dict]
        """

        with Pool(n_core) as p:
            partial_func = partial(self.groupbyPropBySent, savepath=savepath)
            results = p.map(partial_func, list_entities_data)
        return results
   

    def prepare_corpus(self, list_entities_data: List[dict],removeNoMatch:bool=True, keep_filter_prop:List[str]=[], n_core:int=4) -> List[dict]:
        """
        Prepare corpus obtained with WikidataParser for processing

        :param list_entities_data: List en entities dictionnary, as obtained by WikidataParser (basically, data stored in a entity_data folder)
        :type list_entities_data: List[dict]
        :param removeNoMatch: If True, removes property sample where there is a NO-MATCH for either source or target entities, defaults to True
        :type removeNoMatch: bool, optional
        :param keep_filter_prop: list of prop to keep
        :type keep_filter_prop:List[str], optional
        :param n_core: Number of core to use for parrallel processing, defaults to 4
        :type n_core: int, optional
        :return: List of dictionnaries containing the entity ID, the sentences and the properties found in them
        :rtype: List[dict]
        """

        groupedSent = self.multi_groupbyPropBySent(list_entities_data=list_entities_data, n_core=n_core)
        func_clean = self.cleanText
        for ent_cont in groupedSent:
            for x in ent_cont['content']:
                x['sent'] = func_clean(x['sent'])
            # ent_cont['content'] = list(map(lambda x: self.cleanText(x['sent']), ent_cont['content']))
            
            for dict_sent in ent_cont['content']:
                if removeNoMatch:
                    dict_sent['props'] = filter(lambda x: x['source'] != 'NO-MATCH', dict_sent['props'])
                    dict_sent['props'] = filter(lambda x: x['target'] != 'NO-MATCH', dict_sent['props'])
                    dict_sent['props'] = list(dict_sent['props'])
                if keep_filter_prop:
                    dict_sent['props'] = [x for x in dict_sent['props'] if x['prop'] in keep_filter_prop]
                    
                for dict_prop in dict_sent['props']:
                    dict_prop['sent'] = dict_sent['sent']

            ent_cont['content'] = list(filter(lambda x: x['props'], ent_cont['content']))
        return groupedSent
        
    def sample_corpus(self, thresh:int, df_data:pd.DataFrame):

        tmp = []

        for i, prop_count in zip(df_data['prop'].value_counts().index, df_data['prop'].value_counts()):
            if prop_count > thresh:
                vals = df_data[df_data['prop'] == i].sample(n=thresh)
            else:
                vals = df_data[df_data['prop'] == i]
            tmp.append(vals)
        df_samples = pd.concat(tmp)

        df_samples.reset_index(inplace=True)
        df_samples.rename(columns={'index': 'id'}, inplace=True)
        return df_samples



    def __getSourceTargetRoots(self, doc: Doc, terms: List[str], ratio: int = 90) -> List[int]:
        """
        Finds the root node corresponding to the source and target entities of given property.
        The terms are found in spaCy Doc with Fuzzy Matching to find most similar tokens. 
        If Source of Target entities are multi-tokens entities, only take the root token.
        Return a list of position of those tokens in the Doc.

        :param doc: spaCy Doc to process
        :type doc: Doc
        :param terms: List of terms corresponding to the Source and Target entities to search for
        :type terms: List[str]
        :param ratio: Minimum similarity score for fuzzy matching, defaults to 90
        :type ratio: int, optional
        :return: List of the position of the tokens matching the Source and Target entities
        :rtype: List[int]
        """

        # the FuzzyMatcher is used to find subpattern in 
        # documents that will correspond to property values

        matcher = FuzzyMatcher(self.nlp.vocab)
        patterns = [self.nlp.make_doc(text) for text in terms]
        matcher.add("SourceTargetsTerms", patterns)

        matches = matcher(doc)
        # need to use filter_spans to avoid overlapping spans
        # list_candidates = [doc[start : end] for match_id, start, end, match_ratio in matches if match_ratio >= ratio]
        list_candidates = [doc[start : end] for label, start, end, match_ratio, pattern in matches if match_ratio >= ratio]


        # print(list_candidates)
        list_candidates = filter_spans(list_candidates)
        # the first element of the list is the root node of the Source, the second of the Target
        # list_candidates = [x.root.i for x in list_candidates]
        return list_candidates


    def __getSDP(self, graph: nx.Graph, list_candidate_nodes: List[int]):
        """
        Finds SDP path between Source and Target nodes

        :param graph: Graph to process
        :type graph: nx.Graph
        :param list_candidate_nodes: List of nodes which correspond to the Source and Target entity nodes
        :type list_candidate_nodes: List[int]
        :return: The list of nodes corresponding to the SDP path between Source and Target
        :rtype: _type_
        """
        try:   
            # first try to get sdp between two nodes  
            sdp = nx.shortest_path(graph, list_candidate_nodes[0], list_candidate_nodes[1])
        except:
            try:
                # if sdp not found, finds lowest common ancestor, then
                # computes shortest path from lca and both nodes
                lca = nx.lowest_common_ancestor(graph, list_candidate_nodes[0], list_candidate_nodes[1])
                sdp1 = nx.shortest_path(graph, lca, list_candidate_nodes[0]) 
                sdp2 = nx.shortest_path(graph, lca, list_candidate_nodes[1])

                sdp = list(set(sdp1 + sdp2))
            except:
                # no sdp found, because of syntactic error or
                # error in the dependency parsing 
                sdp = 'SYNTACTIC-ERROR'
        return sdp

    def processDocument(self, dict_ent: dict, savepath:str='') -> dict:
        """
        Pipeline to transform corpus of text into directed dependency graphs. 
        Finds the subgraph corresponding to the SDP between Source and Target entities

        :param dict_doc: Dictionary containing the sentence to process, alongside the source and target entities to find
        :type dict_doc: dict
        :param savepath: Path to save folder, defaults to savepath
        :type savepath: str, optional
        :return: Updated dict_doc with graph
        :rtype: dict
        """
        # processes a sentence with spaCy to get the POS tags and dependency parsing
        docs = self.nlp.pipe((x['sent'] for x in dict_ent['content']))

        for dict_sent, doc in zip(dict_ent['content'], docs):

            dict_graph = doc2graph(doc)
            dict_sent.update(dict_graph)
            
            # process each statement associated with the sentence
            for dict_prop in dict_sent['props']:
                
                dict_prop['sdpgraphs'] = []
                # # only gets SDPS for labelled texts
                if dict_prop['prop'] != 'Other':
                    
                    # finds nodes corresponding to the Source and Target entities
                    list_candidate_nodes_src = self.__getSourceTargetRoots(doc, [dict_prop['source']])
                    list_candidate_nodes_trgt = self.__getSourceTargetRoots(doc, [dict_prop['target']])

                    for candidate_nodes_src in list_candidate_nodes_src:

                        src_root_nodes = candidate_nodes_src.root.i 

                        for candidate_nodes_trgt in list_candidate_nodes_trgt:

                            trgt_root_nodes = candidate_nodes_trgt.root.i 

                            # tries to find the Shortest Dependency Path between the Source and Target root nodes
                            sdp = self.__getSDP(dict_graph['graph'], list_candidate_nodes=[src_root_nodes, trgt_root_nodes])

                            # error in the parsing, returns a string telling SYNTAXIC ERROR 
                            if isinstance(sdp, str):
                                sdpgraph = sdp 
                            else:
                            # no error, returns the subgraph matching the SDP
                                sdpgraph = dict_graph['graph'].subgraph(sdp)

                            dict_prop['sdpgraphs'].append(
                                {
                                 "sourceNode": [x.i for x in candidate_nodes_src],
                                 "targetNode": [x.i for x in candidate_nodes_trgt],
                                 "sourceNodeRoot": src_root_nodes,
                                 "targetNodeRoot": trgt_root_nodes,
                                 "sdpgraph": sdpgraph    
                                }
                            )

                else:
                    dict_prop['sdpgraphs'].append(
                        {
                            "sourceNode": None,
                            "targetNode": None,
                            "sourceNodeRoot": None,
                            "targetNodeRoot": None,
                            "sdpgraph": dict_graph['graph']
                        }
                    )


        if savepath:
            saveDocument(savepath=savepath, data= dict_ent)

        return dict_ent

    def processCorpus(self, corpus: List[dict], n_core: int=6, savepath:str = '') -> List[dict]:
        """
        Helper function to process whole corpus

        :param corpus: List of dictionnaries containing the documents to process
        :type corpus: List[dict]
        :param n_core: Number of core to use for parallel processing, defaults to 6
        :type n_core: int, optional
        :param savepath: Path to save processed corpus, defaults to None
        :type savepath: str, optional
        :return: List of dictionnaries containing the processed corpus
        :rtype: List[dict]
        """

        with Pool(n_core) as p:
            partial_func = partial(self.processDocument, savepath=savepath)
            corpus = p.map(partial_func, corpus)

        return corpus



    # def processDocument(self, dict_ent: dict) -> dict:
    #     """
    #     Pipeline to transform corpus of text into directed dependency graphs. 
    #     Finds the subgraph corresponding to the SDP between Source and Target entities
    #     :param dict_ent: Dictionary containing the sentence to process, alongside the source and target entities to find
    #     :type dict_ent: dict
    #     :return: Updated dict_doc with graph
    #     :rtype: dict
    #     """
    #     # processes a sentence with spaCy to get the POS tags and dependency parsing
    #     docs = self.nlp.pipe((x['sent'] for x in dict_ent['content']))
    #     for dict_sent, doc in zip(dict_ent['content'], docs):

    #         dict_graph = doc2graph(doc)
    #         dict_sent.update(dict_graph)
            
    #         # process each statement associated with the sentence
    #         for dict_prop in dict_sent['props']:   
    #             dict_prop['sdpgraphs'] = []
    #             # # only gets SDPS for labelled texts
                    
    #             # finds nodes corresponding to the Source and Target entities
    #             list_candidate_nodes_src = self.__getSourceTargetRoots(doc, [dict_prop['source']])
    #             list_candidate_nodes_trgt = self.__getSourceTargetRoots(doc, [dict_prop['target']])

    #             for candidate_nodes_src in list_candidate_nodes_src:
    #                 src_root_nodes = candidate_nodes_src.root.i 

    #                 for candidate_nodes_trgt in list_candidate_nodes_trgt:

    #                     trgt_root_nodes = candidate_nodes_trgt.root.i 
    #                     # tries to find the Shortest Dependency Path between the Source and Target root nodes
    #                     sdp = self.__getSDP(dict_graph['graph'], list_candidate_nodes=[src_root_nodes, trgt_root_nodes])

    #                     # error in the parsing, returns a string telling SYNTAXIC ERROR 
    #                     if isinstance(sdp, str):
    #                         sdpgraph = sdp 
    #                     else:
    #                     # no error, returns the subgraph matching the SDP
    #                         sdpgraph = dict_graph['graph'].subgraph(sdp)

    #                     dict_prop['sdpgraphs'].append(
    #                         {
    #                             "sourceNode": [x.i for x in candidate_nodes_src],
    #                             "targetNode": [x.i for x in candidate_nodes_trgt],
    #                             "sourceNodeRoot": src_root_nodes,
    #                             "targetNodeRoot": trgt_root_nodes,
    #                             "sdpgraph": sdpgraph    
    #                         }
    #                     )
        return dict_ent

