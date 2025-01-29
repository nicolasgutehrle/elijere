import spacy
from spacy.tokens import Doc, DocBin
import os
from typing import List
import re
from spacy.tokens import Span, Doc

from heideltimetagger.main.heideltime_tagger import  heideltime_tagging, set_heideltime_extension

# from .components.simple_chunker import extract_chunks
from factextraction.model.model import IndexModel

from factextraction.utils.utils import doc2graph

from itertools import groupby
import pkg_resources

class NLPPipe:

    def __init__(self, model = None, heideltime_config = None, factextractor_config = None) -> None:
        if model:
            self.nlp = spacy.load(model)
        else:
            try:
                self.nlp = spacy.load('fr_EMONTAL_NER_PER_LOC')

            except:
                # CHANGER CHEMIN IMPRESSO
                nlpmodel_path = pkg_resources.resource_filename(__name__, f"ner/packages/dist/fr_EMONTAL_NER_PER_LOC-0.1.tar.gz")
                os.system(f"pip install {nlpmodel_path}")
                self.nlp = spacy.load('fr_EMONTAL_NER_PER_LOC')

        
        if not heideltime_config:
            # heidelpath = pkg_resources.resource_filename(__name__, f"heideltimetagger/heideltime-standalone")
            heidelpath = "heideltimetagger/heideltime-standalone"

            self.heideltime_config = {
                "heidelpath": heidelpath,
                "filepath": '',
                "lg": 'FRENCH',
                "dct": '',
                "doctype": 'NEWS',
                "configpath": f'{heidelpath}/config.props',
                "heideltime": f'{heidelpath}/de.unihd.dbs.heideltime.standalone.jar',
                "infer_day": True
            }
        else:
            self.heideltime_config = heideltime_config

        if factextractor_config:
            self.set_fact_extensions()
            extractor = factextractor_config['extractor']
            classifier = factextractor_config['classifier']
            # classifier requires external word embeddings
            if 'we' in factextractor_config.keys():
                we = factextractor_config['we']
                self.factextractor = IndexModel(extractor=extractor, classifier=classifier, we=we)
            else:
                self.factextractor = IndexModel(extractor=extractor, classifier=classifier)

        print("Available components: ", self.nlp.pipe_names)

    def clean_text(self, text: str) -> str:
        """
        Slightly preprocess the given text by removing newline, multiple spaces
        """
        text = text.strip()
        text = text.replace('\n', '')
        text = re.sub(r' {2,}', '', text)
        return text

    def add_metadata2doc(self, doc: str, metadata: dict) -> Doc:
        """
        Preprocess documents to add metadata
        """
        # dict_doc['content'] = self.clean_text(dict_doc['content'])
        # doc = self.clean_text(doc)
        # # doc = self.nlp.make_doc(dict_doc['content'])
        # doc = self.nlp.make_doc(doc)
        # dict_doc['heideltime_config'] = self.heideltime_config
        doc.user_data = metadata
        return doc

    def add_metadata2corpus(self, corpus: List[str], list_metadata: List[dict]) -> List[Doc]:
        """
        Preprocess list of documents to add their respective metadata
        """
        return [self.add_metadata2doc(doc, metadata) for doc, metadata in zip(corpus, list_metadata)]
        
    def prepare_doc(self, doc):
        """
        Cleans text and returns it in Doc format
        """
        doc = self.clean_text(doc)
        # doc = self.nlp.make_doc(dict_doc['content'])
        doc = self.nlp.make_doc(doc)
        return doc

    def prepare_corpus(self, corpus):
        """
        Applies prepare_doc() to a list of str that makes the corpus
        """
        return [self.prepare_doc(doc) for doc in corpus]

    def filterFact(self, doc, fact:dict) -> bool:
        """
        After extracting facts, selects facts to keep. For that, the src and target entities must have previously been found by NER tagging
        Exception is made for Misc entities, which are not detected by the NER tagger

        :param fact: output of the fact extractor prediction step
        :type fact: dict
        :return: Either if fact must be kept or not
        :rtype: bool
        """

        # print(fact['fact'])
        src_ent = fact['ner'][0]
        trgt_ent = fact['ner'][0]

        check_src, check_trgt = False, False

        ents = [(ent, ent.label_, ent.start_char, ent.end_char ) for ent in doc.ents] + [(ent, ent.label_, ent.start_char, ent.end_char ) for ent in doc.spans['TIMEX']]

        # checks if source entity is within an already tagged entity
        for ent in ents:
            if ent[2] <= src_ent['char_start'] <= ent[3] and ent[2] <= src_ent['char_end'] <= ent[3]:
                # print(src_ent)
                check_src = True 

        # same thing for target entity
        if trgt_ent['pred'] != 'Misc':
            for ent in ents:
                if ent[2] <= trgt_ent['char_start'] <= ent[3] and ent[2] <= trgt_ent['char_end'] <= ent[3]:
                    # print(trgt_ent)
                    check_trgt = True
        else:
            check_trgt = True 
        
        if check_src and check_trgt:
            return True
        return False

#     def addFactSpans(self, doc: Doc, filtered_facts:List[dict]) -> None:
#         """
#         Adds selected facts as Spans, where each token of a fact is a separate fact (so as to account for discontinuous facts)
#         Each span is labelled with the fact label and an id

#         :param doc: spaCy doc 
#         :type doc: Doc
#         :param filtered_facts: list of selected facts, from the filterFact function
#         :type filtered_facts: List[dict]
#         """
# #             {
#             #     "pred": source_type,
#             #     "root_node": candidate['source_nodes'][0],
#             #     "start": source_start,
#             #     "end": source_end,
#             #     "char_start": source_char_start,
#             #     "char_end": source_char_end
#             # },


#         fact_spans = []

#         for i, f in enumerate(filtered_facts):
#             nodes = f['fact']['candidate']['nodes']
#             label = f['fact']['pred']

#             score = f['fact']['score']
#             rule = f['fact']['rule']
#             anchortext = f['fact']['anchortext']
#             # source target 

#             source_type = f['ner'][0]['pred']
#             source_node = f['ner'][0]['root_node']
#             source_start = f['ner'][0]['char_start']
#             source_end = f['ner'][0]['char_end']

#             target_type = f['ner'][1]['pred']
#             target_node = f['ner'][1]['root_node']
#             target_start = f['ner'][1]['char_start']
#             target_end = f['ner'][1]['char_end']

#             for n in nodes:
#                 fact_span = Span(doc, n, n + 1, label=f"{label}_{i + 1}" )
                
#                 fact_span._.FACT_score = score
#                 fact_span._.FACT_rule = rule
#                 fact_span._.FACT_anchor = anchortext

#                 fact_spans.append(fact_span)

#         doc.spans['FACTS'] = fact_spans

    def addFactSpans(self, doc: Doc, filtered_facts:List[dict]) -> None:
        """
        Adds selected facts as Spans, where each token of a fact is a separate fact (so as to account for discontinuous facts)
        Each span is labelled with the fact label and an id

        :param doc: spaCy doc 
        :type doc: Doc
        :param filtered_facts: list of selected facts, from the filterFact function
        :type filtered_facts: List[dict]
        """

        def createEntSpan(ent):

            ent_type = ent['pred']
            start = ent['start']
            end = ent['end']
            root_node = ent['root_node']
            charstart = ent['char_start']
            charend = ent['char_end']


            if start < end :
                ent_span = Span(doc, start, end, label=ent_type)
            else:
                ent_span = Span(doc, end, start, label=ent_type)

            ent_span._.ENT_type = ent_type
            ent_span._.ENT_root_node = root_node
            ent_span._.ENT_start = start
            ent_span._.ENT_end = end
            ent_span._.ENT_charstart = charstart
            ent_span._.ENT_charend = charend
            ent_span._.ENT_relations = []

            return ent_span
        
        list_ent = []
        test = []
        for i, f in enumerate(filtered_facts):
            
            f['fact']["id"] = f"{i+1}_{f['fact']['pred']}"

            source_span = createEntSpan(f['ner'][0])

            target_span = createEntSpan(f['ner'][1])

            # print(source_span)
            # print(source_span in test)

            # print(target_span)
            # print(target_span in test)

            test.append(source_span)
            test.append(target_span)

            list_ent.append(
                (source_span, target_span)
            )

        for i, f in enumerate(filtered_facts):

            rel = {
                "id": f['fact']['id'],
                "relation": f['fact']['pred'],
                "score": f['fact']['score'],
                "rule": f['fact']['rule'],
                "anchor": f['fact']['anchortext'],
                "nodes": f['fact']['candidate']['nodes']
                
            }

            list_ent[i][0]._.ENT_relations.append(rel)
            list_ent[i][1]._.ENT_relations.append(rel)
        
        list_ent = [y for x in list_ent for y in x]
        filtered_ents = []

        while True :
            if not list_ent :
                break
            else :
                candidate = list_ent.pop(0)
                identicals = filter(lambda x: x.start_char == candidate.start_char and x.end_char == candidate.end_char, list_ent)
                identicals = list(identicals)
                list_ent = [x for x in list_ent if x not in identicals]
                filtered_ents.append(candidate)

        
        doc.spans['REL'] = filtered_ents


    def set_fact_extensions(self):


        Span.set_extension('ENT_type', default=None, force=True)
        Span.set_extension('ENT_root_node', default=None, force=True)
        Span.set_extension('ENT_start', default=None, force=True)
        Span.set_extension('ENT_end', default=None, force=True)
        Span.set_extension('ENT_charstart', default=None, force=True)
        Span.set_extension('ENT_charend', default=None, force=True)
        Span.set_extension('ENT_relations', default=None, force=True)
        Span.set_extension('ENT_id', default=None, force=True)

        # Span.set_extension('ENT_relation_scores', default=None, force=True)


    def factExtraction(self, doc, thresh=0, fuzzyMatch=False):

        
        self.set_fact_extensions()

        facts = self.factextractor.extractFacts(doc, thresh=thresh, fuzzyMatch=fuzzyMatch)
        # facts = filter(lambda x: self.filterFact(doc, x), facts)
        self.addFactSpans(doc, facts)
        
        # return facts
        # doc.spans['FACTS'] = facts

    def process_corpus(self, corpus: List[str], list_metadata: List[dict] = None, thresh=0, fuzzyMatch=False) -> List[Doc]:
        """
        Process a corpus of documents. The input must be a 
        list of dictionnary, where the textual content of the
        document must be contained in a "content" key. Every
        other key will be add as metadata of the doc.
        """

        corpus = self.prepare_corpus(corpus)
        if list_metadata:
            corpus = self.add_metadata2corpus(corpus, list_metadata)
        # corpus = [self.clean_text(doc) for doc in corpus]
        # corpus = [self.nlp.make_doc(doc) for doc in corpus]
        corpus = list(self.nlp.pipe(corpus))

        # corpus = heideltime_tagging(corpus, list_metadata, self.heideltime_config)
        
        for doc in corpus:
            self.factExtraction(doc, thresh=thresh, fuzzyMatch=fuzzyMatch)

        # corpus = [extract_chunks(doc) for doc in corpus]
        
        return corpus

    def process_doc(self, doc: str, metadata: dict = {}, thresh=0, fuzzyMatch=False) -> Doc:
        """
        Process a document. The input must be a 
        a dictionnary, where the textual content of the
        document must be contained in a "content" key. Every
        other key will be add as metadata of the doc.
        """
        doc = self.prepare_doc(doc)
        if metadata:
            doc = self.add_metadata2doc(doc, metadata)
        # doc = self.clean_text(doc)
        # doc = self.nlp.make_doc(doc)
        doc = self.nlp(doc)
        
        # doc = heideltime_tagging(doc, metadata, self.heideltime_config)
        
        self.factExtraction(doc, thresh=thresh, fuzzyMatch=fuzzyMatch)
        # doc = extract_chunks(doc)

        return doc

    def save2disk(self, data: List[Doc], savepath) -> None:
        """
        Saves collection of docs on the disk.
        """
        if not isinstance(data, list):
            docs = [data]
        else:
            docs = data

        docbin = DocBin(docs=docs, store_user_data=True)

        docbin.to_disk(savepath)

        
    def load_from_disk(self, loadpath) -> Doc:
        """
        Loads document from the disk in the .spacy format
        """

        docbin = DocBin().from_disk(loadpath)
        set_heideltime_extension()
        docs = list(docbin.get_docs(self.nlp.vocab))
        return docs

