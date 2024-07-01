# coding: utf-8
# Author : Iana Atanassova
# 2016-2019

import re
from typing import List

class StringTools:
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    @staticmethod
    def cleanSpaces(s:str) -> str:
        """
        A method that removes extra blank spaces from a string.
        :param s: String to clean
        :type s: str
        :return: Clean str
        :r_type: str
        """

        blanc = "(\\s){1,9}"
        return re.sub(blanc, " ", s.strip())

    @staticmethod
    def locations_of_substring(string: str, substring:str) -> List[int]:
        """
        Returns a list of locations of a substring.
        
        :param string: String to process
        :type string: str
        :param substring: Substring to find
        :type substring: str
        :return: List of location of substring in string
        :rtype: List[int]
        """

        substring_length = len(substring)
        def recurse(locations_found, start):
            location = string.find(substring, start)
            if location != -1:
                return recurse(locations_found + [location], location+substring_length)
            else:
                return locations_found

        return recurse([], 0)


class AbstractSegRule:
    # Constructor
    def __init__(self, name, expr):
        if (name == ""):
            print("ERROR : Empty name in rule definition.")
        if (expr == ""):
            print("ERROR : Empty name in rule definition.")
        self.name = name
        self.expression = expr


class SegFDPRule (AbstractSegRule):

    markupStart = "{##"
    markupEnd = "##}"

    
    @staticmethod
    def convertFDPToMarkup(t:str, fdpRules:List[AbstractSegRule]) -> str:
        """
        A method that replaces all FDP symbols with temporary markup.

        :param t: string to clean
        :type t: str
        :param fdpRules: List of rules to apply
        :type fdpRules: List[AbstractSegRule]
        :return: The cleaned string
        :rtype: str
        """
        # loop over fdp symbols
        for f in fdpRules:
            t = t.replace(f.expression, SegFDPRule.markupStart + f.name + SegFDPRule.markupEnd)
        return t

    @staticmethod
    def convertMarkupToFDP(t:str, fdpRules:List[AbstractSegRule]) -> str:
        """
        A method that replaces all FDP symbols with temporary markup.

        :param t: string to clean
        :type t: str
        :param fdpRules: List of rules to apply
        :type fdpRules: List[AbstractSegRule]
        :return: The cleaned string
        :rtype: str
        """
        # loop over FDP symbols
        for f in fdpRules:
            t = t.replace(SegFDPRule.markupStart + f.name + SegFDPRule.markupEnd, f.expression)
        return t

    def getSplitPositions(self, segment: str) -> List[int]:
        """
        Apply the rule on a text segment and return the result.

        :param segment: String where to search for substrings
        :type segment: str
        :return: List of positions of substrings
        :rtype: List[int]
        """
        return StringTools.locations_of_substring(segment, self.expression)

class SegNFDPRule (AbstractSegRule):

    def apply(self, segment, fdpRules):
        """
        The NFDP expression is identified, then the FDP symbols in it are replaced by temporary mark-up in the form: markupStart + symbol name + markUpend.

        :param segment: _description_
        :type segment: _type_
        :param fdpRules: _description_
        :type fdpRules: _type_
        :return: _description_
        :rtype: _type_
        """
        start = 0
        subSeg = ""

        for m in re.finditer(self.expression, segment):
            s = m.start()
            e = m.end()
            # text before match
            txtAvant = segment[start:s]
            subSeg += txtAvant
            # matched text
            segmatch = m.group(0)
            # replace FDP symbols in matched text by temp markup
            segmatch = SegFDPRule.convertFDPToMarkup(segmatch, fdpRules)

            subSeg += segmatch
            start = e

        # Append the rest of the string after the last match
        if (subSeg != ""):
            segment = subSeg + segment[start:len(segment)]
        return segment


class Segmenter:
    # NFDPRules = []
    # FDPRules = []

    processBibRefs = False
    blockElements = [ "boxed-text", "chem-struct-wrap", "fig",
			"graphic", "media", "preformat", "supplementary-material",
			"table-wrap", "disp-formula", "disp-formula-group" ]


    def applyNFDPRules(self, seg: str) -> str:
        """
        Identification of NFDP expressions.
        Each NFDP expression is identified, then the FDP symbols in it are
        replaced by temporary mark-up in the form: markupStart + symbol name +
        markUpend.

        :param seg: Segment to process
        :type seg: str
        :return: Processed segment
        :rtype: str
        """
        for n in self.NFDPRules:
            seg = n.apply(seg, self.FDPRules)
        return seg

    def __init__(self, rules:str) -> None:
        """
        Constructor : Load rules
        Constructor taking as parameter the names of the rules file to be loaded.

        :param rules: Language in which to select the rules
        :type rules: str
        """
        self.NFDPRules = []
        self.FDPRules = []

        # default rules : FDP
        self.FDPRules.append(SegFDPRule("point", "."))
        self.FDPRules.append(SegFDPRule("excMarc", "!"))
        self.FDPRules.append(SegFDPRule("qMarc", "?"))
        # self.FDPRules.append(SegFDPRule("susp", "..."))

        # default rules : NFDP
#         NICOLAS: ajouter de regle pour ne pas segmenter phrase si pas suivie de majuscule, post-scriptum et nota-bene
        self.NFDPRules.append(SegNFDPRule("nonsent", r"\w\. ?([a-z]|,)"))
        self.NFDPRules.append(SegNFDPRule("nonsent2", r" \."))

        self.NFDPRules.append(SegNFDPRule("postscriptum", "[Pp]\.\-?[Ss]\."))
        self.NFDPRules.append(SegNFDPRule("notabene", "(N\.\-?B\.|Nota.)"))

        #  Enumeration lines, e.g. "* 4. Texte"        
        self.NFDPRules.append(SegNFDPRule("enum1", "^(\s)*[\*\-\>]\s[0-9]+(\.)\s[A-Z][a-z]+"))
        #  Roman numerals (only when followed by a point and in the beginning of a line)
        #  Only 1-100, to limit the noise.
        self.NFDPRules.append(SegNFDPRule("roman-numerals", "^(\s)*(([IVX]){1,5})(\.)"))
        #  Points in URLs
        self.NFDPRules.append(SegNFDPRule("url1", "(https?://)?www\.[^\s]{5,}\w"))
        self.NFDPRules.append(SegNFDPRule("url2", "https?://[^\s]{5,}\w"))
        self.NFDPRules.append(SegNFDPRule("url3", "ftp://[^\s]{5,}\w"))
        #  Text in brackets
        self.NFDPRules.append(SegNFDPRule("brackets", "\(([^\)]+)\)"))
        #  Quoted text, followed by non-uppercase letter
        self.NFDPRules.append(SegNFDPRule("quotes1", "«([^»]+)»"))
        self.NFDPRules.append(SegNFDPRule("quotes2", "\"([^\"]+)\""))
        self.NFDPRules.append(SegNFDPRule("quotes3", "‘([^’]+)’"))
        self.NFDPRules.append(SegNFDPRule("quotes4", "“([^”]+)”"))
        #  "?" followed by a comma
        self.NFDPRules.append(SegNFDPRule("q_comma", "\?,( )"))
        #  Uppercase abbreviations, e.g. A.B.CD:42.E5.F., E.-U.
        self.NFDPRules.append(SegNFDPRule("uc_abbr", "[A-Z0-9É]+([\.:\-][A-Z0-9É]+)+(\.)?"))
        #  Single lowercase letter abbreviations, p.
        self.NFDPRules.append(SegNFDPRule("lc_abbr", "(\s)[a-z]\."))
        #  Single uppercase letter abbreviations, C., T.
        self.NFDPRules.append(SegNFDPRule("uc_abbr2", "(^|\s)[A-Z]\."))
        #  Numbers
        self.NFDPRules.append(SegNFDPRule("num", "[0-9]+\.[0-9]+"))
        #  General abbreviations in Roman languages
        self.NFDPRules.append(SegNFDPRule("PhD", "(Ph)\.([Dd])\."))
        self.NFDPRules.append(SegNFDPRule("cf", "c\.?f\."))
        self.NFDPRules.append(SegNFDPRule("pp", "pp\."))
        #  etc. followed by a lowercase letter
        self.NFDPRules.append(SegNFDPRule("etc", "etc\.(,)? [a-z]"))
        #  Ellipsis followed by a lowercase letter
        self.NFDPRules.append(SegNFDPRule("ellipse", "\.\.\.(,)? [a-z]"))
        #  et al. followed by a non-capitalcase letter
        self.NFDPRules.append(SegNFDPRule("etAl", "et al\.(,)? [^A-Z]"))

        # ENGLISH
        if ("en" in rules):
            # Mr. Ms. Mrs. -->
            self.NFDPRules.append(SegNFDPRule("en-M", "(\s)M(r|rs|s)\."))
            # Abbreviations: Ibid./ibid., eds./ed., Inc., art. -->
            self.NFDPRules.append(SegNFDPRule("en-ibid", "(I|i)(bid)\."))
            self.NFDPRules.append(SegNFDPRule("en-eds", "(\s)(eds|ed)\."))
            self.NFDPRules.append(SegNFDPRule("en-inc", "(\s)([Ii]nc)\."))
            self.NFDPRules.append(SegNFDPRule("en-art", "(\s)(art)\."))
            self.NFDPRules.append(SegNFDPRule("en-e_g", "e\.g\."))
            self.NFDPRules.append(SegNFDPRule("en-spp", "spp\."))
            self.NFDPRules.append(SegNFDPRule("en-ft", "ft\."))
            self.NFDPRules.append(SegNFDPRule("en-ft", "(vs\.)|(vv\.)"))
            # Abbreviations: ad., Add., Addit., Addr., adj., adjs., Adm., admin., advs., advt., advts. -->
            self.NFDPRules.append(SegNFDPRule("en-ad", "(\s)ad(d|j|m)?(i|r|s|in|vs|vt|vts)?(t)?(\.)"))
            # Abbreviations: yd., Yearbk., Yng., Yorks., Yorksh., Yr., Yrs. -->
            self.NFDPRules.append(SegNFDPRule("en-y", "(\s)(Y|y)(d|earbk|ng|orks|orksh|r|rs)?(\.)"))
            # Abbreviations: University avec virgule U. Durham L., U. Cal., U. Aberdeen L., W. Yorks. AS, -->
            self.NFDPRules.append(SegNFDPRule("en-university", "(\s)(U|W)(\.)(\s)?(([A-Z][a-z]+(\.)?(\s)?([A-Z]+((\.)?))?))?,"))
            # Abbreviations: c.a., c.o., c.o.d. -->
            self.NFDPRules.append(SegNFDPRule("en-One_lettre_abrev", "(\s)([a-z](\.)){1,3}"))

        # FRENCH
        if ("fr" in rules):            
            # M. m. Mme. Mlle. --> NICOLAS: ajout M apres lle| et espace optionnel
            self.NFDPRules.append(SegNFDPRule("fr-M", "( )[Mm](me|lle|M)? ?\."))
            # Abbreviations: Ibid./ibid., éds./ed., Inc., art., c-à-d, .fr -->
            self.NFDPRules.append(SegNFDPRule("fr-ibid", "(I|i)(bid)\."))
            self.NFDPRules.append(SegNFDPRule("fr-eds", "(\s)(éds|éd|eds|ed)\."))
            self.NFDPRules.append(SegNFDPRule("fr-inc", "(\s)([Ii]nc)\."))
            self.NFDPRules.append(SegNFDPRule("fr-art", "(\s)(art)\."))
            self.NFDPRules.append(SegNFDPRule("fr-c_a_d", "c\.?-à\.?-d\.?"))
            self.NFDPRules.append(SegNFDPRule("fr-fr", "\.fr"))
            # par ex. -->
            self.NFDPRules.append(SegNFDPRule("fr-par_ex", "par(\s)ex(\.)"))
            # Abbreviations: et al., et approx., et coll., et s., et suiv., et sup. -->
            self.NFDPRules.append(SegNFDPRule("fr-et_qqc", "(\s)et(\s)(al|approx|coll|s|suiv|sup)(\.)"))
            # Abbreviations: op., op. cit., op. cit., op. laud., op. posth. -->
            self.NFDPRules.append(SegNFDPRule("fr-et_qqc", "(\s)op(\.)(\s)?(cit|laud|posth)?(\.)?"))
            # Abbreviations: ouvr., ouvr. cit., ouvr. cité,  -->
            self.NFDPRules.append(SegNFDPRule("fr-ouvr_qqc", "(\s)ouvr(\.)(\s)?(cit(\.)|cité)?"))
            # Abbreviations: pag., pagin., plaq. ...  -->
            self.NFDPRules.append(SegNFDPRule("fr-pag_qqc", "(\s)(prélim|pap|pag|pagin|par(\s)art|parch|pass(\.)(\s)cité|plaq|port)(\.)"))
            # Abbreviations: reimpression reproduction rousseur ...  -->
            self.NFDPRules.append(SegNFDPRule("fr-reimpr", "(\s)(r(e|é)imp(R)?|rel|reprod|rouss)(\.)"))
            # Abbreviations: sc., sect.  -->
            self.NFDPRules.append(SegNFDPRule("fr-section", "(\s)(sc|sect)(\.)"))
            # Abbreviations: tome., trad., trans., typ.   -->
            self.NFDPRules.append(SegNFDPRule("fr-tbiblio", "(\s)(t|tom|trad|trans(cr)?|typ)(\.)"))
            # Abbreviations: LL. MM. RR., R. P. J., S. Exc., S. Ém.,etc   -->
            self.NFDPRules.append(SegNFDPRule("fr-civilite", "(S(\.)(S|(St)|H|A|E|M|Em|Ém|Exc|M(\.))(\.)((I|R|E|S)(\.)?)?)|St(\.)|S(\.)-Lt|LL(\.)(\s)?(AA|MM|EE|ÉÉm|EExc|GG)(\.)(\s)?(II(\.)|RR(\.))?|F(\.)|R(\.)P(\.)(J.)?"))
            # Abbreviations: B. Arch., B.Com.   -->
            self.NFDPRules.append(SegNFDPRule("fr-bachelor", "(\s)B(\.)(\s)?(Arch|Com|Éd|Ing|Int|Mus|Pharm|Ps|Sc|Serv|Th|Urb)(\.)((\s)(A|Inf|Soc)(\.))?"))
            # Abbreviations: Licence   -->
            self.NFDPRules.append(SegNFDPRule("fr-licence", "LL?(\.)(\s)?(B|D|L|M|Ph|Arch|Com|Ing|Int|Mus|Pharm|Ps|Sc|Serv|Th|Urb|Éd)(\.)((\s)?(L|A|Inf|Soc|compt)(\.))?|LL?(\.)(\s)ès(\s)(L|A|Inf|Soc)?(\.)"))

        # JATS
        if ("jats" in rules):
            # e.g.  <xref ref-type="bibr" rid="pctr-0010008-b006", "6</xref> -->
            self.NFDPRules.append(SegNFDPRule("jats-1", "&lt;xref.*?&gt;.*?&lt;\/xref&gt;"))
            # e.g. <?Pub Caret?> -->
            self.NFDPRules.append(SegNFDPRule("jats-2", "&lt;\?.*?\?&gt;"))

        # PV : pour traiter les point-virgules comme FDP (applications specifiques)
        if ("pv" in rules):
            self.FDPRules.append(SegFDPRule("pv-semicolon", ";"))

    def getSentences(self, text: str) -> List[str]:
        """
        A method that segments a given string into sentences.

        :param text: Text to process
        :type text: str
        :return: Segmented text
        :rtype: List[str]
        """
        
        def clean_segment(segment):
            """
            NICOLAS:
            Method to be used with the filter function.
            Returns false if given segment must be removed, else 
            returns True. 
            Segments to be removed are either:
            * less than 5 characters
            * contains exclusively non-alphanumeric character
            """
            nonalpha_re = re.compile(r'^[^a-zA-Z\s:]*$')

            if len(segment) <= 4:
                return False
            elif re.search(nonalpha_re, segment):
                return False
            else:
                return True
        
        nonsentcleaner_re = re.compile(r'(\w)\. ?(,? ?[a-z])')
        
        text = StringTools.cleanSpaces(text)
        text = self.applyNFDPRules(text)
        splitPositions = []
        for f in self.FDPRules:
            splitPositions.extend(f.getSplitPositions(text))

        segments = []
        splitPositions.sort()
        start = 0
        for pos in splitPositions:
            segments.append(text[start:pos+1].strip())
            start = pos+1
        segments.append(text[start:len(text)])

        # Replace all temporary markup with corresponding FDP symbols.
        for i in range(len(segments)):
            s = SegFDPRule.convertMarkupToFDP(segments[i], self.FDPRules)
            # Clear multiple marks
            s = re.sub("\\.\\?!", ".", s)
            s = re.sub("\\?!", "?", s)

            segments[i] = s

#       NICOLAS : retire point entre deux non phrases, 
#       ex: Depuis vingt-sept mois, nous. ne l'avions pas vu.
#       ex: Depuis vingt-sept mois, nous. ,ne l'avions pas vu.

        segments = [re.sub(nonsentcleaner_re, r'\1 \2' ,s) for s in segments]
     
#       NICOLAS: function to clean bad segments 
        segments = list(filter(clean_segment, segments))
        # for i, s in enumerate(segments):
        #     print(i, s)
        #     print()
        
        
        # try:
        #     segments.remove("")
        #     segments.remove(".")
        #     segments.remove(" .")
        #     segments.remove("  .")
        #     segments.remove(",")
        # except:
        #     pass

        return segments


# NICOLAS: ajouter ligne ci-dessous pour distinguer entre programme principal et
# import module
if __name__ == '__main__':
    ## TEST

    # seg = Segmenter("en")
    # text = "Analysis of muscle suspensions from young Pax7?/? mice revealed a " \
    # + "significantly increased number of hematopoietic progenitors and adipogenic cells " \
    # + " (Seale et al. 2000). We also observed altered proportions of CD45- and Sca1-expressing" \
    # + " cells in uninjured and regenerating muscle (see Figure 1A). The putative stem cell " \
    # + "subfraction coexpressing CD45 and Sca1 may have been exhausted prematurely during " \
    # + "postnatal Pax7??/? muscle development. It is also conceivable that a reduced proportion " \
    # + "of stem cells in the Pax7??/? CD45+:Sca1+ muscle fractions was not detected in our assay " \
    # + "due to a low efficiency of retroviral transduction (approximately 10% of surviving " \
    # + "CD45+:Sca1+ cells with GFP virus). The identification of additional markers expressed " \
    # + "by adult muscle-derived stem cells is required to more thoroughly explore these issues."
    #

    seg = Segmenter('fr')
    text = "P.S. -Nos musiciens, il faut le reconnaître, viennent nombreux aux répétitions et préparent déjà le prochain concert qui sera donné vers la mi-février."
    sents = seg.getSentences(text)

    for i, s in enumerate(sents):
        print(i, s)
        print()
    # print(len(sents))
