from benchmark.scripts.predicate import Predicate
from benchmark.models.treernn.domain import p2id

class GaugeQuery:
    ''' age & location & (n1 & n2 & n3 ... & nk) & (p1 | p2 | ... pm) '''

    @staticmethod
    def copy(obj):
        g = GaugeQuery()
        g.age_predicates = [p for p in obj.age_predicates]
        g.interest_predicates = [p for p in obj.interest_predicates]
        g.neginterest_predicates = [p for p in obj.neginterest_predicates]
        g.archetype_predicates = [p for p in obj.archetype_predicates]
        g.negarchetype_predicates = [p for p in obj.negarchetype_predicates]
        g.language_predicates = [p for p in obj.language_predicates]
        g.neglangauge_predicates = [p for p in obj.neglangauge_predicates]
        g.gender_predicates = [p for p in obj.gender_predicates]
        g.location_predicates = [p for p in obj.location_predicates]
        g.loctype_predicates = [p for p in obj.loctype_predicates]
        g.exp_array = []

        return g

    def __init__(self):

        super(GaugeQuery, self).__init__()
        self.age_predicates = []
        self.interest_predicates = []
        self.neginterest_predicates = []
        self.archetype_predicates = []
        self.negarchetype_predicates = []
        self.language_predicates = []
        self.neglangauge_predicates = []
        self.gender_predicates = []
        self.location_predicates = []
        self.loctype_predicates = []

        self.exp_array = []

    def __len__(self):
        s = 0
        for k, v in vars(self).items():
            if k.endswith("predicates"):
                s+=len(v)
        return s
    
    @staticmethod
    def from_string(string):
        q = string.split(' ')
        query = GaugeQuery()
        for term in q:
            if term in ['(',')', '&', '|']:
                continue
            p = Predicate.from_string(term)
            if p.col == 'age':
                query.add_age_predicate(p)
            elif p.col == 'interest' and p.op == '==':
                query.add_interest_predicate(p)
            elif p.col == 'interest' and p.op == '!=':
                query.add_neginterest_predicate(p)
            elif p.col == 'archetype' and p.op == '==':
                query.add_archetype_predicate(p)
            elif p.col == 'archetype' and p.op == '!=':
                query.add_negarchetype_predicate(p)
            elif p.col == 'language' and p.op == '==':
                query.add_language_predicate(p)
            elif p.col == 'language' and p.op == '!=':
                query.add_neglanguage_predicate(p)
            elif p.col == 'gender':
                query.add_gender_predicate(p)
            elif p.col == 'location':
                query.add_location_predicate(p)
            elif p.col == 'loc_type':
                query.add_loctype_predicate(p)
        return query        
                
                
    def add_age_predicate(self, age_predicate):
        self.age_predicates.append(age_predicate)

    def add_interest_predicate(self, interest_predicate):
        self.interest_predicates.append(interest_predicate)

    def add_neginterest_predicate(self, neginterest_predicate):
        self.neginterest_predicates.append(neginterest_predicate)

    def add_archetype_predicate(self, archetype_predicate):
        self.archetype_predicates.append(archetype_predicate)

    def add_negarchetype_predicate(self, negarchetype_predicate):
        self.negarchetype_predicates.append(negarchetype_predicate)

    def add_language_predicate(self, language_predicate):
        self.language_predicates.append(language_predicate)

    def add_neglanguage_predicate(self, neglanguage_predicate):
        self.neglangauge_predicates.append(neglanguage_predicate)
    def add_gender_predicate(self, gender_predicate):
        self.gender_predicates.append(gender_predicate)

    def add_location_predicate(self, location_predicate):
        self.location_predicates.append(location_predicate)

    def add_loctype_predicate(self, loctype_predicate):
        self.loctype_predicates.append(loctype_predicate)

    def fill_predicates(self, predicates, op):
        
        if len(predicates) == 0:
            return
        if len(predicates) == 1:
            id_ = p2id[predicates[0]]
            self.exp_array.append(id_)
            return
        self.exp_array.append('(')
        for p in predicates:
            id_ = p2id[p]
            self.exp_array.append(id_)
            self.exp_array.append(op)
        self.exp_array.pop()
        self.exp_array.append(')')
    
    def build_exp_array(self):
        
        self.fill_predicates(self.age_predicates, '&')
        if self.gender_predicates: self.exp_array.append('&')
        self.fill_predicates(self.gender_predicates, '|')
        if self.loctype_predicates: self.exp_array.append('&')
        self.fill_predicates(self.loctype_predicates, '|')
        if self.interest_predicates: self.exp_array.append('&')
        self.fill_predicates(self.interest_predicates, '|')
        if self.neginterest_predicates: self.exp_array.append('&')
        self.fill_predicates(self.neginterest_predicates, '&')
        if self.archetype_predicates: self.exp_array.append('&')
        self.fill_predicates(self.archetype_predicates, '|')
        if self.negarchetype_predicates: self.exp_array.append('&')
        self.fill_predicates(self.negarchetype_predicates, '&')
        if self.language_predicates: self.exp_array.append('&')
        self.fill_predicates(self.language_predicates, '|')
        if self.neglangauge_predicates: self.exp_array.append('&')
        self.fill_predicates(self.neglangauge_predicates, '&')        
        if self.location_predicates: self.exp_array.append('&')            
        self.fill_predicates(self.location_predicates, '|')

                      

