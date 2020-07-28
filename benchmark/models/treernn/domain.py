import numpy as np
import json

from benchmark.scripts.predicate import Predicate

class Predicate2Id:
    __instance = None
    def __init__(self, path):

        if not Predicate2Id.__instance:
            self.path = path
            try:
                with open(self.path, 'r') as json_file:
                    self.predicate2id = json.load(json_file)
            except:
                self.predicate2id = {}
                self.predicate2id['last'] = 0
            Predicate2Id.__instance = self

    @staticmethod
    def get_instance(path):
        if not Predicate2Id.__instance:
            Predicate2Id(path)
        return Predicate2Id.__instance
    
    def __getitem__(self, predicate):
        repr_ = predicate.__str__()
        if repr_ in self.predicate2id:
            return self.predicate2id[repr_]
        else:
            self.predicate2id[repr_] = 'x%s'%str(self.predicate2id['last'])
            self.predicate2id['last'] +=1
            return self.predicate2id[repr_]
    
    def dump(self):
        with open(self.path, 'w') as json_file:
            json.dump(self.predicate2id, json_file)

    

class Domain:
    def __init__(self, values, col_name, allowed_ops, type):

        self.values = values
        self.col_name = col_name
        self.allowed_ops = allowed_ops
        self.type = type

    def gen_random_value(self):
        if self.type == 'c':
            return np.random.choice(self.values)
        else:
            return np.random.randint(self.values[0], self.values[1])
    
    def gen_random_op(self):
        return np.random.choice(self.allowed_ops)

def read_txt(path):
    with open(path, 'r') as file:
        values = file.read().split('\n')[:-1]
        if len(values[-1])==0:
            values = values[:-1]
    return values


COLUMN_VALUES = "data/column_values/"
age_values = list(map(float, read_txt(COLUMN_VALUES+'age.txt')))
age_domain = Domain(age_values, 'age', ['>','<=','<','>=','==','!='], 'n')

location_values = read_txt(COLUMN_VALUES+'location.txt')
location_values = [x.lower() for x in location_values]
location_domain = Domain(location_values, 'location', ['==','!='], 'c')

language_values = read_txt(COLUMN_VALUES+'language.txt')
language_values = [x.lower() for x in language_values]
language_domain = Domain(language_values, 'language', ['==','!='], 'c')

gender_values = read_txt(COLUMN_VALUES+'gender.txt')
gender_values = [x.lower() for x in gender_values]
gender_domain = Domain(gender_values, 'gender', ['==','!='], 'c')

archetype_values = read_txt(COLUMN_VALUES+'archetype.txt')
archetype_values = [x.lower() for x in archetype_values]
archetype_domain = Domain(archetype_values, 'archetype', ['==','!='], 'c')

interest_values = read_txt(COLUMN_VALUES+'interest.txt')
interest_values = [x.lower() for x in interest_values]
interest_domain = Domain(interest_values, 'interest', ['==','!='], 'c')

p2id = Predicate2Id.get_instance('data/p2id.json')

col_domains = {'age': age_domain,
               'gender': gender_domain,
               'interest': interest_domain,
               'archetype':archetype_domain,
               'language': language_domain,
               'location': location_domain}