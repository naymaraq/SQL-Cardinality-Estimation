import numpy as np

from benchmark.scripts.predicate import Predicate
from benchmark.models.treernn.domain import p2id, col_domains

class Indexify: 
    
    def __init__(self, domains, p2id):
        
        self.domains = domains
        self.__p2id = p2id
        self.__id2Predicate = {v:k for k,v in self.__p2id.predicate2id.items() if k!='last'}

        u_cols = []
        u_vals = []
        u_ops = []

        for col in sorted(self.domains):
            u_cols.append(col)

            col_doamin_v = self.domains[col].values
            if col == 'age':
                ages = np.arange(int(col_doamin_v[0]), int(col_doamin_v[1])+15)
                ages = [str(a) for a in ages]  
                u_vals.extend(ages)
            else:
                u_vals.extend(list(col_doamin_v))

        u_cols.append('loc_type')
        u_vals.extend(['live', 'work'])
        u_ops.extend(['==','!=', '>', '>=', '<', '<='])

        self.u_cols = {val: i for i,val in enumerate(sorted(list(set(u_cols))))}
        self.u_vals = {val: i for i,val in enumerate(sorted(list(set(u_vals))))}
        self.u_ops = {val:i for i,val in enumerate(sorted(list(set(u_ops))))}
        
    def reset_indecies(self):
        self.__id2Predicate = {v:k for k,v in self.__p2id.predicate2id.items() if k!='last'}
    
    def featurize(self, p_index):

        p = Predicate.from_string(self.__id2Predicate[p_index])
        col_one_hot = np.zeros(len(self.u_cols), dtype=np.float32)
        col_one_hot[self.u_cols[p.col]] = 1.0
        
        op_one_hot = np.zeros(len(self.u_ops), dtype=np.float32)
        op_one_hot[self.u_ops[p.op]] = 1.0

        val_one_hot = np.zeros(len(self.u_vals), dtype=np.float32)
        val_one_hot[self.u_vals[p.val]] = 1.0
        return np.hstack((col_one_hot,op_one_hot,val_one_hot))

index = Indexify(col_domains, p2id)