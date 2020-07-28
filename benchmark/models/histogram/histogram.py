import math
import json
from itertools import combinations
import numpy as np
######################################################################
##############################query###################################
from benchmark.scripts.query import GaugeQuery

class BigramEstimator:

    def __init__(self, prob_dict):
        self.prob_dict = prob_dict

    def bi_estimate_or(self, attr_list, max_depth=20):
        '''(i_1 or i_2 or ... or i_K)'''
        assert isinstance(attr_list, list)
        sign, prob = -1, 0
        depth = min(max_depth, len(attr_list))
        for i in range(1, depth+1):
            sign *=-1
            for tpl in combinations(attr_list, i):
                a = self.bi_estimate_and(list(tpl))
                prob += sign * a
        return prob

    def bi_estimate_and(self, attr_list):
        '''(i_1 and i_2 and ... and i_K)'''
        assert isinstance(attr_list, list)
        N = len(attr_list)
        if N == 2:
            x,y = attr_list
            pair_name = '{},{}'.format(x,y)
            return self.prob_dict.get(pair_name, 0)
        if N == 1:
            return self.prob_dict.get(attr_list[0], 0)
    
        max_prob = -1
        for i in range(N):
            given_attr, prob = attr_list[i], 1
            denom = self.prob_dict.get(given_attr,0)**(N-2)
            if denom == 0: continue
            for j in range(N):
                if j == i: continue
                pair_name = '{},{}'.format(attr_list[j], given_attr)
                temp = self.prob_dict.get(pair_name, 0)
                prob *= temp
            prob /= denom
            if prob > max_prob: max_prob = prob
        return max(max_prob,0)

    def bi_estimate_neg_and(self, attr_list):
        '''!a1 and !a2 and !a3 ... and !ak'''
        assert isinstance(attr_list, list)
        return 1 - self.bi_estimate_or(attr_list)


    def bi_estimate_or_withone_and(self, attr_list, neg_attr):
        '''(i_1 or i_2 or ... or i_K) and !n_1'''
        assert isinstance(attr_list, list)
        assert isinstance(neg_attr, str)
        p_n = self.prob_dict.get(neg_attr, 0)
        pr = (1- p_n) - self.bi_estimate_neg_and(list(set(attr_list+[neg_attr])))
        return pr

    def estimate(self, attr_list, neg_attr_list=None):
        '''(i_1 or i_2 or ... or i_K) and !n_1 and !n_2 ... and !n_T'''

        if len(neg_attr_list) + len(attr_list) == 0:
            return 1
        elif len(neg_attr_list)==0 and len(attr_list)>0:
            return self.bi_estimate_or(attr_list)
        elif len(attr_list)==0 and len(neg_attr_list)>0:
            return self.bi_estimate_neg_and(neg_attr_list)

        #assume there are at least one element in both arrays
        query =["pos_attributes"] + neg_attr_list
        min_prob = float("inf")
        history = []
        T = len(query)
        for i in range(T):
            given_attr, prob = query[i], 1
            pos_condition = True if given_attr == "pos_attributes" else False
            if pos_condition:
                denom = self.bi_estimate_or(attr_list)
                denom = denom**(T-2)
            else:
                denom = (1 - self.prob_dict.get(given_attr,0))**(T-2) 

            if denom == 0: continue
            for j in range(T):
                if i == j: continue
                pair = [query[j], given_attr]
                is_item_positive = pair[0] == "pos_attributes"
                if not is_item_positive and not pos_condition:
                    prob *= self.bi_estimate_neg_and(pair)
                elif is_item_positive:
                    prob *= self.bi_estimate_or_withone_and(attr_list, pair[1])
                elif pos_condition:
                    prob *= self.bi_estimate_or_withone_and(attr_list, pair[0])
            
            prob /= denom
            history.append(prob)
            if prob < min_prob: min_prob = prob
        
        #median = np.median(history)
        mean = np.mean(history)
        # print(history)
        return mean


class UnigramEstimator:

    def __init__(self, prob_dict):
        self.prob_dict = prob_dict


    def uni_estimate_and(self, attr_list):
        '''a1 and a2 and a3 ... and ak'''
        res_prob = 1
        for attr in attr_list:
            temp = self.prob_dict.get(attr, 0)
            res_prob *= temp
        return res_prob

    def uni_estimate_neg_and(self, attr_list):
        '''!a1 and !a2 and !a3 ... and !ak'''
        res_prob = 1
        for attr in attr_list:
            temp = self.prob_dict.get(attr, 0)
            res_prob *= (1-temp)
        return res_prob

    def uni_estimate_or(self, attr_list):
        '''a1 or a2 or ... or ak'''
        return 1 - self.uni_estimate_neg_and(attr_list)


    def estimate(self, attr_list, neg_attr_list):
        '''(a1 or a2 or a3 or...or ak) and !n1 and !n2 and !n3...'''
        if len(attr_list) + len(neg_attr_list) ==0:
            return 1
        elif len(attr_list) == 0:
            return self.uni_estimate_neg_and(neg_attr_list)
        elif len(neg_attr_list) == 0:
            return self.uni_estimate_or(attr_list)
        return self.uni_estimate_or(attr_list) * self.uni_estimate_neg_and(neg_attr_list)


class HistogramEstimator:

    def __init__(self, prior_dict_path):
        
        with open(prior_dict_path, 'r') as handle:
            self.data = json.load(handle)
            self.data = {k.lower():b for k,b in self.data.items()}
            for k,v in self.data.items():
                if isinstance(v, dict):
                    self.data[k] = {k1.lower():v1 for k1,v1 in v.items()}

    def check_gender(self, gender):
        assert gender in ['male', 'female', 'both']

#    def check_loc_type(self, loc_type):
#        assert loc_type in ['work', 'live', 'current']

    def estimate_age_prob_given_gender(self, age_range, gender):
        '''
        age_range[range] -- range of user аge, like range(10,56)
        age_prob[dict] -- like age_prob = {'18':0.01,'21':0.03,...}
        '''
        age_prob = self.data["{}_{}_prob".format(gender, "age")]
        return sum([age_prob[str(i)] if str(i) in age_prob  else 0 for i in age_range])

    def estimate_loc_prob_given_gender(self, locations, gender, loc_type):
        if len(locations) == 0:
            return 1
        loc_prob = self.data["{}_{}loc_prob".format(gender, loc_type)]
        return sum([loc_prob[loc] if loc in loc_prob  else 0 for loc in locations])


    def estimate(self, query):


        interests = [i.val for i in query.interest_predicates]
        neginterests = [i.val for i in query.neginterest_predicates]
        locations = [i.val for i in query.location_predicates]
        languages = [i.val for i in query.language_predicates]
        neglanguages = [i.val for i in query.neglangauge_predicates]
        archetypes = [i.val for i in query.archetype_predicates]
        negarchetypes = [i.val for i in query.negarchetype_predicates]
        loc_type = query.loctype_predicates[0].val

        if len(query.age_predicates) == 1:
            i = query.age_predicates[0]
            if i.op == '==':
                age_range = [i.val]
            elif i.op == '!=':
                age_range = [j for j in range(10,100) if j!=int(i.val)]
        else:
            start, end = query.age_predicates[:2]
            if start.op == '>=':
                start = int(start.val)
            elif start.op == '>':
                start = int(start.val) +1

            if end.op == '<':
                end = int(end.val)
            elif end.op == '<=':
                end = int(end.val) +1
            age_range = [j for j in range(start,end)]
        
        gender = []
        for g in query.gender_predicates:
            if g.op == '==':
                gender.append(g.val)
            elif g.op == '!=' and g.val=='male':
                gender.append('female')
            else:
                gender.append('male')

        final_result = 0
        for g in gender:

            interest_estimator = UnigramEstimator(self.data["{}_{}_prob".format(g, "interest")])
            archetype_estimator = UnigramEstimator(self.data["{}_{}_prob".format(g, "archetype")])
            language_estimator = UnigramEstimator(self.data["{}_{}_prob".format(g, "language")])


            p_int = interest_estimator.estimate(interests, neginterests)
            p_arch = archetype_estimator.estimate(archetypes, negarchetypes)
            p_lng = language_estimator.estimate(languages, neglanguages)

            p_age = self.estimate_age_prob_given_gender(age_range, g)
            p_loc = self.estimate_loc_prob_given_gender(locations, g, loc_type)

            p =  p_int * p_arch * p_lng * p_age * p_loc

            final_result +=  p * self.data["{}_{}_count".format(g, 'interest')]

        return math.ceil(final_result)

