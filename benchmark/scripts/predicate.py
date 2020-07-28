class Predicate:

    def __init__(self, col, op, val):

        self.col = col
        self.op = op
        self.val = val

    def __str__(self):
        return "{}{}{}".format(self.col, self.op, self.val)
    
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_string(sp):
        for op in ['==', '!=', '>=', '<=', '>', '<']: #>< must be at the end
            if op in sp:
                col, val = sp.split(op)
                return Predicate(col, op, val)