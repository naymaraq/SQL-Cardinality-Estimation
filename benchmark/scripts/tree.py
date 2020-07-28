from operator import attrgetter


def priority(op):
    if op == '&': return 1
    if op == '|': return 0
    return -1


def toPostfix(exp_array):
    stack = []
    output = []
    for i in range(len(exp_array)):
        term = exp_array[i]
        if term == '(':
            stack.append('(')
        elif term == ')':
            top = stack[-1]
            while (top != '('):
                output.append(top)
                stack.pop()
                top = stack[-1]
            stack.pop()
        elif term in ['&', '|']:
            while (len(stack) != 0):
                top = stack[-1]
                if top == '(' or priority(top) < priority(term): break
                output.append(top)
                stack.pop()
            stack.append(term)
        else:
            output.append(term)
    while (len(stack) != 0):
        top = stack[-1]
        output.append(top)
        stack.pop()

    return output


def exp2Prefix(exp):
    re_exp = list(reversed(exp))
    re_exp = [')' if x == '(' else '(' if x == ')' else x for x in re_exp]
    re_exp_posfix = toPostfix(re_exp)
    prefix = list(reversed(re_exp_posfix))
    return prefix


class ExpTree:
    # Constructor to create a node 
    def __init__(self, value):
        self.value = value
        self.children = []
        self.num_children = 0
        self.parent = None

    def set_idx(self, i):
        self.idx = i
        for child in self.children:
            i = i + 1
            i = child.set_idx(i)
        return i

    def traverse_by_values(self):

        stack_values = []
        stack = [self]
        while len(stack) > 0:
            node = stack.pop()
            stack_values.append((node.idx, node.value))
            if len(node.children):
                stack.extend(node.children[::-1])
        stack_values = [item[1] for item in stack_values]
        return stack_values

    def __repr__(self):
        return self.__str__()

    def add_child(self, children):

        if isinstance(children, list):
            for ch in children:
                ch.parent = self
            self.children.extend(children)
            self.num_children += len(children)
        else:
            children.parent = self
            self.num_children += 1
            self.children.append(children)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __str__(self):
        tree = '%s\n' % self.value
        children = list(self.children)
        for i, c in enumerate(sorted(children, key=attrgetter('value')), start=1):
            nodes = c.__str__().split('\n')
            for j, node in enumerate(nodes, start=1):
                if i != len(children) and j == 1:
                    tree += '>--- %s\n' % node
                elif i != len(children) and j > 1:
                    tree += '>    %s\n' % node
                elif i == len(children) and j == 1:
                    tree += '>--- %s\n' % node
                else:
                    tree += '    %s\n' % node
        return tree.strip()


def isOperator(c):
    if c in ['&', '|']:
        return True
    return False


def isOperand(c):
    return c.startswith('x')


def constructTree(pfix):
    stack = []
    for i in range(len(pfix) - 1, -1, -1):

        if isOperand(pfix[i]):
            stack.append(ExpTree(pfix[i]))

        else:

            v1 = stack.pop()
            v2 = stack.pop()

            nn = ExpTree(pfix[i])
            if v1.value == pfix[i]:
                nn.add_child(v1.children)
            else:
                nn.add_child(v1)

            if v2.value == pfix[i]:
                nn.add_child(v2.children)
            else:
                nn.add_child(v2)

            stack.append(nn)

    return stack[0]


END = '$'


def serialize(root, file):
    file.write(root.value + ' ')
    for ch in root.children:
        serialize(ch, file)
    file.write('%s ' % END)


def deserialize(fp):
    data = fp.read().rstrip().split(' ')
    stack = []
    for item in data:
        if item != END:
            node = ExpTree(item)
            stack.append(node)
        else:
            if len(stack) == 1:
                break

            n_ch = stack.pop()
            stack[-1].add_child(n_ch)

    return stack[0]
