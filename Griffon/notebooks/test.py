# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from CoqGym.gallina import GallinaTermParser
from lark import Tree, Token
from pygments.lexers import CoqLexer
from functools import reduce

from typing import List, Tuple, Union

string = '"my super test"'
print(string)
print(string[1:-1])

"'forall (n : nat) (_ : list Digit n), list Digit (Init.Nat.pred n)'"


# %%
class TextPosition():

    def __init__(self, start:int, end:int):
        self.start = start
        self.end = end

class MyNode():

    def __init__(self, children:List,  text_pos : TextPosition) :
        self.children = children
        self.text_pos = text_pos

class TokenStream():

    def __init__(self, tokens : List[str]):
        self.tokens = tokens
        self.length = len(tokens)
        self.index = 0

    def advance(self):
        self.index += 1

    def peek(self, expected_token)->bool:
        return expected_token == self.tokens[self.index]

    def consume(self, expected_token:str)->int:
        assert self.index < self.length,             "tried to consume past the tokens length"

        assert expected_token == self.tokens[self.index],             f"expected {expected_token} but was {self.tokens[self.index]}"
        self.index = self.index + 1
        return self.index - 1

    def try_consume(self, expected_token:str)->int:
        assert self.index < self.length
        if expected_token == self.tokens[self.index]:
            self.index = self.index + 1
            return self.index - 1
        else:
            return self.index


# %%
sexp = '(App (Ind (((Mutind (MPfile (DirPath ((Id Logic) (Id Init) (Id Coq)))) (DirPath ()) (Id eq)) 0) (Instance ()))) ((Ind (((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) (Instance ()))) (App (Const ((Constant (MPfile (DirPath ((Id Nat) (Id Init) (Id Coq)))) (DirPath ()) (Id add)) (Instance ()))) ((Var (Id n)) (Construct ((((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) 1) (Instance ()))))) (Var (Id n))))'
term = '@eq nat (Nat.add n O) n'

term_parser = GallinaTermParser(caching=False)

tree = term_parser.parse(sexp)


lexer = CoqLexer()
lexer.stripall = True
tokens = list(lexer.get_tokens(term))
excluded_tokens = ["@", "(", ")"]
token_list = [token[1] for token in tokens if not (token[1].isspace() or token[1] in excluded_tokens)]

print(tree)


# %%
class MyToken():

    def __init__(self, token:str, id:int) -> None:
        self.token = token
        self.id = id

def add_prefix_preorder_id(node : Tree, start_index:int = 0)->Tuple[Union[Tree,MyToken], int]:
    if isinstance(node, Token):
        return (MyToken(node, start_index), start_index + 1)
    else:
        node.id = start_index
        new_start_index = start_index + 1
        new_children = []
        for child in node.children:
            res = add_prefix_preorder_id(child, new_start_index)
            new_child, new_start_index = res
            new_children.append(new_child)

        node.children = new_children
        return node, new_start_index


def get_token_sequence(de_bruijn_stack:List[str], node:Union[Tree, Token])->List[Tuple[str, int]]:
    if isinstance(node, MyToken):
        return [(node.token, node.id)]
    elif node.data == "constructor_anonymous":
        assert len(node.children) == 0
        return [("_", node.id)]
    elif node.data == "constructor_prop":
        assert len(node.children) == 0
        return [("Prop", node.id)]
    elif node.data == "constructor_set":
        assert len(node.children) == 0
        return [("Set", node.id)]
    elif node.data == "constructor_instance":
        assert len(node.children) == 0
        # not sure
        return []
    elif node.data == "constructor_dirpath":
        # is not in the general case because it can be empty
        res = []
        for child in node.children:
            res += get_token_sequence(de_bruijn_stack, child)
        return res
    elif node.data == "constructor_prod":
        assert len(node.children) == 3
        label, type_, term = node.children

        result = [("forall", node.id)]
        new_var = get_token_sequence(de_bruijn_stack, label)

        assert len(new_var) == 1
        result += new_var

        result += get_token_sequence(de_bruijn_stack, type_)

        # append the new bound variable to the top of the de_bruijn_stack
        de_bruijn_stack.append(new_var[0][0])
        result += get_token_sequence(de_bruijn_stack, term)
        de_bruijn_stack.pop()
        return result
    elif node.data == "constructor_rel":
        assert len(node.children) == 1
        assert node.children[0].data == "int"
        de_bruijn_index = int(node.children[0].children[0].value)
        # let's hide the de bruijn mechanism from the network
        return [(de_bruijn_stack[-de_bruijn_index], node.id)]
    else:
        assert len(node.children) > 0
        res = []
        for child in node.children:
            res += get_token_sequence(de_bruijn_stack, child)
        return res

add_prefix_preorder_id(tree)
seq = get_token_sequence([], tree)

def get_dict_of_list(node:Union[Tree, MyToken]):
    def inner(node:Union[Tree, MyToken])->List[Tuple[int, List[int]]]:
        if isinstance(node, MyToken):
            return []
        else:
            if node.children:
                res = [(node.id, [child.id for child in node.children])]
                for child in node.children:
                    res += inner(child)
                return res
            else:
                return []
    res = inner(node)
    return dict(inner(node))

res = get_dict_of_list(tree)

# %%