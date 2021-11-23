from lark import Tree, Token, Transformer
from lark.visitors import Discard

from CoqGym.gallina import GallinaTermParser
import nltk

from typing import List, Set, Tuple, Union
from griffon.constants import NUM_SUB_TOKENS


from griffon.coq_dataclasses import Stage1Statement, Stage1Token


class HandleSpecialNodes(Transformer):

    def constructor_fix(self, children):
        children = children[0].children[-1].children
        assert len(children) % 3 == 0
        number_of_declaration = len(children) // 3

        name_children = children[:number_of_declaration]
        type_info_children = children[number_of_declaration : number_of_declaration * 2]
        body_children = children[number_of_declaration * 2:]

        new_order_children = []
        for i in range(number_of_declaration):
            new_order_children.append(name_children[i])
            new_order_children.append(type_info_children[i])
            new_order_children.append(body_children[i])

        return Tree("constructor_fix", new_order_children)

    def constructor_cofix(self, children):
        tmp = self.constructor_fix(children)
        tmp.data = "constructor_cofix"
        return tmp

class TreeShortener(Transformer):

    def __default__(self, data, children, meta):
        new_children = []
        for child in children:
            if isinstance(child, Tree) and child.data != "constructor_rel" and len(child.children) == 1:
                new_children.append(child.children[0])
            else:
                new_children.append(child)
        return Tree(data, new_children)

class TreePruner(Transformer):

    allowed_childless_constructors = ["constructor_anonymous",
                                      "constructor_prop",
                                      "constructor_set"]

    always_removed_constructors = ["constructor_instance", "constr__case_info"]

    def __default__(self, data, children, meta):

        assert data not in ["constructor_meta",
                                 "constructor_proj"], \
                f"{data} is not handled"

        if data in self.always_removed_constructors or (len(children) == 0 and data not in self.allowed_childless_constructors):
            return Discard
        else:
            # shouldn't have to do this myself no?
            return Tree(data, [child for child in children if child != Discard])


def clean_string(s : str):
    if s.startswith('"'):
        s = s[1:-1]
    return s

class MyToken():

    def __init__(self, token:str, id:int) -> None:
        self.value = clean_string(token)
        self.id = id

    def __str__(self):
        return self.value

class MyTree():

    def __init__(self, data:str, children:List[Union['MyTree', MyToken]], id:int):
        self.data = data
        self.children = children
        self.id = id

def add_prefix_preorder_id(node : Union[Tree,Token], start_index:int = 0)->Tuple[Union[MyTree,MyToken], int]:
    if not isinstance(node, Tree):
        return (MyToken(node.value, start_index), start_index + 1)
    else:

        new_start_index = start_index + 1
        new_children = []
        for child in node.children:
            res = add_prefix_preorder_id(child, new_start_index)
            new_child, new_start_index = res
            new_children.append(new_child)

        new_node = MyTree(node.data, new_children, start_index)
        return new_node, new_start_index

def get_token_sequence(de_bruijn_stack:List[str], node:Union[MyTree, MyToken])->List[Tuple[str, int]]:
    if isinstance(node, MyToken):
        return [(node.value, node.id)]
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
        return []
    elif node.data == "constructor_dirpath":
        # is not in the general case because it can be empty

        # we unite all of the path component in one token to limit the number of tokens
        if node.children:
            assert (all(isinstance(child, MyToken) for child in node.children))
            return [("_".join(child.value for child in node.children), node.id)] # type: ignore
        else:
            return []

    elif node.data == "constructor_prod" or node.data == "constructor_lambda":
        assert len(node.children) == 3
        label, type_, term = node.children

        if node.data == "constructor_prod":
            result:List[Tuple[str, int]] = [("forall", node.id)]
        else:
            result:List[Tuple[str, int]] = [("lambda", node.id)]

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
        assert isinstance(node.children[0], MyToken)
        de_bruijn_index = int(node.children[0].value)
        # let's hide the de bruijn mechanism from the network

        return [(de_bruijn_stack[-de_bruijn_index], node.id)]

    elif node.data == "constructor_case":
        # let's say we accept empty branches
        assert len(node.children) >= 2
        type_info, discriminant, *branches = node.children

        result = [("match", node.id)]

        result += get_token_sequence(de_bruijn_stack, type_info)
        result += get_token_sequence(de_bruijn_stack, discriminant)

        for branch in branches:
            result += get_token_sequence(de_bruijn_stack, branch)

        return result

    elif node.data == "constructor_fix" or node.data == "constructor_cofix":

        assert len(node.children) % 3 == 0
        number_of_declarations = len(node.children) // 3

        # we need to look ahead in the children to get the bound variables
        # and hide the debruijn mechanism

        names = [get_token_sequence(de_bruijn_stack, name) for name in node.children[0::3]]

        type_infos = [get_token_sequence(de_bruijn_stack, type_info) for type_info in node.children[1::3]]

        for l in names:
            name = l[0][0]
            de_bruijn_stack.append(name)

        bodies = [get_token_sequence(de_bruijn_stack, body) for body in node.children[2::3]]

        for _ in names:
            de_bruijn_stack.pop()

        if node.data == "constructor_fix":
            result = [("fix", node.id)]
        else:
            result = [("cofix", node.id)]

        for i in range(number_of_declarations):
            result += names[i]
            result += type_infos[i]
            result += bodies[i]

        return result

    elif node.data == "constructor_evar":

        # temporary fix because I want to move forward
        return [("evar", node.id)]

    elif node.data == "constructor_cast":

        value, type_ = node.children

        result = get_token_sequence(de_bruijn_stack, value)
        result += get_token_sequence(de_bruijn_stack, type_)

        return result

    elif node.data == "constructor_letin":

        name, term, type_, body = node.children

        result = [("let", node.id)]
        result += get_token_sequence(de_bruijn_stack, name)
        result += get_token_sequence(de_bruijn_stack, term)
        result += get_token_sequence(de_bruijn_stack, type_)

        de_bruijn_stack.append(result[1][0])
        result += get_token_sequence(de_bruijn_stack, body)
        de_bruijn_stack.pop()

        return result

    else:
        assert node.data not in ["constructor_meta",
                                 "constructor_proj"], \
                f"{node.data} is not handled"
        assert len(node.children) > 0
        res = []
        for child in node.children:
            res += get_token_sequence(de_bruijn_stack, child)

        return res

def get_dict_of_list(node:Union[MyTree, MyToken]):
    def inner(node:Union[MyTree, MyToken])->List[Tuple[int, List[int]]]:
        if isinstance(node, MyToken):
            return [(node.id, [])]
        else:
            if node.children:
                res = [(node.id, [child.id for child in node.children])]
                for child in node.children:
                    res += inner(child)
                return res
            else:
                return []
    res = inner(node)
    return dict(res)

def insert_name(name:str, tree:Tree):
    children = [MyToken(name, 0), tree]
    return Tree("custom_node", children)


class Stage1StatementCreator():

    def __init__(self, gallina_parser : GallinaTermParser, sub_tokenizer:nltk.tokenize.RegexpTokenizer):
        self.parser = gallina_parser
        self.sub_tokenizer = sub_tokenizer

    def __call__(self, sexp:str, statement_name:str)->Stage1Statement:
        tree = self.parser.parse(sexp)
        tree = insert_name(statement_name, tree)

        transform = HandleSpecialNodes() * TreeShortener() * TreePruner()
        tree = transform.transform(tree)

        tree, _ = add_prefix_preorder_id(tree)
        seq = get_token_sequence([], tree)

        tokens:List[str] = [token for token, _ in seq]
        tokens_to_node:List[int] = [node_idx for _, node_idx in seq]

        stage_1_tokens = [self.get_stage1token(token) for token in tokens]

        graph = get_dict_of_list(tree)
        return Stage1Statement(statement_name, stage_1_tokens, None, graph, tokens_to_node)

    def only_tokens(self, sexp)->List[Stage1Token]:
        tree = self.parser.parse(sexp)

        transform = HandleSpecialNodes() * TreeShortener() * TreePruner()
        tree = transform.transform(tree)

        tree, _ = add_prefix_preorder_id(tree)
        seq = get_token_sequence([], tree)
        tokens:List[str] = [token for token, _ in seq]

        return [self.get_stage1token(token) for token in tokens]

    def get_stage1token(self, token:str)->Stage1Token:
        subtokens = self.sub_tokenizer.tokenize(token)
        if len(subtokens) > NUM_SUB_TOKENS:
            subtokens = subtokens[-NUM_SUB_TOKENS:]
        return Stage1Token(subtokens)