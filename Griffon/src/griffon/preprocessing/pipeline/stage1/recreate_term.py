from lark import Tree, Token
from pygments.lexers import CoqLexer
from functools import reduce

from CoqGym.gallina import GallinaTermParser

from typing import List, Set, Tuple, Union

import networkx as nx

from griffon.coq_dataclasses import Stage1Statement


class MyToken():

    def __init__(self, token:str, id:int) -> None:
        self.value = token
        self.id = id
        self.productive = False

def add_prefix_preorder_id(node : Union[Tree,Token], start_index:int = 0)->Tuple[Union[Tree,MyToken], int]:
    if not isinstance(node, Tree):
        return (MyToken(node.value, start_index), start_index + 1)
    else:
        node.id = start_index
        node.productive = False
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
        node.productive = True
        return [(node.value, node.id)]
    elif node.data == "constructor_anonymous":
        assert len(node.children) == 0
        node.productive = True
        return [("_", node.id)]
    elif node.data == "constructor_prop":
        assert len(node.children) == 0
        node.productive = True
        return [("Prop", node.id)]
    elif node.data == "constructor_set":
        assert len(node.children) == 0
        node.productive = True
        return [("Set", node.id)]
    elif node.data == "constructor_instance":
        node.productive = False
        return []
    elif node.data == "constructor_dirpath":
        # is not in the general case because it can be empty
        res = []
        for child in node.children:
            res += get_token_sequence(de_bruijn_stack, child)

        node.productive = bool(res)
        return res
    elif node.data == "constructor_prod" or node.data == "constructor_lambda":
        assert len(node.children) == 3
        label, type_, term = node.children

        if node.data == "constructor_prod":
            result = [("forall", node.id)]
        else:
            result = [("lambda", node.id)]

        new_var = get_token_sequence(de_bruijn_stack, label)

        assert len(new_var) == 1
        result += new_var

        result += get_token_sequence(de_bruijn_stack, type_)

        # append the new bound variable to the top of the de_bruijn_stack
        de_bruijn_stack.append(new_var[0][0])
        result += get_token_sequence(de_bruijn_stack, term)
        de_bruijn_stack.pop()

        node.productive = True
        return result

    elif node.data == "constructor_rel":
        assert len(node.children) == 1
        assert node.children[0].data == "int"
        de_bruijn_index = int(node.children[0].children[0].value)
        # let's hide the de bruijn mechanism from the network

        node.productive = True
        return [(de_bruijn_stack[-de_bruijn_index], node.id)]

    elif node.data == "constructor_case":
        # let's say we accept empty branches
        assert len(node.children) >= 3
        case_info, type_info, discriminant, *branches = node.children

        result = [("match", node.id)]

        result += get_token_sequence(de_bruijn_stack, type_info)
        result += get_token_sequence(de_bruijn_stack, discriminant)

        for branch in branches:
            result += get_token_sequence(de_bruijn_stack, branch)

        node.productive = True
        return result

    elif node.data == "constructor_fix" or node.data == "constructor_cofix":

        original_constructor = node.data

        if node.data =="constructor_fix":
            assert len(node.children) == 1
            node.productive = True
            node = node.children[0]
            assert node.data == "constr__pfixpoint___constr__constr____constr__constr"
            node.productive = True
            node = node.children[-1]
        else:
            assert len(node.children) == 1
            node.productive = True
            node = node.children[0]
            assert node.data == 'constr__pcofixpoint___constr__constr____constr__constr'
            node.productive = True
            node = node.children[-1]

        assert node.data == "constr__prec_declaration___constr__constr____constr__constr"
        assert len(node.children) % 3 == 0
        number_of_declarations = len(node.children) // 3

        if original_constructor == "constructor_fix":
            keywords = [[("fix", node.id)] for _ in range(number_of_declarations)]
        else:
            keywords = [[("cofix", node.id)]  for _ in range(number_of_declarations)]

        name_children = node.children[:number_of_declarations]
        names = [get_token_sequence(de_bruijn_stack, name) for name in node.children[:number_of_declarations*1]]

        type_info_children =node.children[number_of_declarations:number_of_declarations*2]
        type_infos = [get_token_sequence(de_bruijn_stack, type_info) for type_info in node.children[number_of_declarations*1:number_of_declarations*2]]


        for l in names:
            name = l[0][0]
            de_bruijn_stack.append(name)

        body_children = node.children[number_of_declarations*2:]
        bodies = [get_token_sequence(de_bruijn_stack, body) for body in node.children[number_of_declarations*2:]]

        for _ in names:
            de_bruijn_stack.pop()

        if original_constructor == "constructor_fix":
            result = [("fix", node.id)]
        else:
            result = [("cofix", node.id)]

        reordered_children = []
        for i in range(number_of_declarations):
            result += keywords[i]
            result += names[i]
            result += type_infos[i]
            result += bodies[i]

            reordered_children.append(name_children[i])
            reordered_children.append(type_info_children[i])
            reordered_children.append(body_children[i])

        node.children = reordered_children
        node.productive = True
        return result

    elif node.data == "constructor_evar":

        # temporary fix because I want to move forward
        node.productive = True
        return [("evar", node.id)]

    elif node.data == "constructor_cast":

        value, cast_type, type_ = node.children

        result = get_token_sequence(de_bruijn_stack, value)
        result += get_token_sequence(de_bruijn_stack, type_)

        node.productive = True
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

        node.productive = True
        return result

    else:
        assert node.data not in ["constructor_meta",
                                 "constructor_proj"], \
                f"{node.data} is not handled"
        assert len(node.children) > 0
        res = []
        for child in node.children:
            res += get_token_sequence(de_bruijn_stack, child)
        if len(node.children) == 1:
            node.productive = node.children[0].productive
        else:
            node.productive = reduce(lambda x, y: x or y.productive, node.children, False)
        return res

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
    return dict(inner(node))

def insert_name(name:str, tree:Tree):
    children = [MyToken(name, 0), tree]
    return Tree("custom_node", children)

def prune_non_productive(n : Union[Tree, MyToken]):
    if isinstance(n, Tree):
        productive_children = [prune_non_productive(child) for child in n.children if child.productive]
        n.children = productive_children
    return n

def remove_one_in_one_out_nodes(node:Union[Tree, MyToken], vital_nodes:Set[int])->Union[Tree, MyToken]:
    if isinstance(node, MyToken):
        return node
    else:
        squashed_children = [remove_one_in_one_out_nodes(child, vital_nodes) for child in node.children]

        new_children = []
        for child in squashed_children:
            if isinstance(child, Tree) and len(child.children) == 1:
                assert child not in vital_nodes
                new_children.append(child.children[0])
            else:
                new_children.append(child)

        node.children = new_children

        return node


class Stage1StatementCreator():

    def __init__(self, gallina_parser : GallinaTermParser):
        self.parser = gallina_parser

    def __call__(self, sexp:str, statement_name:str)->Stage1Statement:
        tree = self.parser.parse(sexp)
        tree = insert_name(statement_name, tree)
        add_prefix_preorder_id(tree)
        seq = get_token_sequence([], tree)

        tree = prune_non_productive(tree)


        tokens, nodes = zip(*seq)
        tokens_to_node = dict(enumerate(nodes))

        vital_nodes = tokens_to_node.values()
        tree = remove_one_in_one_out_nodes(tree, vital_nodes)

        graph = nx.from_dict_of_lists(get_dict_of_list(tree))
        return Stage1Statement(statement_name, tokens, graph, tokens_to_node)

    def only_tokens(self, sexp)->List[str]:
        tree = self.parser.parse(sexp)
        add_prefix_preorder_id(tree)
        seq = get_token_sequence([], tree)
        tokens, _ = zip(*seq)
        return tokens

sexp = '(App (Ind (((Mutind (MPfile (DirPath ((Id Peano) (Id Init) (Id Coq)))) (DirPath ()) (Id le)) 0) (Instance ()))) ((Construct ((((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) 1) (Instance ()))) (App (Fix (((0 0) 0) (((Name (Id tree_size)) (Name (Id forest_size))) ((Prod (Name (Id t)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 0) (Instance ()))) (Ind (((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) (Instance ())))) (Prod (Name (Id f)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 1) (Instance ()))) (Ind (((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) (Instance ()))))) ((Lambda (Name (Id t)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 0) (Instance ()))) (Case ((ci_ind ((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 0)) (ci_npar 0) (ci_cstr_ndecls (2)) (ci_cstr_nargs (2)) (ci_pp_info ((ind_tags ()) (cstr_tags ((false false))) (style RegularStyle)))) (Lambda (Name (Id t)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 0) (Instance ()))) (Ind (((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) (Instance ())))) (Rel 1) ((Lambda (Name (Id a)) (Const ((Constant (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id A)) (Instance ()))) (Lambda (Name (Id f)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 1) (Instance ()))) (App (Construct ((((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) 2) (Instance ()))) ((App (Rel 4) ((Rel 1)))))))))) (Lambda (Name (Id f)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 1) (Instance ()))) (Case ((ci_ind ((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 1)) (ci_npar 0) (ci_cstr_ndecls (1 2)) (ci_cstr_nargs (1 2)) (ci_pp_info ((ind_tags ()) (cstr_tags ((false) (false false))) (style RegularStyle)))) (Lambda (Name (Id f)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 1) (Instance ()))) (Ind (((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) (Instance ())))) (Rel 1) ((Lambda (Name (Id b)) (Const ((Constant (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id B)) (Instance ()))) (App (Construct ((((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) 2) (Instance ()))) ((Construct ((((Mutind (MPfile (DirPath ((Id Datatypes) (Id Init) (Id Coq)))) (DirPath ()) (Id nat)) 0) 1) (Instance ())))))) (Lambda (Name (Id t)) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 0) (Instance ()))) (Lambda (Name (Id "f\'")) (Ind (((Mutind (MPfile (DirPath ((Id SerTop)))) (DirPath ()) (Id tree)) 1) (Instance ()))) (App (Const ((Constant (MPfile (DirPath ((Id Nat) (Id Init) (Id Coq)))) (DirPath ()) (Id add)) (Instance ()))) ((App (Rel 5) ((Rel 2))) (App (Rel 4) ((Rel 1)))))))))))))) ((Var (Id t))))))'

term_parser = GallinaTermParser(caching=False)
tree = term_parser.parse(sexp)

statement_creator = Stage1StatementCreator(term_parser)
res = statement_creator(sexp, "hypothesis")
