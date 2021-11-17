"""
The ASTGraph is a simple graph representation of the ASTs obtained from semantic or the java parser.
As the CodeTransformer requires a mapping between tokens and AST nodes, a simple algorithm is given to locate tokens
in the AST based on the source span attribute of AST nodes. Tokens are mapped to the node that encompasses the token
as tightly as possible w.r.t. the source span in the code snippet.
"""

from code_transformer.preprocessing.nlp.text import RangeInterval

import networkx as nx

from collections import OrderedDict as odict


class ASTGraph:
    """
    ASTs have some useful properties:
     1) the span of a child node is always contained in the span of the parent node
     2) - the spans of a node's children do not intersect
         This assumption does not hold!
         As an approximation to contain runtime we only traverse into the child with the smallest source span. This
         seems to be sufficient
     3) children of a node are ordered
    """

    ROOT_ID = 0

    def __init__(self, nodes, root):
        self.nodes = nodes
        self.root = root

    @staticmethod
    def from_semantic(json_graph):
        nodes = odict()
        vertex_id_to_n_id = {}
        if 'errors' in json_graph:
            raise Exception(json_graph['errors'])
        for n_id, vertex in enumerate(json_graph['vertices']):
            node = ASTNode(vertex['term'], RangeInterval.from_semantic(vertex['span']))
            nodes[n_id] = node
            vertex_id_to_n_id[vertex['vertexId']] = n_id

        if not 'edges' in json_graph:
            raise Exception(f"No edges in json_graph! {json_graph}")
        for edge in json_graph['edges']:
            source_id = vertex_id_to_n_id[edge['source']]
            target_id = vertex_id_to_n_id[edge['target']]
            nodes[source_id].children.append(target_id)

        root = nodes[ASTGraph.ROOT_ID]
        return ASTGraph(nodes, root)

    @staticmethod
    def from_java_parser(json_graph):
        nodes = odict()

        def traverse_node(node, parent_id):
            for childNode in node['childNodes']:
                if 'sourceRange' in childNode:
                    range = RangeInterval.from_java_parser(childNode['sourceRange'])
                else:
                    range = RangeInterval.empty_interval()
                child_ast_node = ASTNode(childNode['type'], range)
                n_id = len(nodes)
                nodes[n_id] = child_ast_node
                nodes[parent_id].children.append(n_id)
                traverse_node(childNode, n_id)

        node = json_graph
        root_node = ASTNode(node['type'], RangeInterval.from_java_parser(node['sourceRange']))
        nodes[0] = root_node
        traverse_node(node, 0)
        return ASTGraph(nodes, 0)

    @staticmethod
    def from_compressed(compressed_ast):
        for node in compressed_ast.nodes.values():
            node.source_span = RangeInterval.from_compressed(node.source_span)
        return compressed_ast

    def compress(self):
        for node in self.nodes.values():
            node.source_span = node.source_span.compress()
        return self

    def find_smallest_encompassing_interval(self, interval):
        def traverse_children_of(node_id):
            smallest_interval = None
            child_with_smallest_interval = None
            for child_id in self.nodes[node_id].children:
                child_source_span = self.nodes[child_id].source_span
                if child_source_span.contains(interval) \
                        and (smallest_interval is None or child_source_span.is_smaller_than(smallest_interval)):
                    smallest_interval = child_source_span
                    child_with_smallest_interval = child_id
            if child_with_smallest_interval is not None:
                return traverse_children_of(child_with_smallest_interval)
            else:
                return node_id

        return traverse_children_of(self.ROOT_ID)

    def prune(self):
        """
        Prunes 'Empty' nodes that appear sometimes as leafs in ASTs generated by semantic
        """

        def prune_children_of(n_id):
            last = True
            for child_id in self.nodes[n_id].children[::-1]:
                if last and self.nodes[child_id].node_type == 'Empty' and not self.nodes[child_id].children:
                    self.nodes[n_id].children.remove(child_id)
                    del self.nodes[child_id]
                else:
                    prune_children_of(child_id)

        prune_children_of(self.ROOT_ID)

        # Rearrange node IDs to fill eventual holes in node IDs caused by pruning
        map_node_id = {old_id: new_id for new_id, old_id in enumerate(self.nodes.keys())}
        updated_nodes = dict()
        for n_id, node in self.nodes.items():
            updated_node = ASTNode(node.node_type, node.source_span)
            updated_node.children = [map_node_id[child_id] for child_id in node.children]
            updated_nodes[map_node_id[n_id]] = updated_node
        self.nodes = updated_nodes

    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, set) or isinstance(idx, tuple):
            return [self.nodes[n] for n in idx]
        return self.nodes[idx]

    def keys(self):
        return self.nodes.keys()

    def to_networkx(self, create_using=nx.DiGraph):
        dict_of_lists = {n_id: node.children for n_id, node in self.nodes.items()}
        return nx.from_dict_of_lists(dict_of_lists, create_using=create_using)

    def get_node_types(self):
        return {n_id: node.node_type for n_id, node in self.nodes.items()}


class ASTNode:

    def __init__(self, node_type, source_span):
        self.children = []
        self.node_type = node_type
        self.source_span = source_span

    def __str__(self):
        return f"{self.node_type} {self.source_span} {self.children}"