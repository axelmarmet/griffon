
transformer contains pointer

pointer calculate attention over memory embeddings (in a weird stateful way but it works)

standard transformer decoder computes logits in the standard way

logits are combined

loss applied per sample (necessary because they all have different vocabs)