
# Griffon

<img align="right" src="doc/images/logo.png" alt="drawing" width="200"/>

Griffon is the combination of a novel transformer architecture and a Coq plugin that allows user to ask for useful lemma suggestions while trying to prove a theorem. The model is trained on the CoqGym dataset[[1]](#1). The encoder is based on the Code Transformer architecture[[2]](#2). The decoder uses a two level attention, first looking at the titles of each hypothesis in the context to compute an attention distribution over statements before attending to tokens in each statement.

## References

<a id="1">[1]</a> 
Kaiyu Yang, Jia Deng
Learning to Prove Theorems via Interacting with Proof Assistants
CoRR,abs/1905.09381, 2019

<a id="2">[1]</a> 
Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, and Stephan Günnemann
Language-agnostic representation learning of source code from structure and context.
International  Conferenceon Learning Representations (ICLR), 2021

