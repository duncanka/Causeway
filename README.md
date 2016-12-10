# Causeway causal language tagger

Causeway is a system for detecting explicit causal relations in text. It tags text using the scheme described in [Dunietz et al., 2015](http://www.cs.cmu.edu/~jdunietz/publications/causal-language-annotation.pdf). The system itself is described in [Dunietz et al., 2017](http://www.cs.cmu.edu/~jdunietz/publications/causeway-system.pdf). The results described in that paper can be reproduced using the [`run_all_pipelines.sh`](https://github.com/duncanka/causeway/blob/master/Causeway/scripts/run_all_pipelines.sh) script and the data from the [BECauSE](https://github.com/duncanka/BECauSE) corpus.

Causeway relies on the [NLPypline](https://github.com/duncanka/NLPypline) framework for NLP pipelines.

Running Causeway depends on setting up a number of external packages. If you are interested in running/modifying the system or reproducing the results, please [contact Jesse Dunietz](http://www.cs.cmu.edu/~jdunietz/), who will be happy to assist. (Proper documentation will be written up if enough people express interest.)


#### Citations

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. Automatically Tagging Constructions of Causation and Their Slot-Fillers. In press; to be published in 2017. *Transactions of the Association for Computational Linguistics*.</sub>

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. Annotating Causal Language Using Corpus Lexicography of Constructions. *Proceedings of LAW IX – The 9th Linguistic Annotation Workshop* (2015): 188-196.</sub>
