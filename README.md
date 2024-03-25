# SEEK: Explainable Representations for Relation Prediction in Knowledge Graphs

This repository provides an implementation described in the paper: https://doi.org/10.24963/kr.2023/62.

## Pre-requesites
* install python 3.6.8;
* install python libraries by running the following command:  ```pip install -r req.txt```.

## Methodology

<img src="https://github.com/liseda-lab/seek/blob/main/methodology.png" width="50%"/>

Run the command:
```
python3 run_seek_explanations.py
```

## How to cite

```
@inproceedings{10.24963/kr.2023/62,
author = {Sousa, Rita T. and Silva, Sara and Pesquita, Catia},
title = {Explainable representations for relation prediction in knowledge graphs},
year = {2023},
isbn = {978-1-956792-02-7},
url = {https://doi.org/10.24963/kr.2023/62},
doi = {10.24963/kr.2023/62},
abstract = {Knowledge graphs represent real-world entities and their relations in a semantically-rich structure supported by ontologies. Exploring this data with machine learning methods often relies on knowledge graph embeddings, which produce latent representations of entities that preserve structural and local graph neighbourhood properties, but sacrifice explain-ability. However, in tasks such as link or relation prediction, understanding which specific features better explain a relation is crucial to support complex or critical applications.We propose SEEK, a novel approach for explainable representations to support relation prediction in knowledge graphs. It is based on identifying relevant shared semantic aspects (i.e., subgraphs) between entities and learning representations for each subgraph, producing a multi-faceted and explainable representation.We evaluate SEEK on two real-world highly complex relation prediction tasks: protein-protein interaction prediction and gene-disease association prediction. Our extensive analysis using established benchmarks demonstrates that SEEK achieves significantly better performance than standard learning representation methods while identifying both sufficient and necessary explanations based on shared semantic aspects.},
booktitle = {Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning},
articleno = {62},
numpages = {12},
location = {Rhodes, Greece},
series = {KR '23}
}
```
