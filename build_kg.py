import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import networkx as nx
import pickle

def _identity(x): return x

def _rdflib_to_networkx_graph(
        graph,
        nxgraph,
        calc_weights,
        edge_attrs,
        transform_s=_identity, transform_o=_identity):
    """Helper method for multidigraph, digraph and graph.
    Modifies nxgraph in-place!
    Arguments:
        graph: an rdflib.Graph.
        nxgraph: a networkx.Graph/DiGraph/MultiDigraph.
        calc_weights: If True adds a 'weight' attribute to each edge according
            to the count of s,p,o triples between s and o, which is meaningful
            for Graph/DiGraph.
        edge_attrs: Callable to construct edge data from s, p, o.
           'triples' attribute is handled specially to be merged.
           'weight' should not be generated if calc_weights==True.
           (see invokers below!)
        transform_s: Callable to transform node generated from s.
        transform_o: Callable to transform node generated from o.
    """
    assert callable(edge_attrs)
    assert callable(transform_s)
    assert callable(transform_o)
    for s, p, o in graph:
        if p == RDFS.subClassOf or p==rdflib.term.URIRef('http://hasAnnotation'):
            ts, to = transform_s(s), transform_o(o)  # apply possible transformations
            data = nxgraph.get_edge_data(ts, to)
            if data is None or isinstance(nxgraph, nx.MultiDiGraph):
                # no edge yet, set defaults
                data = edge_attrs(s, p, o)
                if calc_weights:
                    data['weight'] = 1
                nxgraph.add_edge(ts, to, **data)
            else:
                # already have an edge, just update attributes
                if calc_weights:
                    data['weight'] += 1
                if 'triples' in data:
                    d = edge_attrs(s, p, o)
                    data['triples'].extend(d['triples'])


def process_GO_annotations(annotations_file_path):
    """
    Process a GOA file and return a dictionary mapping protein IDs to GO term URLs.
    Arguments:
        annotations_file_path (str): Path to the file containing GO annotations.
    Returns:
        dic_annotations (dict): A dictionary where keys are protein IDs and values are lists of GO term URLs.
    """
    file_annot = open(annotations_file_path, 'r')
    file_annot.readline()
    dic_annotations = {}
    for annot in file_annot:
        list_annot = annot.split('\t')
        id_prot, GO_term = list_annot[1], list_annot[4]

        url_GO_term = "http://purl.obolibrary.org/obo/GO_" + GO_term.split(':')[1]
        url_prot = id_prot

        if url_prot not in dic_annotations:
            dic_annotations[url_prot] = [url_GO_term]
        else:
            dic_annotations[url_prot] = dic_annotations[url_prot] + [url_GO_term]
    file_annot.close()
    return dic_annotations


def process_HP_annotations(annotations_file_path):
    """
    Process a HP annotations file and return a dictionary mapping entity IDs to HPO term URLs.
    Parameters:
        annotations_file_path (str): Path to the file containing HPO annotations.
    Returns:
        dic_annotations (dict): A dictionary where keys are entity IDs and values are lists of HPO term URLs.
    """
    file_annot = open(annotations_file_path, 'r')
    file_annot.readline()
    dic_annotations = {}
    for annot in file_annot:
        list_annot = annot[:-1].split('\t')
        id_ent, HPO_term = list_annot[0], list_annot[1]

        url_HPO_term = 'http://purl.obolibrary.org/obo/HP_' + HPO_term
        url_ent = id_ent

        if url_ent not in dic_annotations:
            dic_annotations[url_ent] = [url_HPO_term]
        else:
            dic_annotations[url_ent] = dic_annotations[url_ent] + [url_HPO_term]

    file_annot.close()
    return dic_annotations


def process_Chebi_annotations(annotations_file_path):
    """
    Process a ChEBI annotations file and return a dictionary mapping drug IDs to ChEBI term URLs.
    Parameters:
        annotations_file_path (str): Path to the file containing ChEBI annotations.
    Returns:
        dic_annotations (dict): A dictionary where keys are drug IDs and values are lists of ChEBI term URLs.
    """
    file_annot = open(annotations_file_path, 'r')
    dic_annotations = {}
    for annot in file_annot:
        drug, chebi_term = annot.strip().split('	')

        url_chebi_term = "http://purl.obolibrary.org/obo/CHEBI_" + chebi_term.split('chebi:')[1]
        url_drug = drug.split('drugbank:')[1]

        if url_drug not in dic_annotations:
            dic_annotations[url_drug] = [url_chebi_term]
        else:
            dic_annotations[url_drug] = dic_annotations[url_drug] + [url_chebi_term]
    file_annot.close()
    return dic_annotations


def process_ATC_annotations(annotations_file_path):
    """
    Process an ATC annotations file and return a dictionary mapping drug IDs to ATC term URLs.
    Parameters:
        annotations_file_path (str): Path to the file containing ATC annotations.
    Returns:
        dic_annotations (dict): A dictionary where keys are drug IDs and values are lists of ATC term URLs.
    
    """
    file_annot = open(annotations_file_path, 'r')
    dic_annotations = {}
    for annot in file_annot:
        drug, atc_term = annot.strip().split('	')

        url_atc_term = "http://purl.bioontology.org/ontology/UATC/" + atc_term.split('atc:')[1]
        url_drug = drug.split('drugbank:')[1]

        if url_drug not in dic_annotations:
            dic_annotations[url_drug] = [url_atc_term]
        else:
            dic_annotations[url_drug] = dic_annotations[url_drug] + [url_atc_term]
    file_annot.close()
    return dic_annotations


def add_annotations(g, dic_annotations):
    """
    Add annotations to an rdflib graph based on a dictionary of annotations.
    Parameters:
        g (rdflib.Graph): An RDFLib graph to which annotations will be added.
        dic_annotations (dict): A dictionary where keys are entity IDs and values are lists of annotation URLs.
    Returns:
        rdflib.Graph: The RDFLib graph with the added annotations.
    """
    dic_labels_classes = {}
    for s, p, o in g.triples((None,RDFS.label, None)):
        dic_labels_classes[str(s)] = str(o)
    for ent in dic_annotations:
        for a in dic_annotations[ent]:
            g.add((rdflib.term.URIRef(ent), rdflib.term.URIRef('http://hasAnnotation'),rdflib.term.URIRef(a)))
    return g, dic_labels_classes


def construct_kg(ontology_file_path, annotations_file_path, type_dataset):
    """
    Construct a rdflib graph by combining an ontology with specific annotations.
    Parameters:
        ontology_file_path (str): Path to the ontology file.
        annotations_file_path (str): Path to the annotations file.
        type_dataset (str): Type of dataset indicating the kind of annotations and format of the annotations file.
                        Accepted values are "PPI", "DDI", "DDI-ATC", and "GDA".
    Returns:
        rdflib.Graph: The RDFLib graph containing the ontology and added annotations.
    """
    g_ontology = rdflib.Graph()
    if type_dataset=="PPI":
        g_ontology.parse(ontology_file_path, format="xml")
        dic_annotations = process_GO_annotations(annotations_file_path)
    elif type_dataset =="DDI":
        g_ontology.parse(ontology_file_path, format="xml")
        dic_annotations = process_Chebi_annotations(annotations_file_path)
    elif type_dataset=="DDI-ATC":
        g_ontology.parse(ontology_file_path, format="turtle")
        dic_annotations = process_ATC_annotations(annotations_file_path)
    elif type_dataset=="GDA":
        g_ontology.parse(ontology_file_path, format="xml")
        dic_annotations = process_HP_annotations(annotations_file_path)
    return add_annotations(g_ontology, dic_annotations)


def run_save_graph(ontology_file_path, annotations_file_path, path_graph, path_label_classes, type='PPI'):
    """
    Construct a rdflib graph from an ontology and annotations, convert it to a NetworkX graph, 
    and save the graph and label classes to files.
    Parameters:
        ontology_file_path (str): Path to the ontology file.
        annotations_file_path (str): Path to the annotations file.
        path_graph (str): Path where the NetworkX graph will be saved.
        path_label_classes (str): Path where the dictionary of label classes will be saved.
        type (str): Type of dataset indicating the kind of annotations and format of the ontology.
                Accepted values are "PPI", "DDI", "DDI-ATC", and "GDA". Default is "PPI".
    Returns:
        None
    """
    g, dic_labels_classes = construct_kg(ontology_file_path, annotations_file_path, type)
    
    # Convert the RDFLib graph to a NetworkX graph
    G = nx.DiGraph()
    _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})

    # Save the NetworkX graph to the specified path
    with open(path_graph, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    # Save the dictionary of label classes to the specified path using pickle
    with open(path_label_classes, 'wb') as path_file:
        pickle.dump(dic_labels_classes, path_file)

    

# if __name__== '__main__':

#     ontology_file_path = "./PPI/go.owl" 
#     annotations_file_path = "./PPI/go_annotations.gaf"
#     path_graph = "./PPI/KG.gpickle"
#     path_label_classes = "./PPI/Labelclasses.pkl"
#     run_save_graph(ontology_file_path, annotations_file_path, path_graph, path_label_classes, 'PPI')