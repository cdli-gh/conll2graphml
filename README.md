# conll2graphml
This script creates a co-occurrence network of named entities mentioned in annotated texts. It reads input data from CoNLL-U formatted files and extracts relevant information to construct a graph. The graph represents the relationships between named entities based on their co-occurrence in the texts.

## Todo
### Priority
- Strip any morphological annotations before proceeding
- feed the POS tags to look out for from the command or a file, not in the script
- period attribute should be optional
  ### Nice to have
- use the misc column to add additional attributes to the nodes, such as roles
- use the syntax information to create directed edges
- use the syntax information to infer roles (as per @Chiarcos' idea)

## Usage
```
python conll2graph.py conll node_attr edge_attr NE_to_remove
```
  
example:   
```
python conll2graph.py adab.conll nodes_attributes.tsv edges_attributes.tsv NEs_to_remove.txt
```

### Arguments
- `conll`: The path to the CDLI-CoNLL formatted file containing the annotations.
- `node_attr`: The path to the TSV file containing node attributes.
- `edge_attr`: The path to the TSV file containing edge attributes.
- `NE_to_remove`: The path to the TXT file of named entities to remove from the graph.

### Requirements
#### CoNLL
- lemma should be followed with the sense in square brackets, eg Jean-Jacques[1]
- POS is requred for the lemmata to add to the graph
- Morphological annotations must be stripped piror to running the script

#### node_attr and edge_attr
- column names will be used as attributes names

#### NE_to_remove
- One item per line
 

## Dependencies

Make sure you have the following dependencies installed:

- `re`
- `csv`
- `sys`
- `argparse`
- `pandas`
- `numpy`
- `networkx`
- `community`
- `string`

## Steps

The script follows these steps to create the co-occurrence network:

1. Read in the data and extract relevant information from the CDLI-CoNLL file.
2. Create a graph and add nodes and edges representing the co-occurrence relationships between named entities.
3. Read in node attributes from a TSV file and add them to the nodes in the graph.
4. Read in edge attributes from a TSV file and add them to the edges in the graph.
5. Remove specified named entities from the graph.
6. Perform pre-computations on the graph, such as calculating node degrees and weighted degrees, modularity, and identifying k-plex communities.
7. Assign node sizes based on the degree values.
8. Output the graph data to a GraphML file and create separate graph files for each period attribute.
9. Output unmatched node IDs and names to separate files.

## Output

The script produces the following output files:

- `output-graph.graphml`: The co-occurrence network graph in GraphML format.
- `graph_{period}.graphml`: Separate graph files for each period attribute, where `{period}` represents the specific period attribute.
- `unmatched_ids.txt`: A file containing unmatched node IDs.
- `unmatched_names.txt`: A file containing unmatched node names.

Note: Make sure you have write permissions in the script's directory for generating the output files.

Please ensure that you have the necessary input files in the specified formats and run the script with the appropriate command line arguments.
