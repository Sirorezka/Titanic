
import numpy as np
from sklearn import tree
from IPython.display import Image  
from io import StringIO, TextIOBase 
import pydot

def my_data_processing ():

	pass;


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")



def visualize_tree2 (clf, my_feature_names, class_names = []):

    print ("WOWW!")

    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data
                         ,feature_names=my_feature_names  
                         ,class_names=class_names
                         ,filled=True, rounded=True  
                         ,special_characters=True)  

    
    print ("HOOORAY!")
    print (dot_data.getvalue()[1:100])
    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png()) 