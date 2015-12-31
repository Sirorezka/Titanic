
import numpy as np
from sklearn import tree
from IPython.display import Image  
from io import StringIO, TextIOBase 
import pydot
import chardet


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

    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data
                         ,feature_names=my_feature_names  
                         ,class_names=class_names
                         ,filled=True, rounded=True  
                         ,special_characters=True)  



    print ("WOWW!")
    print (type(dot_data.getvalue()))

    f = open('img\\tree.dot', 'r')

    graph = pydot.graph_from_dot_data(f.read())
    #Image(graph.create_png()) 


def save_tree_img (filename, clf, my_feature_names, class_names = []):

        print (my_feature_names)
        tree.export_graphviz(clf, out_file=filename
                         ,feature_names=my_feature_names 
                         ,class_names=class_names
                         ,filled=True
                         #,rounded=True  
                         #,special_characters=True
                         )  




# Basic test-control sampling procedure
def generate_data_train_test (X, y, percent):
    if (percent >=1 or percent<=0):
        return


    num_train_data = round(0.75 * X.shape[0],0)     # number of point to sample

    rand_range = range(0, X.shape[0])
    train_arr = np.array(np.random.choice(rand_range, size=num_train_data, replace=False)) # sampling train

    X_train = X[train_arr,:]
    y_train = y[train_arr]

    indices = np.full(X.shape[0], True, bool)
    indices[train_arr] = False
    X_test = X[indices,:]
    y_test = y[indices]


    return X_train, y_train, X_test, y_test




def my_data_processing (data):

    pass;
