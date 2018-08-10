from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools


#from wrapper import Wrapper
# from tsne import TSNE
#from vtsne import VTSNE
from tsne_utils import Wrapper, VTSNE

import pdb

def preprocess(X, y, perplexity=30, metric='euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    #digits = datasets.load_digits(n_class=6)
    #pos = digits.data
    #y = digits.target
    n_points = X.shape[0]
    distances2 = pairwise_distances(X, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return n_points, pij


def compute_tsne(X, y, t, draw_ellipse=True):
    n_points, pij2d = preprocess(X,y)
    i, j = np.indices(pij2d.shape)
    i = i.ravel()
    j = j.ravel()
    pij = pij2d.ravel().astype('float32')
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    n_topics = 2
    n_dim = 2
    n_iter = 500

    # ! debugging
    n_iter = 0 

    print(n_points, n_dim, n_topics)

    model = VTSNE(n_points, n_topics, n_dim)
    wrap = Wrapper(model, batchsize=4096, epochs=1)
    
    #train:
    for itr in range(n_iter):
        wrap.fit(pij, i, j)

    # Visualize the results
    embed = model.logits.weight.cpu().data.numpy()
    
    #compute distances
    classes = np.unique(y).astype(int)
    distances = np.zeros((len(classes),len(classes)))
    for subset in itertools.combinations(classes, 2):
        print(subset)
        distances[subset] = np.sqrt(((embed[y==subset[0]].mean(0)-embed[y==subset[1]].mean(0))**2).sum())
    
    f = plt.figure()
    if not draw_ellipse:
        plt.scatter(embed[:, 0], embed[:, 1], c=y * 1.0 / y.max())
        plt.axis('off')
    else:
        # Visualize with ellipses
        var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
        ax = plt.gca()
        for xy, (w, h), c in zip(embed, var, y):
            e = Ellipse(xy=xy, width=w, height=h, ec=None, lw=0.0)
            e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
            e.set_alpha(0.5)
            ax.add_artist(e)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)
        plt.axis('off')

    # instead of saving the figure, let's convert it to an image and return it
    canvas = FigureCanvas(f)
    ax = f.gca()
    ax.axis('off')

    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    width, height = f.get_size_inches() * f.get_dpi()
    image = image.reshape(int(height), int(width), 3)
    plt.close(f)

    return distances, image


