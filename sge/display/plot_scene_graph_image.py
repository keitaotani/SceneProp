import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import optimize
from graphviz import Digraph


class PointSEF:
    """Calculate the energy and force of the static electric field of a point charge
    
    Parameters
    ----------
    x : np.ndarray
        The position of the point charge. Shape (n_charges, n_dim)
    q : np.ndarray
        The charge of the point charge. Shape (n_charges,)
    eps : float
        The epsilon to avoid zero division

    Examples
    --------
    >>> x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> q = np.array([1, 1, 1, 1])
    >>> sef = PointSEF(x, q)
    >>> sef.energy()
    5.414213562373095
    >>> sef.force()
    array([[-1.35355339, -1.35355339],
           [ 1.35355339, -1.35355339],
           [-1.35355339,  1.35355339],
           [ 1.35355339,  1.35355339]])
    """

    def __init__(self, x, q, eps=1e-8):
        self.x = x
        self.q = q
        self.eps = eps

    def energy(self):
        """Calculate the energy of the static electric field

        Returns
        -------
        energy : float
            The energy of the static electric field
        """
        distance = np.sqrt(np.sum((self.x[:, None, :] - self.x[None, :, :]) ** 2 + self.eps, axis=-1))
        np.fill_diagonal(distance, np.inf)
        return np.sum(self.q[:, None] * self.q[None, :] / distance) / 2
    
    def force(self):
        """Calculate the force of the static electric field

        Returns
        -------
        force : np.ndarray
            The force of the static electric field. Shape (n_charges, n_dim)
        """
        vector = self.x[:, None, :] - self.x[None, :, :]
        distance = np.sqrt(np.sum(vector ** 2 + self.eps, axis=-1))
        qq = self.q[:, None] * self.q[None, :]
        np.fill_diagonal(distance, np.inf)
        return np.sum(qq[:, :, None] * vector / distance[:, :, None] ** 3, axis=1)


def display_scene_graph_on_image(
        image, boxes, box_labels, rels, rel_labels,
        fig, ax,
        text_size=16,
        optimization_method='TNC'):
    """
    Display the scene graph on the image

    Parameters
    ----------
    image : np.ndarray
        The image to display. Shape (height, width, 3)
    boxes : np.ndarray
        The bounding boxes of the objects. Shape (n_objects, 4)
        The format of the bounding box is (x1, y1, x2, y2)
    box_labels : List[str]
        The labels of the objects. Length is n_objects
    rels : np.ndarray
        The relations between the objects. Shape (n_relations, 2)
        The format of the relation is (subject_id, object_id)
    rel_labels : List[str]
        The labels of the relations. Length is n_relations
    fig : matplotlib.figure.Figure
        The figure to display the scene graph
    ax : matplotlib.axes.Axes
        The axes to display the scene graph
    text_size : int
        The size of the text
    optimization_method : str
        The optimization method to optimize the positions of the text boxes
        'TNC' or 'L-BFGS-B'
    """
    
    colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Display the image
    ax.imshow(image)

    text_boxes = []
    lower_bounds = []
    upper_bounds = []

    # Display the boxes
    for i, ((x1, y1, x2, y2), box_label) in enumerate(zip(boxes, box_labels)):
        cx = (x1 + x2) / 2 + 1e-8 * np.random.rand()
        cy = (y1 + y2) / 2 + 1e-8 * np.random.rand()
        color = colorlist[i % len(colorlist)]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        t = ax.text(
            cx, cy, box_label,
            va="center", ha="center",
            weight="bold", size=text_size, color="white",
            bbox={"facecolor": color, "pad": 0, "linewidth": 0, "edgecolor": color})
        text_boxes.append(t)
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(plt.gca().transData.inverted())
        x_bounds = [x1 + abs(bb.width) / 2, x2 - abs(bb.width) / 2]
        y_bounds = [y1 + abs(bb.height) / 2, y2 - abs(bb.height) / 2]
        lower_bounds += [min(x_bounds), min(y_bounds)]
        upper_bounds += [max(x_bounds), max(y_bounds)]
    
    # Display the relations
    for (sid, oid), rel_label in zip(rels, rel_labels):
        assert sid < len(boxes) and oid < len(boxes)
        x1, y1 = text_boxes[sid].get_position()
        x2, y2 = text_boxes[oid].get_position()
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        t = ax.text(x, y, rel_label, va="center", ha="center", weight="bold", size=text_size, color="white")
        text_boxes.append(t)
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(plt.gca().transData.inverted())
        lower_bounds += [- abs(bb.width) / 2, - abs(bb.height) / 2]
        upper_bounds += [  abs(bb.width) / 2,   abs(bb.height) / 2]
    
    # Optimize the positions of the text boxes
    def to_abscoords(x):
        x = x.copy()
        n_obj = len(boxes)
        for i, (sid, oid) in enumerate(rels):
            x[i + n_obj] += (x[sid] + x[oid]) / 2
        return x
    
    def to_sef(x):
        x = x.reshape(-1, 2)
        x = to_abscoords(x)
        sef = PointSEF(x, np.ones(len(x)))
        return sef
    
    x0 = np.array([t.get_position() for t in text_boxes]).reshape(-1)
    res = optimize.minimize(
        lambda x: to_sef(x).energy(),
        x0,
        method=optimization_method,  # TNC or L-BFGS-B
        jac=lambda x: - to_sef(x).force().reshape(-1),
        bounds=list(zip(lower_bounds, upper_bounds)))
    opt_x = to_abscoords(res.x.reshape(-1, 2))

    for t, (x, y) in zip(text_boxes, opt_x):
        t.set_position((x, y))

    # Draw the arrows
    for (sid, oid), rel_label in zip(rels, rel_labels):
        x1, y1 = text_boxes[sid].get_position()
        x2, y2 = text_boxes[oid].get_position()
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        color = colorlist[sid % len(colorlist)]
        ax.arrow(x, y, x2 - x, y2 - y, head_width=0, head_length=0, width=4, fc=color, ec=color)
        ax.arrow(x1, y1, x - x1, y - y1, head_width=30, head_length=30, width=4, fc=color, ec=color)
    

def display_scene_graph_on_graphviz(
        box_labels, rels, rel_labels
    ):
    colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    g = Digraph(format='png', engine='neato')

    for i, text in enumerate(box_labels):
        color = colorlist[i % len(colorlist)]
        g.node(str(i), text, shape="box", style="filled", fillcolor=color, color=color, fontcolor="white", fontfamily="sans-serif", fontsize="12", fontname="Helvetica", fontweight="bold", width="0", height="0", border="0")

    for i in range(len(rels)):
        color = colorlist[rels[i, 0] % len(colorlist)]
        g.edge(str(rels[i][0].item()), str(rels[i][1].item()), label=rel_labels[i], fontcolor="black", fontfamily="sans-serif", fontsize="12", fontname="Helvetica", color=color, width="2", arrowsize="1", penwidth="2")
        
    return g