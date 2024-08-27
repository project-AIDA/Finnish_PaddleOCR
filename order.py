import numpy as np

class OrderPolygons:
    def __init__(self, text_direction = 'lr'):
        self.text_direction = text_direction

    # Defines whether two lines overlap vertically
    def _y_overlaps(self, u, v):
        #u_y_min < v_y_max and u_y_max > v_y_min
        return u[3] < v[2] and u[2] > v[3]
    
    # Defines whether two lines overlap horizontally
    def _x_overlaps(self, u, v):
        #u_x_min < v_x_max and u_x_max > v_x_min
        return u[1] < v[0] and u[0] > v[1]
    
    # Defines whether one line (u) is above the other (v)
    def _above(self, u, v):
        #u_y_min < v_y_min
        return u[3] < v[3]

    # Defines whether one line (u) is left of the other (v)
    def _left_of(self, u, v):
        #u_x_max < v_x_min
        return u[0] < v[1]  
    
    # Defines whether one line (w) overlaps with two others (u,v)
    def _separates(self, w, u, v):
        if w == u or w == v:
            return 0
        #w_y_max < (min(u_y_min, v_y_min))
        if w[2] < min(u[3], v[3]):
            return 0
        #w_y_min > max(u_y_max, v_y_max)
        if w[3] > max(u[2], v[2]):
            return 0
        #w_x_min < u_x_max and w_x_max > v_x_min
        if w[1] < u[0] and w[0] > v[1]:
            return 1
        return 0

    # Slightly modified version of the Kraken implementation at
    # https://github.com/mittagessen/kraken/blob/master/kraken/lib/segmentation.py
    def reading_order(self, lines):
        """Given the list of lines, computes
        the partial reading order.  The output is a binary 2D array
        such that order[i,j] is true if line i comes before line j
        in reading order."""
        # Input lines are arrays with 4 polygon coordinates:
        # 0=x_right/x_max, 1=x_left/x_min, 2=y_down/y_max, 3=y_up/y_min
        
        # Array where the order of precedence between the lines is defined
        order = np.zeros((len(lines), len(lines)), 'B')

        # Defines reading direction: default is from left to right
        if self.text_direction == 'rl':
            def horizontal_order(u, v):
                return not self._left_of(u, v)
        else:
            horizontal_order = self._left_of

        for i, u in enumerate(lines):
            for j, v in enumerate(lines):
                if self._x_overlaps(u, v):
                    if self._above(u, v):
                        # line u is placed before line v in reading order
                        order[i, j] = 1
                else:
                        
                    if [w for w in lines if self._separates(w, u, v)] == []:
                        if horizontal_order(u, v):
                            order[i, j] = 1
                    elif self._y_overlaps(u, v) and horizontal_order(u, v):
                        order[i, j] = 1
                    
        return order
    
    # Taken from the Kraken implementation at 
    # https://github.com/mittagessen/kraken/blob/master/kraken/lib/segmentation.py
    def topsort(self, order):
        """Given a binary array defining a partial order (o[i,j]==True means i<j),
        compute a topological sort.  This is a quick and dirty implementation
        that works for up to a few thousand elements."""

        n = len(order)
        visited = np.zeros(n)
        L = []

        def _visit(k):
            if visited[k]:
                return
            visited[k] = 1
            a, = np.nonzero(np.ravel(order[:, k]))
            for line in a:
                _visit(line)
            L.append(k)

        for k in range(n):
            _visit(k)
        return L

    def order(self, lines):
        order = self.reading_order(lines)
        sorted = self.topsort(order)

        return sorted