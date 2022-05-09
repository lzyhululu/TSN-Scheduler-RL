

class Edge:
    """
    it should be used in the routing algorithm -> route_calcu.py
    expanded when use dynamic routing
    Now everything for simple coding.
    """
    def __init__(self, index, start_node, end_node):
        self.id = index
        self.start_node = start_node
        self.end_node = end_node
