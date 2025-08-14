import graphviz
import numpy as np
import os
import json

class ObjectNode():
    # The node to store the object information
    def __init__(
        self, parent, node_class, node_id, parent_relation=None, is_part=False
    ):
        self.parent = parent
        self.children = {}
        self.node_class = node_class
        self.node_id = node_id
        self.parent_relation = parent_relation
        self.is_part = is_part

    def delete(self):
        if self.parent:
            del self.parent.children[self.node_id]

    def add_child(self, child):
        self.children[child.node_id] = child

    def get_parent(self):
        return self.parent

    def __str__(self):
        return self.node_id

class SceneGraph:
    def __init__(self, root_node: ObjectNode):
        self.root = root_node
        self.object_nodes = {self.root.node_id: self.root}

    def add_node(
        self, parent: ObjectNode, node_class, node_id, parent_relation, is_part=False
    ):
        object_node = ObjectNode(
            parent=parent,
            node_class=node_class,
            node_id=node_id,
            parent_relation=parent_relation,
            is_part=is_part
        )

        parent.add_child(object_node)
        self.object_nodes[node_id] = object_node
        return object_node
    
    def get_root(self):
        return self.root
    
    def visualize(self, save_dir):
        # Visualize the action-conditioned scene graph
        dag = graphviz.Digraph(
            directory=f"{str(save_dir)}", filename="scene_graph"
        )
        queue = [self.root]

        dag.node(
            name=self.root.node_id,
            label=self.root.node_id,
            shape="egg",
            color="lightblue2",
            style="filled",
        )
        while len(queue) > 0:
            node = queue.pop(0)
            for child in list(node.children.values()):
                if child.is_part:
                    color = "darkorange"
                elif child.parent_relation == "inside":
                    color = "green"
                elif len(child.children) == 0:
                    color = "lightsalmon"
                else:
                    color = "lightblue2"

                # if len(child.children) == 0:
                #     color = "lightsalmon"
                dag.node(
                    name=child.node_id,
                    label=child.node_id,
                    shape="egg",  ## if child.is_object() else "diamond",
                    color=color,
                    style="filled",
                )

                dag.edge(
                    tail_name=node.node_id, 
                    head_name=child.node_id, 
                    label=child.parent_relation
                )

                queue.append(child)

        dag.render()
    
    import json

    def export_json(self, save_path):
        def serialize_node(node):
            return {
                "node_id": node.node_id,
                "is_part": node.is_part,
                "parent_relation": node.parent_relation,
                "children": [
                    serialize_node(child) for child in node.children.values()
                ]
            }

        scene_graph_dict = serialize_node(self.root)
        print(scene_graph_dict)

        with open(save_path, "w") as f:
            json.dump(scene_graph_dict, f, indent=4)
