import tkinter as tk
from tkinter import filedialog, messagebox
import math

class GraphVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Visualizer")

        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        self.load_button = tk.Button(root, text="Load File", command=self.load_file)
        self.load_button.pack()

        self.nodes = {}
        self.edges = []

    def load_file(self):
        file_path = "PathFinder-test"
        with open(file_path, 'r') as file:
                lines = file.readlines()

        self.parse_file(lines)
        self.draw_graph()


    def parse_file(self, lines):
        self.nodes.clear()
        self.edges.clear()

        section = None
        for line in lines:
            line = line.strip()

            if line == "Nodes:":
                section = "nodes"
                continue
            elif line == "Edges:":
                section = "edges"
                continue
            elif line == "Origin:":
                section = "origin"
                continue
            elif line == "Destinations:":
                section = "goals"
                continue

            if section == "nodes" and line:
                node_id, coords = line.split(":")
                x, y = map(int, coords.strip("() ").split(","))
                self.nodes[int(node_id)] = (x, y)

            elif section == "edges" and line:
                edge, cost = line.split(":")
                a, b = map(int, edge.strip("() ").split(","))
                cost = int(cost)
                self.edges.append((a, b))

    def draw_graph(self):
        self.canvas.delete("all")

        # Draw edges
        for a, b in self.edges:
            if a in self.nodes and b in self.nodes:
                x1, y1 = self.nodes[a]
                x2, y2 = self.nodes[b]
                self.canvas.create_line(x1 * 50, y1 * 50, x2 * 50, y2 * 50, fill="black")

        # Draw nodes
        for node_id, (x, y) in self.nodes.items():
            x_screen, y_screen = x * 50, y * 50
            self.canvas.create_oval(x_screen - 5, y_screen - 5, x_screen + 5, y_screen + 5, fill="blue")
            self.canvas.create_text(x_screen, y_screen - 10, text=str(node_id), fill="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphVisualizer(root)
    root.mainloop()
