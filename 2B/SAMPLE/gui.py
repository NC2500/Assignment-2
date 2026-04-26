"""
TBRGS Graphical User Interface (GUI)
Built with Tkinter for easy interaction
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from tbrgs import TBRGS
from config import PathConfig, MLConfig
from graph_builder import BoroondaraGraphBuilder
import os


class TBRGSGUI:
    """Main GUI class for the Traffic-Based Route Guidance System"""

    def __init__(self, root):
        self.root = root
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry("1200x800")

        # Initialize TBRGS in background
        self.tbrgs = None
        self.graph_builder = None
        self.available_sites = []

        # Model options
        self.model_options = ['lstm', 'gru', 'randomforest', 'cnnlstm', 'transformer', 'mlp']
        self.search_options = ['astar', 'bfs', 'dfs', 'gbfs']

        self.setup_ui()
        self.init_system()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Traffic-Based Route Guidance System",
                               font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Status indicator
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                      foreground="blue")
        self.status_label.grid(row=0, column=2, sticky=tk.E)

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        # Model selection
        ttk.Label(config_frame, text="ML Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_var = tk.StringVar(value='lstm')
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_var,
                                   values=self.model_options, state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Search algorithm
        ttk.Label(config_frame, text="Search Algorithm:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.search_var = tk.StringVar(value='astar')
        search_combo = ttk.Combobox(config_frame, textvariable=self.search_var,
                                    values=self.search_options, state="readonly", width=15)
        search_combo.grid(row=0, column=3, sticky=tk.W)

        # Number of routes
        ttk.Label(config_frame, text="Top-K Routes:").grid(row=0, column=4, sticky=tk.W, padx=(20, 5))
        self.k_var = tk.IntVar(value=5)
        k_spinbox = ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.k_var, width=5)
        k_spinbox.grid(row=0, column=5, sticky=tk.W)

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Route Query", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)

        # Origin
        ttk.Label(input_frame, text="Origin:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.origin_var = tk.StringVar(value=PathConfig.DEFAULT_ORIGIN)
        self.origin_combo = ttk.Combobox(input_frame, textvariable=self.origin_var, width=30)
        self.origin_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 20))

        # Destination
        ttk.Label(input_frame, text="Destination:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.dest_var = tk.StringVar(value=PathConfig.DEFAULT_DESTINATION)
        self.dest_combo = ttk.Combobox(input_frame, textvariable=self.dest_var, width=30)
        self.dest_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))

        # Find routes button
        self.find_btn = ttk.Button(input_frame, text="Find Routes",
                                   command=self.find_routes, width=15)
        self.find_btn.grid(row=0, column=4, padx=(20, 0))

        # Results section - split into left (routes list) and right (map/visualization)
        results_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        results_paned.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Left: Routes list
        routes_frame = ttk.LabelFrame(results_paned, text="Routes", padding="10")
        results_paned.add(routes_frame, weight=1)
        routes_frame.columnconfigure(0, weight=1)
        routes_frame.rowconfigure(0, weight=1)

        self.routes_tree = ttk.Treeview(routes_frame, columns=('Time', 'Nodes'),
                                        displaycolumns=(0, 1), show='tree headings')
        self.routes_tree.heading('#0', text='Route')
        self.routes_tree.heading('Time', text='Time (min)')
        self.routes_tree.heading('Nodes', text='# Nodes')
        self.routes_tree.column('#0', width=150)
        self.routes_tree.column('Time', width=80)
        self.routes_tree.column('Nodes', width=80)
        self.routes_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.routes_tree.bind('<<TreeviewSelect>>', self.on_route_select)

        routes_scrollbar = ttk.Scrollbar(routes_frame, orient=tk.VERTICAL,
                                         command=self.routes_tree.yview)
        routes_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.routes_tree.configure(yscrollcommand=routes_scrollbar.set)

        # Right: Route details and visualization
        detail_frame = ttk.LabelFrame(results_paned, text="Route Details", padding="10")
        results_paned.add(detail_frame, weight=2)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=0)

        # Route path display
        self.path_text = scrolledtext.ScrolledText(detail_frame, height=8, width=50)
        self.path_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Matplotlib figure for travel time breakdown
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, detail_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bottom status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

    def init_system(self):
        """Initialize TBRGS system in background thread"""
        def init_worker():
            try:
                # Load graph builder first to get site list
                self.graph_builder = BoroondaraGraphBuilder()
                self.graph_builder.build_graph(method='hybrid')
                self.available_sites = self.graph_builder.get_available_sites()

                # Initialize TBRGS
                self.tbrgs = TBRGS(model_type='lstm')  # Start with default

                # Update combo boxes
                site_ids = [site[0] for site in self.available_sites]
                self.root.after(0, lambda: self.update_site_combos(site_ids))

                self.root.after(0, lambda: self.set_status("Ready"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to initialize: {e}"))

        threading.Thread(target=init_worker, daemon=True).start()

    def update_site_combos(self, site_ids):
        """Update the origin/destination comboboxes with available sites"""
        self.origin_combo['values'] = site_ids
        self.dest_combo['values'] = site_ids

    def set_status(self, message: str):
        """Set status bar message"""
        self.status_var.set(message)

    def find_routes(self):
        """Find routes between origin and destination"""
        origin = self.origin_var.get()
        destination = self.dest_var.get()

        if not origin or not destination:
            messagebox.showwarning("Input Error", "Please select both origin and destination")
            return

        if origin == destination:
            messagebox.showwarning("Input Error", "Origin and destination must be different")
            return

        self.set_status(f"Finding routes from {origin} to {destination}...")
        self.find_btn.config(state='disabled')

        def route_worker():
            try:
                # Update model if changed
                model_type = self.model_var.get()
                if self.tbrgs is None or self.tbrgs.predictor.model_type != model_type:
                    self.tbrgs = TBRGS(model_type=model_type)

                k = self.k_var.get()
                method = self.search_var.get()

                routes = self.tbrgs.find_top_k_paths(origin, destination, k=k, method=method)

                self.root.after(0, lambda: self.display_routes(routes))
                self.root.after(0, lambda: self.set_status("Ready"))

            except Exception as e:
                import traceback
                self.root.after(0, lambda: messagebox.showerror("Error", f"Route finding failed:\n{str(e)}\n{traceback.format_exc()}"))
            finally:
                self.root.after(0, lambda: self.find_btn.config(state='normal'))

        threading.Thread(target=route_worker, daemon=True).start()

    def display_routes(self, routes):
        """Display routes in the treeview"""
        # Clear existing
        for item in self.routes_tree.get_children():
            self.routes_tree.delete(item)

        if not routes:
            messagebox.showinfo("No Routes", "No routes found. Try a different origin/destination pair.")
            return

        for i, route in enumerate(routes):
            time_min = f"{route.travel_time_minutes:.1f}"
            nodes_count = str(len(route.path))
            self.routes_tree.insert('', 'end', text=f"Route {i+1}",
                                   values=(time_min, nodes_count), iid=str(i))

        # Select first route
        self.routes_tree.selection_set('0')
        self.on_route_select(None)

    def on_route_select(self, event):
        """Display details of selected route"""
        selection = self.routes_tree.selection()
        if not selection:
            return

        idx = int(selection[0])
        if idx >= len(self.current_routes if hasattr(self, 'current_routes') else []):
            return

        route = self.current_routes[idx]

        # Update path text
        path_str = " -> ".join(route.path)
        detailed = f"Origin: {route.origin}\n"
        detailed += f"Destination: {route.destination}\n"
        detailed += f"Total Travel Time: {route.travel_time_minutes:.1f} minutes\n"
        detailed += f"Total Travel Time (seconds): {route.total_travel_time:.0f}\n"
        detailed += f"Number of Intersections: {max(0, len(route.path)-2)}\n"
        detailed += f"\nPath ({len(route.path)} nodes):\n{path_str}\n\n"

        if route.edge_times:
            detailed += "Edge Breakdown:\n"
            for i, (time_sec, (from_n, to_n)) in enumerate(
                zip(route.edge_times, zip(route.path[:-1], route.path[1:]))
            ):
                from_name = self.tbrgs.nodes[from_n].location if from_n in self.tbrgs.nodes else from_n
                to_name = self.tbrgs.nodes[to_n].location if to_n in self.tbrgs.nodes else to_n
                detailed += f"  {i+1}. {from_name}\n     -> {to_name}\n"
                detailed += f"     Time: {time_sec/60:.1f} min ({time_sec:.0f} sec)\n"

        self.path_text.delete('1.0', tk.END)
        self.path_text.insert('1.0', detailed)

        # Update bar chart
        self.ax.clear()
        if route.edge_times:
            labels = [f"{i+1}" for i in range(len(route.edge_times))]
            times_min = [t/60 for t in route.edge_times]
            bars = self.ax.bar(labels, times_min, color='steelblue')
            self.ax.set_xlabel('Edge (road segment)')
            self.ax.set_ylabel('Travel Time (minutes)')
            self.ax.set_title('Travel Time per Segment')
            self.ax.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, times_min):
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    @property
    def current_routes(self):
        # Need to access routes from tree selection
        return self._routes if hasattr(self, '_routes') else []

    def display_routes(self, routes):
        """Display routes and store them"""
        self._routes = routes
        for item in self.routes_tree.get_children():
            self.routes_tree.delete(item)

        if not routes:
            messagebox.showinfo("No Routes", "No routes found.")
            return

        for i, route in enumerate(routes):
            time_min = f"{route.travel_time_minutes:.1f}"
            nodes_count = str(len(route.path))
            self.routes_tree.insert('', 'end', text=f"Route {i+1}",
                                   values=(time_min, nodes_count), iid=str(i))

        self.routes_tree.selection_set('0')
        self.on_route_select(None)


def main():
    """Launch the GUI"""
    root = tk.Tk()
    app = TBRGSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
