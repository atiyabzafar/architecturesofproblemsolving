import numpy as np
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
from typing import List, Tuple, Set

# ============================================================
# Agent Classes
# ============================================================

class Person(Agent):
    """
    A person agent that maintains a binary assignment (x) and 
    a knowledge base (kb) of clauses.
    """
    def __init__(self, unique_id, model, K):
        super().__init__(unique_id, model)
        self.K = K
        # Initialize random binary assignment for K variables
        self.x = [random.choice([0, 1]) for _ in range(K)]
#        self.kb = set()  # Knowledge base of clauses
        self.kb = [] # list of clauses , oldest first , knowledge base of clauses
        self.true_violations = 0  # Violations against global clause set C
        self.centr = 0.0  # Centrality measure
        self.in_neighbors = None

       # self.neighbor_cache = None  # Cache neighbors


    def add_clause_to_kb(self, clause):
        self.kb.append(clause)
        # bound-knowledgebase-size analogue
        M = self.model.M
        while len(self.kb) > M:
            self.kb.pop(0)

    # def clause_violated(self, clause, assign):
    #     """Check if AND/XOR clause is violated."""
    #     operator, var_indices = clause
        
    #     if operator == "AND":
    #         # ALL variables must be 1
    #         for var_idx in var_indices:
    #             if assign[var_idx - 1] == 0:
    #                 return True
    #         return False
        
    #     elif operator == "XOR":
    #         # EXACTLY ONE variable must be 1
    #         #count_ones = sum(assign[var_idx - 1] for var_idx in var_indices)
    #         #return count_ones != 1

    #         #matching Netlogo code
    #         parity_sum = sum(assign[var_idx - 1] for var_idx in var_indices)
    #         return (parity_sum % 2) != 1  # violated if NOT odd parity
        
    #     else:
    #         raise ValueError(f"Unknown operator: {operator}")
        
    # def clause_violated(self, clause, assign):
    #     """Check if AND/XOR clause is violated."""
    #     operator, var_indices = clause
        
    #     if operator == "AND":
    #         # ALL variables must be 1
    #         for var_idx in var_indices:
    #             if assign[var_idx - 1] == 0:
    #                 return True
    #         return False
        
    #     elif operator == "XOR":
    #         # EXACTLY ONE variable must be 1
    #         #count_ones = sum(assign[var_idx - 1] for var_idx in var_indices)
    #         #return count_ones != 1

    #         #matching Netlogo code
    #         parity_sum = sum(assign[var_idx - 1] for var_idx in var_indices)
    #         return (parity_sum % 2) != 1  # violated if NOT odd parity
        
    #     else:
    #         raise ValueError(f"Unknown operator: {operator}")

    def clause_violated(self, clause, assign):
        """Check if AND/XOR clause is violated."""
        operator, var_indices = clause
        
        if operator == "AND":
            # ALL variables must be 1
            for var_idx in var_indices:
                if assign[var_idx - 1] == 0:
                    return True
            return False
        
        elif operator == "XOR":
            # EXACTLY ONE variable must be 1
            #count_ones = sum(assign[var_idx - 1] for var_idx in var_indices)
            #return count_ones != 1

            #matching Netlogo code
            parity_sum = sum(assign[var_idx - 1] for var_idx in var_indices)
            return (parity_sum % 2) != 1  # violated if NOT odd parity
        
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def violation_count(self, kb_list, assign):
        """Count violated clauses."""
        return sum(1 for clause in kb_list if self.clause_violated(clause, assign))

    def cache_neighbors(self):
        self.in_neighbors = list(self.model.network.predecessors(self.unique_id))

            
    # def violation_count(self, kb_list, assign):
    #     """Inlined for speed - eliminates function call overhead"""
    #     count = 0
    #     for clause in kb_list:
    #         satisfied = False
    #         for j, sign in clause:
    #             val = assign[j - 1]
    #             # Inline the logic
    #             if (sign == 1 and val == 1) or (sign == -1 and val == 0):
    #                 satisfied = True
    #                 break
    #         if not satisfied:
    #             count += 1
    #     return count

    def step(self):
        """
        Agents learn by communicating and observing.
        Communication is now proportional to in-degree.
        """
        my_kb_set = set(self.kb)

        # 1) Private observation
        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)

            self.add_clause_to_kb(obs_clause)
            self.local_update_around(obs_clause)
            my_kb_set.add(obs_clause)
            # if obs_clause not in self.kb:
            #     self.kb.add(obs_clause)
            #     self.local_update_around(obs_clause)

        # 2) Communication - proportional to in-degree
        # Get all incoming edges (predecessors)
        in_neighbors = self.in_neighbors
        
        if in_neighbors:
            comm_scale = self.model.comm_scale
            for nbr_id in in_neighbors:
                # Get edge/weight
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get('weight', 1.0)

                # Calculate probability for this link
                link_probability = min(1.0, comm_scale * link_weight)
                
                if random.random() < link_probability:
                    nbr_agent = self.model.agent_list[nbr_id]  # Use fast array/index, not lookup
                    if nbr_agent.kb:  # Only if neighbor knows something
                        nbr_kb_set = set(nbr_agent.kb)
                        unknowns = nbr_kb_set - my_kb_set
                        if unknowns:
                            cprime = random.choice(list(unknowns))
                            self.add_clause_to_kb(cprime)
                            self.local_update_around(cprime)
                            my_kb_set.add(cprime)


#             # For each incoming link, independently try to elicit information
#             for nbr_id in in_neighbors:
#                 # Get the edge weight
#                 edge_data = self.model.network[nbr_id][self.unique_id]
#                 link_weight = edge_data.get('weight', 1.0)

#                 # Calculate probability for this link
#                 link_probability = min(1.0, comm_scale * link_weight)

#                 # Probabilistically elicit information
#                 if random.random() < link_probability:
#                     nbr = self.model.schedule.agents[nbr_id]

#                     # Find clauses this neighbor knows that we don't
#                     #unknowns = nbr.kb - self.kb
#                     neighbour_kb = set(nbr.kb)
#                     unknowns = list(neighbour_kb - my_kb_set)

#                     if unknowns:
#                         # Elicit one random unknown clause
#                         cprime = random.choice(list(unknowns))
#                         self.add_clause_to_kb(cprime)
# #                        self.kb.add(cprime)
#                         self.local_update_around(cprime)

    def local_update_around(self, clause):
        """
        Local optimization around a newly learned clause.
        Works with AND/XOR clause format: (operator, (var_indices))
        """
        # Extract variable indices and shuffle (matches NetLogo)
        operator, var_indices = clause
        indices = list(var_indices)
        random.shuffle(indices)  # ✓ ADDED
        
        # Find clauses that mention any of these variables
        related_kb = [cl for cl in self.kb 
                    if any(var in cl[1] for var in indices)]
        
        if not related_kb:
            return
        
        # Baseline violations
        Vi_total = self.violation_count(related_kb, self.x)
        
        for j in indices:
            idx = j - 1
            old = self.x[idx]
            
            # Clauses affected by flipping j
            # ✓ FIXED: Use cl[1] to access variables in (operator, vars) format
            affected_kb = [cl for cl in related_kb if j in cl[1]]
            
            # Old violations
            old_aff = self.violation_count(affected_kb, self.x)
            
            # Try flip
            self.x[idx] = 1 - old
            
            # New violations  
            new_aff = self.violation_count(affected_kb, self.x)
            
            # New total
            Vnew_total = Vi_total - old_aff + new_aff
            
            # Accept only if improvement
            if Vnew_total >= Vi_total:
                self.x[idx] = old  # Revert
            else:
                Vi_total = Vnew_total  # Update baseline


#     def local_update_around(self,clause):
#         """
#         Optmising by minimising the violations
#         """
#         #getting all indices from the clause
# #        indices=[lit[0] for lit in clause]
# #        random.shuffle(indices)
#         operator, var_indices = clause  # Extract from new format
#         indices = list(var_indices)
        
#         # Find related clauses (access variables with cl[1])
#         related_kb = [cl for cl in self.kb if any(var in cl[1] for var in indices)]
#         #finding clauses that mention any of these variables
#     #    related_kb=[]
#     #    for cl in self.kb:
#     #        for lit in cl:
#     #            if any(lit[0] in indices):
#     #                related_kb.append(cl)
#         # kb is a set, convert to list for filtering
        
#         #related_kb = [cl for cl in self.kb 
#         #            if any(lit[0] in indices for lit in cl)]


#         if not related_kb:
#             return 
#         # Baseline: violations in the related part of the kb
        
#         Vi_total = self.violation_count(related_kb, self.x)

#         for j in indices:
#             idx = j - 1

#             old = self.x[idx]

#             #clauses actually affected by flipping j
#             affected_kb= []
#             for cl in related_kb:
#                 lits=[]
#                 for lit in cl:
#                     lits.append(lit[0])
#                 if j in lits:
#                     affected_kb.append(cl)

#             # Old violations among affected clauses
#             old_aff = self.violation_count(affected_kb, self.x)
            
#             # Tentative flip
#             self.x[idx] = 1 - old
            
#             # New violations among affected clauses
#             new_aff = self.violation_count(affected_kb, self.x)

#             # New total for related-kb after the flip
#             Vnew_total = Vi_total - old_aff + new_aff
            
#             # Accept only if strict improvement; otherwise revert
#             if Vnew_total >= Vi_total:
#                 self.x[idx] = old  # REVERT if not improving
#             else:
#                 Vi_total = Vnew_total  # THIS WAS MISSING! Update baseline for next iteration


# ============================================================
# Model Class
# ============================================================

class ProblemSolvingModel(Model):
    """
    A model of collective problem-solving on a social network.
    Agents maintain binary assignments to K variables and learn clauses
    through private observation and social communication.
    """
    
    def __init__(self, 
                 N=50,                    # Number of agents
                 K=20,                    # Number of binary variables
                 alpha=4,               # Clause density (M = round(alpha * K))
                 obs_prob=0.1,            # Probability of private observation
                 clause_interval=10,      # Ticks between clause replacements
                 R=1000,                  # Run horizon (number of ticks)
                 setup_source="generate", # "generate" , "dataset" or "graph"
                 file_path=None,          # path to dataset 
                 input_graph=None,        # input networkx graph
                 type_network="Random",   # Network type
                 connect_prob=0.1,        # For Random network
                 n_size=4,                # For Small World
                 rewire_prob=0.1,         # For Small World
                 min_deg=2,               # For Scale Free
                 nlayers=3,               # For Hierarchical
                 intra_layer_connectance=0.5,  # For Hierarchical
                 inter_layer_connectance=0.1,  # For Hierarchical
                 random_layersize=False,  # For Hierarchical
                 seed=None):
        
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Model parameters
        self.N = N
        self.K = K
        self.alpha = alpha
        self.M = round(alpha * K)  # Number of clauses
        self.obs_prob = obs_prob
        self.clause_interval = clause_interval
        self.R = R
        self.setup_source = setup_source
        self.type_network = type_network
        self.input_graph = input_graph
        
        # Network parameters
        self.connect_prob = connect_prob
        self.n_size = n_size
        self.rewire_prob = rewire_prob
        self.min_deg = min_deg
        self.nlayers = nlayers
        self.intra_layer_connectance = intra_layer_connectance
        self.inter_layer_connectance = inter_layer_connectance
        self.random_layersize = random_layersize
        self.file_path = file_path
        
        # Global state
        self.C = []  # Universal clause set
        self.avg_true_V = 0
        self.min_true_V = 0
        self.homogeneity = 0

        self.comm_scale = 0.0  # Communication scale factor
        
        # Network (directed graph with weighted edges)
        self.network = nx.DiGraph()
        
        # Validate
        if self.setup_source == "dataset" and self.file_path is None:
            raise ValueError("file_path must be specified when setup_source='file'")


        # Scheduler
        self.schedule = RandomActivation(self)
        
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "avg_violations": lambda m: m.avg_true_V,
                "min_violations": lambda m: m.min_true_V,
                "homogeneity": lambda m: m.homogeneity,
                "avg_centrality": lambda m: np.mean([a.centr for a in m.schedule.agents]),
            },
            agent_reporters={
                "violations": "true_violations",
                "centrality": "centr",
                "kb_size": lambda a: len(a.kb),
            }
        )
        self.generate_clauses()
        self.setup_network()
        if self.N <5000:
            self.compute_centrality()
        self.calc_performances()
        
    def generate_clauses(self):
        """
        Generate M random clauses of length 2 or 3
        """
        self.C = []
        for _ in range(self.M):
            self.C.append(self.random_clause())

    # def random_clause(self):
    #     """
    #     create a random clause of length 2 or 3 involving 1..K varible_index
    #     """
    #     #L = 2 if random.random() < 0.5 else 3
    #     L=2
    #     indices = random.sample(range(1, self.K + 1), L)
    #     # Create as tuple directly
    #     clause = tuple((j, random.choice([-1, 1])) for j in indices)
    #     return self.canonicalise_clause(clause)
    
    # def canonicalise_clause(self, clause):
    #     """Sort clause by variable index, then by sign."""
    #     return tuple(sorted(clause, key=lambda lit: (lit[0], lit[1])))

    def random_clause(self):
        """Create random AND or XOR clause over 2 variables."""
        L = 2  # Always 2 variables
        indices = random.sample(range(1, self.K + 1), L)
        
        # 50% AND, 50% XOR
        operator = "AND" if random.random() < 0.5 else "XOR"
        clause = (operator, tuple(indices))
        
        return self.canonicalise_clause(clause)

    def canonicalise_clause(self, clause):
        """Sort variables within clause."""
        operator, variables = clause
        return (operator, tuple(sorted(variables)))



    def setup_network(self):
        """
        setup networks of different types based on the parameters
        """

        if self.setup_source == "generate":
            #create agents
            # for i in range(self.N):
            #     agent=Person(i,self,self.K)
            #     self.network.add_node(i)
            #     self.schedule.add(agent)
            self.agent_list = [None] * self.N
            for i in range(self.N):
                agent = Person(i, self, self.K)
                self.schedule.add(agent)
                self.agent_list[i] = agent
            if self.type_network == "Random":
                self.setup_random_network()
            if self.type_network == "Small World":
                self.setup_small_world_network()
            if self.type_network == "Scale Free":
                self.setup_scale_free_network()
            if self.type_network == "Hierarchical":
                self.setup_hierarchical_network()
        elif self.setup_source == "dataset":
            # Load from GraphML file
            self.load_network_from_graphml(self.file_path)

        elif self.setup_source == "graph":
            if self.input_graph is None:
                raise ValueError("input_graph must be provided when setup_source='graph'")
            self.load_network_from_graph(self.input_graph)
        
        else:
            raise ValueError(f"Unknown setup_source: {self.setup_source}")
        
        for agent in self.schedule.agents:
            agent.cache_neighbors()

        # Compute communication scale
        self.compute_comm_scale()

        # Cache neighbors for all agents
        #for agent in self.schedule.agents:
        #    agent.cache_neighbors()

        
    def setup_random_network(self):
        """Create Erdős-Rényi random directed network."""
        for i in range(self.N):
            for j in range(self.N):
                if i != j and random.random() < self.connect_prob:
                    weight = 0.0001 + random.random() * 0.9999
                    self.network.add_edge(i, j, weight=weight)
    
    def setup_small_world_network(self):
        """Create Watts-Strogatz small world network (converted to directed)."""
        # Create ring lattice - only one direction per edge
        for i in range(self.N):
            for offset in range(1, self.n_size + 1):
                j = (i + offset) % self.N
                # Random direction: 50% i→j, 50% j→i
                if random.random() < 0.5:
                    self.network.add_edge(i, j, weight=1.0)
                else:
                    self.network.add_edge(j, i, weight=1.0)

        # Rewire edges (same logic as before)
        edges_to_rewire = list(self.network.edges())
        for i, j in edges_to_rewire:
            if random.random() < self.rewire_prob:
                self.network.remove_edge(i, j)
                k = random.choice([n for n in range(self.N) if n != i and not self.network.has_edge(i, n)])
                weight = 0.0001 + random.random() * 0.9999
                self.network.add_edge(i, k, weight=weight)

    def setup_scale_free_network(self):
        """Create Barabási-Albert scale-free network (converted to directed)."""
        # Start with min_deg nodes fully connected
        for i in range(self.min_deg):
            for j in range(self.min_deg):
                if i != j:
                    self.network.add_edge(i, j, weight=1.0)
        
        # Add remaining nodes with preferential attachment
        for i in range(self.min_deg, self.N):
            # Calculate degree distribution for preferential attachment
            degrees = dict(self.network.in_degree())
            total = sum(degrees.values()) if degrees else self.min_deg
            
            targets = []
            for _ in range(self.min_deg):
                if not degrees:
                    target = random.choice(range(i))
                else:
                    # Preferential attachment
                    ran = random.random() * total
                    acc = 0
                    for node, deg in degrees.items():
                        acc += deg
                        if ran <= acc:
                            target = node
                            break
                    else:
                        target = list(degrees.keys())[-1]
                
                if target not in targets:
                    targets.append(target)
                    weight = 0.0001 + random.random() * 0.9999
                    # Create only ONE directed edge with random direction
                    if random.random() < 0.5:
                        self.network.add_edge(i, target, weight=weight)
                    else:
                        self.network.add_edge(target, i, weight=weight)


    def setup_hierarchical_network(self):
        """Create hierarchical layered network."""
        # Determine layer sizes
        if self.random_layersize:
            layer_sizes = [0] * self.nlayers
            remaining = self.N
            while remaining > 0:
                i = random.randint(0, self.nlayers - 1)
                layer_sizes[i] += 1
                remaining -= 1
        else:
            # Near-equal split
            base = self.N // self.nlayers
            rem = self.N - base * self.nlayers
            layer_sizes = [base] * self.nlayers
            for j in range(rem):
                layer_sizes[j] += 1
        
        # Assign agents to layers
        agents = list(range(self.N))
        random.shuffle(agents)
        layers = []
        start = 0
        for sz in layer_sizes:
            layer = agents[start:start + sz]
            layers.append(layer)
            start += sz
        
        # Intra-layer directed links
        for layer in layers:
            for i in layer:
                for j in layer:
                    if i != j and random.random() < self.intra_layer_connectance:
                        weight = 0.0001 + random.random() * 0.9999
                        self.network.add_edge(i, j, weight=weight)
        
        # Inter-layer directed links
        for a in range(len(layers)):
            for b in range(len(layers)):
                if a != b:
                    for i in layers[a]:
                        for j in layers[b]:
                            if random.random() < self.inter_layer_connectance:
                                weight = 0.0001 + random.random() * 0.9999
                                self.network.add_edge(i, j, weight=weight)

    def convert_undirected_to_asymmetric_directed(self, undirected_graph):
        """
        Convert an undirected graph to directed by assigning each edge
        a random direction (matches NetLogo's convert-links-to-tlinks).

        Args:
            undirected_graph: NetworkX undirected graph

        Returns:
            NetworkX directed graph with single directed edge per undirected edge
        """
        directed_graph = nx.DiGraph()

        # Copy nodes
        directed_graph.add_nodes_from(undirected_graph.nodes(data=True))

        # For each undirected edge, create ONE directed edge with random direction
        for u, v in undirected_graph.edges():
            # Skip self-loops
            if u == v:
                continue

            # Random direction: 50% u→v, 50% v→u
            if random.random() < 0.5:
                directed_graph.add_edge(u, v, weight=1.0)
            else:
                directed_graph.add_edge(v, u, weight=1.0)

        return directed_graph

    def load_network_from_graphml(self, filepath):
        """
        Load network structure from GraphML file and create integer-indexed network.
        Original node IDs and weights from file are ignored.
        
        Parameters:
        -----------
        filepath : str
            Path to the GraphML file (e.g., "data/network.graphml")
        
        Notes:
        ------
        - Loads only the network topology (which nodes connect to which)
        - Node IDs are remapped to 0, 1, 2, ..., N-1
        - Weights are randomly assigned (0.0001 to 1.0)
        - Converts undirected graphs to directed
        """
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GraphML file not found: {filepath}")
        
        try:
            # Load graph structure from file
            loaded_graph = nx.read_graphml(filepath)
            print(f"Loaded graph from {filepath}")
            print(f"  Nodes: {loaded_graph.number_of_nodes()}")
            print(f"  Edges: {loaded_graph.number_of_edges()}")
            print(f"  Type: {'Directed' if loaded_graph.is_directed() else 'Undirected'}")
            
            # Update N to match the loaded graph
            self.N = loaded_graph.number_of_nodes()
            
            # Convert to directed if needed
            if not loaded_graph.is_directed():
                loaded_graph = self.convert_undirected_to_asymmetric_directed(loaded_graph)
               # loaded_graph = loaded_graph.to_directed()
                print("  Converted to directed graph")
            
            # Create mapping: original node ID → integer index (0..N-1)
            original_node_ids = list(loaded_graph.nodes())
            node_mapping = {orig_id: i for i, orig_id in enumerate(original_node_ids)}
            
            # Create new clean integer-indexed network
            self.network = nx.DiGraph()
            # Bulk add nodes to NetworkX graph
            self.network.add_nodes_from(range(self.N))

            # Create all Person agents at once (as a list)
            self.agent_list = [Person(i, self, self.K) for i in range(self.N)]
#            agents = [Person(i, self, self.K) for i in range(self.N)]
            # Add to the scheduler if possible in a batch, or within a fast loop
            for agent in self.agent_list:
                self.schedule.add(agent)
            edge_list = [ (node_mapping[orig_u], node_mapping[orig_v])
              for orig_u, orig_v in loaded_graph.edges() ]

            # Generate all weights at once
            random_weights = 0.0001 + np.random.rand(len(edge_list)) * 0.9999

            edge_tuples = [ (u, v, {'weight': w}) for (u, v), w in zip(edge_list, random_weights) ]

            self.network.add_edges_from(edge_tuples)
            # # Create nodes with integer indices and corresponding agents
            # for i in range(self.N):
            #     self.network.add_node(i)  # Node ID is just the integer i
            #     agent = Person(i, self, self.K)
            #     self.schedule.add(agent)
            
            # # Add edges using integer indices with random weights
            # for orig_u, orig_v in loaded_graph.edges():
            #     # Map original IDs to integer indices
            #     u_idx = node_mapping[orig_u]
            #     v_idx = node_mapping[orig_v]
                
            #     # Assign random weight (ignore any weight from file)
            #     weight = 0.0001 + random.random() * 0.9999
                
            #     self.network.add_edge(u_idx, v_idx, weight=weight)
            
            print(f"  Created {self.N} agents with indices 0..{self.N-1}")
            print(f"  Added {self.network.number_of_edges()} directed edges with random weights")
            
        except Exception as e:
            raise IOError(f"Error loading GraphML file: {str(e)}")


    def load_network_from_graph(self, input_graph):
        """
        Load network structure from an existing NetworkX graph.
        Original node IDs and weights are ignored - topology only is preserved.
        
        Parameters
        -----------
        input_graph : networkx.Graph or networkx.DiGraph
            Input NetworkX graph (can be directed or undirected)
        
        Notes
        ------
        - Loads only the network topology (which nodes connect to which)
        - Node IDs are remapped to 0, 1, 2, ..., N-1
        - Weights are randomly assigned (0.0001 to 1.0)
        - Converts undirected graphs to directed (one random direction per edge)
        - Creates agents matching the number of nodes in the graph
        """
        import copy
        
        # Make a copy to avoid modifying the original graph
        loaded_graph = copy.deepcopy(input_graph)
        
        print(f"Loading graph from NetworkX object")
        print(f"  Nodes: {loaded_graph.number_of_nodes()}")
        print(f"  Edges: {loaded_graph.number_of_edges()}")
        print(f"  Type: {'Directed' if loaded_graph.is_directed() else 'Undirected'}")
        
        # Update N to match the loaded graph
        self.N = loaded_graph.number_of_nodes()
        
        # Create new clean integer-indexed network
        self.network = nx.DiGraph()
        
        # Convert to directed if needed
        if not loaded_graph.is_directed():
            loaded_graph = self.convert_undirected_to_asymmetric_directed(loaded_graph)
            print("  Converted to directed graph")
        
        # Create mapping: original node ID -> integer index (0..N-1)
        original_node_ids = list(loaded_graph.nodes())
        node_mapping = {orig_id: i for i, orig_id in enumerate(original_node_ids)}
        
        # Bulk add nodes to NetworkX graph
        self.network.add_nodes_from(range(self.N))
        
        # Create all Person agents at once
        self.agent_list = [None] * self.N
        for i in range(self.N):
            agent = Person(i, self, self.K)
            self.schedule.add(agent)
            self.agent_list[i] = agent
        
        print(f"  Created {self.N} agents with indices 0..{self.N-1}")
        
        # Prepare all edges with mapped indices and random weights
        edge_list = [(node_mapping[orig_u], node_mapping[orig_v]) 
                    for orig_u, orig_v in loaded_graph.edges()]
        
        # Generate all weights at once (using NumPy for speed)
        random_weights = 0.0001 + np.random.rand(len(edge_list)) * 0.9999
        edge_tuples = [(u, v, {'weight': w}) for (u, v), w in zip(edge_list, random_weights)]
        
        # Bulk add edges
        self.network.add_edges_from(edge_tuples)
        
        print(f"  Added {self.network.number_of_edges()} directed edges with random weights")


    def compute_centrality(self):
        """
        Compute in-degree centrality for all agents.
        Normalized to [0, 1] where 1 is the highest centrality agent.
        """
        # Check if network uses non-uniform weights
        uses_weights = any(
            data.get('weight', 1.0) != 1.0 
            for u, v, data in self.network.edges(data=True)
        )
        
        if uses_weights:
            # Weighted: compute in-strength
            in_strengths = []
            for agent_id in range(self.N):
                in_edges = self.network.in_edges(agent_id, data=True)
                in_strength = sum(data.get('weight', 1.0) for u, v, data in in_edges)
                in_strengths.append(in_strength)
            
            max_strength = max(in_strengths) if in_strengths else 1.0
            max_strength = max(max_strength, 1.0)
            
            for agent_id, in_strength in enumerate(in_strengths):
                agent = self.schedule.agents[agent_id]
                agent.centr = in_strength / max_strength
        
        else:
            # Unweighted: compute in-degree
            in_degrees = [self.network.in_degree(agent_id) for agent_id in range(self.N)]
            
            max_degree = max(in_degrees) if in_degrees else 1.0
            max_degree = max(max_degree, 1.0)
            
            for agent_id, in_degree in enumerate(in_degrees):
                agent = self.schedule.agents[agent_id]
                agent.centr = in_degree / max_degree


    def compute_comm_scale(self):
        """
        Compute the global communication scale factor.
        Ensures that on average, each agent receives ~1 token of information.
        """
        # Check if network uses non-uniform weights
        uses_weights = any(
            data.get('weight', 1.0) != 1.0 
            for u, v, data in self.network.edges(data=True)
        )

        if uses_weights:
            # Weighted network: average in-strength
            in_strengths = []
            for agent_id in range(self.N):
                in_edges = self.network.in_edges(agent_id, data=True)
                in_strength = sum(data.get('weight', 1.0) for u, v, data in in_edges)
                in_strengths.append(in_strength)

            avg_base_inflow = np.mean(in_strengths) if in_strengths else 0
        else:
            # Unweighted network: average in-degree
            in_degrees = [self.network.in_degree(agent_id) for agent_id in range(self.N)]
            avg_base_inflow = np.mean(in_degrees) if in_degrees else 0

        # Scale factor: 1 / average
        self.comm_scale = (1.0 / avg_base_inflow) if avg_base_inflow > 0 else 0.0

    def calc_performances(self):
        for agent in self.schedule.agents:
            agent.true_violations = agent.violation_count(self.C,agent.x)
        violations = [agent.true_violations for agent in self.schedule.agents]
        self.avg_true_V = np.mean(violations) if violations else 0
        self.min_true_V = min(violations) if violations else 0
        self.homogeneity = self.compute_homogeneity()

    def compute_homogeneity(self):
        if self.N == 0:
            return 0
        
        total = 0
        for j in range(self.K):
            ones = sum(1 for agent in self.schedule.agents if agent.x[j] == 1)
            zeros = self.N - ones
            total += max(ones, zeros) / self.N
        
        return total / self.K
    
    def replace_universal_clause(self):
        """
        Replace one random clause in the universal set C.
        Returns (clause_id, old_clause, new_clause).
        """
        u = random.randint(0, self.M - 1)
        old = self.C[u]
        new = self.random_clause()
        self.C[u] = new
        return (u, old, new)
        
    def step(self):
        """
        Execute one step of the model.
        Corresponds to NetLogo's go procedure.
        """
#        self.compute_comm_scale()
        # All agents perform learning step
        self.schedule.step()
        
        # Periodically replace a universal clause
        if self.schedule.steps % self.clause_interval == 0:
            cid, old_c, new_c = self.replace_universal_clause()
            # Note: clause network updates would go here if visualizing
        
        # Update metrics
        self.calc_performances()
        
        # Collect data
        self.datacollector.collect(self)
    
    def run(self, steps=None):
        """Run the model for specified number of steps (or until R)."""
        if steps is None:
            steps = self.R
        
        for _ in range(steps):
            if self.schedule.steps >= self.R:
                break
            self.step()
