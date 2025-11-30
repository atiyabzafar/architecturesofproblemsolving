import numpy as np
import networkx as nx
from mesa import Agent, Model
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
        # Mesa 3.0+: super().__init__(model) is sufficient if unique_id is optional,
        # but passing unique_id is safer for graph mapping.
        super().__init__(model) 
        self.unique_id = unique_id # Explicitly set ID to match graph node index
        self.K = K
        
        # Initialize random binary assignment for K variables
        self.x = [random.choice([0, 1]) for _ in range(K)]
        
        # Knowledge base: List of clauses (FIFO)
        self.kb = [] 
        
        self.true_violations = 0  # Violations against global clause set C
        self.centr = 0.0          # Centrality measure
        self.in_neighbors = None  # Predecessors cache

    def add_clause_to_kb(self, clause):
        """Add clause to KB and maintain FIFO size limit."""
        self.kb.append(clause)
        M = self.model.M
        while len(self.kb) > M:
            self.kb.pop(0)

    def cache_neighbors(self):
        """Cache in-neighbors (predecessors) for communication."""
        if self.model.network.has_node(self.unique_id):
            self.in_neighbors = list(self.model.network.predecessors(self.unique_id))
        else:
            self.in_neighbors = []

    def clause_violated(self, clause, assign):
        """Check if AND/XOR clause is violated."""
        operator, var_indices = clause
        
        if operator == "AND":
            # ALL variables must be 1. Violated if ANY is 0.
            for var_idx in var_indices:
                if assign[var_idx - 1] == 0:
                    return True
            return False
            
        elif operator == "XOR":
            # Parity sum must be Odd (1). Violated if Even (0).
            parity_sum = sum(assign[var_idx - 1] for var_idx in var_indices)
            return (parity_sum % 2) != 1
            
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def violation_count(self, kb_list, assign):
        """Count how many clauses in list are violated."""
        return sum(1 for clause in kb_list if self.clause_violated(clause, assign))

    def step(self):
        """
        Agents learn by observing and communicating.
        """
        my_kb_set = set(self.kb)

        # 1) Private observation
        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)
            self.add_clause_to_kb(obs_clause)
            self.local_update_around(obs_clause)
            my_kb_set.add(obs_clause)

        # 2) Communication - proportional to in-degree
        if self.in_neighbors:
            comm_scale = self.model.comm_scale
            
            for nbr_id in self.in_neighbors:
                # Get edge weight
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get('weight', 1.0)
                
                # Calculate probability
                link_probability = min(1.0, comm_scale * link_weight)
                
                if random.random() < link_probability:
                    # Direct access via model.agent_list (O(1))
                    nbr_agent = self.model.agent_list[nbr_id]
                    
                    if nbr_agent.kb:
                        nbr_kb_set = set(nbr_agent.kb)
                        unknowns = nbr_kb_set - my_kb_set
                        
                        if unknowns:
                            cprime = random.choice(list(unknowns))
                            self.add_clause_to_kb(cprime)
                            self.local_update_around(cprime)
                            my_kb_set.add(cprime)

    def local_update_around(self, clause):
        """
        Local optimization: Try flipping variables involved in the new clause.
        """
        operator, var_indices = clause
        indices = list(var_indices)
        random.shuffle(indices) 
        
        related_kb = [cl for cl in self.kb if any(var in cl[1] for var in indices)]
        if not related_kb:
            return

        Vi_total = self.violation_count(related_kb, self.x)

        for j in indices:
            idx = j - 1
            old = self.x[idx]
            affected_kb = [cl for cl in related_kb if j in cl[1]]
            
            old_aff = self.violation_count(affected_kb, self.x)
            self.x[idx] = 1 - old
            new_aff = self.violation_count(affected_kb, self.x)
            
            Vnew_total = Vi_total - old_aff + new_aff
            
            if Vnew_total >= Vi_total:
                self.x[idx] = old # Revert
            else:
                Vi_total = Vnew_total # Update baseline

# ============================================================
# Model Class
# ============================================================

class ProblemSolvingModel(Model):
    """
    A model of collective problem-solving on a social network.
    Mesa 3.0+ Compatible (No RandomActivation).
    """
    def __init__(self,
                 N=50,
                 K=20,
                 alpha=4.2,
                 obs_prob=0.1,
                 clause_interval=10,
                 R=1000,
                 setup_source="generate",
                 file_path=None,
                 input_graph=None, 
                 type_network="Random",
                 connect_prob=0.1,
                 n_size=4,
                 rewire_prob=0.1,
                 min_deg=2,
                 nlayers=3,
                 intra_layer_connectance=0.5,
                 inter_layer_connectance=0.1,
                 random_layersize=False,
                 seed=None):
        
        super().__init__(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Parameters
        self.N = N
        self.K = K
        self.alpha = alpha
        self.M = round(alpha * K)
        self.obs_prob = obs_prob
        self.clause_interval = clause_interval
        self.R = R
        self.setup_source = setup_source
        self.file_path = file_path
        self.input_graph = input_graph
        self.type_network = type_network
        
        # Network params
        self.connect_prob = connect_prob
        self.n_size = n_size
        self.rewire_prob = rewire_prob
        self.min_deg = min_deg
        self.nlayers = nlayers
        self.intra_layer_connectance = intra_layer_connectance
        self.inter_layer_connectance = inter_layer_connectance
        self.random_layersize = random_layersize

        # Global State
        self.C = [] 
        self.avg_true_V = 0
        self.min_true_V = 0
        self.homogeneity = 0
        self.comm_scale = 0.0
        
        self.network = nx.DiGraph()
        self.agent_list = [] # For O(1) access by ID

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "avg_violations": lambda m: m.avg_true_V,
                "min_violations": lambda m: m.min_true_V,
                "homogeneity": lambda m: m.homogeneity
            },
            agent_reporters={
                "violations": "true_violations",
                "centrality": "centr",
                "kb_size": lambda a: len(a.kb)
            }
        )

        # Initialization
        self.generate_clauses()
        self.setup_network()
        
        if self.N < 5000:
            self.compute_centrality()
            self.calc_performances()

    def generate_clauses(self):
        self.C = []
        for _ in range(self.M):
            self.C.append(self.random_clause())

    def random_clause(self):
        L = 2
        indices = random.sample(range(1, self.K + 1), L)
        operator = "AND" if random.random() < 0.5 else "XOR"
        clause = (operator, tuple(indices))
        return self.canonicalise_clause(clause)

    def canonicalise_clause(self, clause):
        operator, variables = clause
        return (operator, tuple(sorted(variables)))

    def setup_network(self):
        """Setup network and agents."""
        # Reset
        self.agent_list = [None] * self.N
        self.network = nx.DiGraph()
        
        # 1. Generate Topology (Nodes 0..N-1)
        if self.setup_source == "generate":
            self.network.add_nodes_from(range(self.N))
            if self.type_network == "Random": self.setup_random_network()
            elif self.type_network == "Small World": self.setup_small_world_network()
            elif self.type_network == "Scale Free": self.setup_scale_free_network()
            elif self.type_network == "Hierarchical": self.setup_hierarchical_network()
                
        elif self.setup_source == "dataset":
            self.load_network_from_graphml(self.file_path)
            
        elif self.setup_source == "graph":
            self.load_network_from_graph(self.input_graph)

        # 2. Create Agents (Mesa 3.0: Just initialize, auto-adds to self.agents)
        # Only create if not already created by load functions
        if self.agent_list[0] is None:
            for i in range(self.N):
                # Pass i as unique_id explicitly
                agent = Person(unique_id=i, model=self, K=self.K)
                self.agent_list[i] = agent

        # 3. Finalize
        for agent in self.agent_list:
            agent.cache_neighbors()
            
        self.compute_comm_scale()

    # --- Network Generators ---

    def setup_random_network(self):
        for i in range(self.N):
            for j in range(self.N):
                if i != j and random.random() < self.connect_prob:
                    w = 0.0001 + random.random() * 0.9999
                    self.network.add_edge(i, j, weight=w)

    def setup_small_world_network(self):
        for i in range(self.N):
            for offset in range(1, self.n_size + 1):
                j = (i + offset) % self.N
                if random.random() < 0.5: self.network.add_edge(i, j, weight=1.0)
                else: self.network.add_edge(j, i, weight=1.0)
        
        edges = list(self.network.edges())
        for i, j in edges:
            if random.random() < self.rewire_prob:
                self.network.remove_edge(i, j)
                possible = [n for n in range(self.N) if n != i and not self.network.has_edge(i, n)]
                if possible:
                    k = random.choice(possible)
                    w = 0.0001 + random.random() * 0.9999
                    self.network.add_edge(i, k, weight=w)

    def setup_scale_free_network(self):
        for i in range(self.min_deg):
            for j in range(self.min_deg):
                if i != j: self.network.add_edge(i, j, weight=1.0)
        
        for i in range(self.min_deg, self.N):
            degrees = dict(self.network.in_degree())
            total = sum(degrees.values()) if degrees else 1
            targets = []
            for _ in range(self.min_deg):
                if not degrees:
                    target = random.choice(range(i))
                else:
                    ran = random.random() * total
                    acc = 0
                    target = list(degrees.keys())[-1]
                    for node, deg in degrees.items():
                        acc += deg
                        if ran <= acc:
                            target = node
                            break
                
                if target not in targets and target != i:
                    targets.append(target)
                    w = 0.0001 + random.random() * 0.9999
                    if random.random() < 0.5: self.network.add_edge(i, target, weight=w)
                    else: self.network.add_edge(target, i, weight=w)

    def setup_hierarchical_network(self):
        base = self.N // self.nlayers
        rem = self.N % self.nlayers
        layer_sizes = [base + 1 if i < rem else base for i in range(self.nlayers)]
        
        agents = list(range(self.N))
        random.shuffle(agents)
        layers = []
        start = 0
        for sz in layer_sizes:
            layers.append(agents[start:start+sz])
            start += sz
            
        for layer in layers:
            for i in layer:
                for j in layer:
                    if i != j and random.random() < self.intra_layer_connectance:
                        w = 0.0001 + random.random() * 0.9999
                        self.network.add_edge(i, j, weight=w)
        
        for a in range(len(layers)):
            for b in range(len(layers)):
                if a != b:
                    for i in layers[a]:
                        for j in layers[b]:
                            if random.random() < self.inter_layer_connectance:
                                w = 0.0001 + random.random() * 0.9999
                                self.network.add_edge(i, j, weight=w)

    def convert_undirected_to_asymmetric_directed(self, undirected_graph):
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(undirected_graph.nodes(data=True))
        for u, v in undirected_graph.edges():
            if u == v: continue
            if random.random() < 0.5: directed_graph.add_edge(u, v, weight=1.0)
            else: directed_graph.add_edge(v, u, weight=1.0)
        return directed_graph

    def load_network_from_graphml(self, filepath):
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        loaded_graph = nx.read_graphml(filepath)
        self.N = loaded_graph.number_of_nodes()
        
        if not loaded_graph.is_directed():
            loaded_graph = self.convert_undirected_to_asymmetric_directed(loaded_graph)
            
        node_ids = list(loaded_graph.nodes())
        node_mapping = {orig: i for i, orig in enumerate(node_ids)}
        
        self.network = nx.DiGraph()
        self.network.add_nodes_from(range(self.N))
        
        # Re-create agents
        self.agent_list = [None] * self.N
        # In Mesa 3.0, removing/recreating agents is done via self.agents
        # But here we just overwrite self.agent_list and let garbage collection handle old ones
        # self.agents is managed by Mesa automatically when Person() is called
        
        for i in range(self.N):
            agent = Person(i, self, self.K)
            self.agent_list[i] = agent
            
        for u, v in loaded_graph.edges():
            idx_u, idx_v = node_mapping[u], node_mapping[v]
            w = 0.0001 + random.random() * 0.9999
            self.network.add_edge(idx_u, idx_v, weight=w)

    def load_network_from_graph(self, input_graph):
        import copy
        loaded_graph = copy.deepcopy(input_graph)
        self.N = loaded_graph.number_of_nodes()
        
        if not loaded_graph.is_directed():
            loaded_graph = self.convert_undirected_to_asymmetric_directed(loaded_graph)
            
        node_ids = list(loaded_graph.nodes())
        node_mapping = {orig: i for i, orig in enumerate(node_ids)}
        
        self.network = nx.DiGraph()
        self.network.add_nodes_from(range(self.N))
        
        self.agent_list = [None] * self.N
        for i in range(self.N):
            agent = Person(i, self, self.K)
            self.agent_list[i] = agent
            
        for u, v in loaded_graph.edges():
            idx_u, idx_v = node_mapping[u], node_mapping[v]
            w = 0.0001 + random.random() * 0.9999
            self.network.add_edge(idx_u, idx_v, weight=w)

    # --- Metrics ---

    def compute_comm_scale(self):
        if self.network.number_of_edges() == 0:
            self.comm_scale = 0.0
            return
        in_strengths = []
        for i in range(self.N):
            str_i = sum(data.get('weight', 1.0) for _, _, data in self.network.in_edges(i, data=True))
            in_strengths.append(str_i)
        avg_flow = np.mean(in_strengths) if in_strengths else 0
        self.comm_scale = (1.0 / avg_flow) if avg_flow > 0 else 0.0

    def compute_centrality(self):
        in_strengths = []
        for i in range(self.N):
            str_i = sum(data.get('weight', 1.0) for _, _, data in self.network.in_edges(i, data=True))
            in_strengths.append(str_i)
        max_s = max(in_strengths) if in_strengths else 1.0
        max_s = max(max_s, 1.0)
        
        for i, strength in enumerate(in_strengths):
            if self.agent_list[i]:
                self.agent_list[i].centr = strength / max_s

    def calc_performances(self):
        for agent in self.agent_list:
            agent.true_violations = agent.violation_count(self.C, agent.x)
        violations = [a.true_violations for a in self.agent_list]
        self.avg_true_V = np.mean(violations) if violations else 0
        self.min_true_V = min(violations) if violations else 0
        self.homogeneity = self.compute_homogeneity()

    def compute_homogeneity(self):
        if self.N == 0: return 0
        total_consensus = 0
        for j in range(self.K):
            ones = sum(1 for a in self.agent_list if a.x[j] == 1)
            zeros = self.N - ones
            total_consensus += max(ones, zeros) / self.N
        return total_consensus / self.K

    def replace_universal_clause(self):
        u = random.randint(0, self.M - 1)
        old = self.C[u]
        new = self.random_clause()
        self.C[u] = new
        return (u, old, new)

    def step(self):
        """Execute one step using Mesa 3.0 shuffle_do."""
        # 1. Agents act (Learning)
        # self.agents is the new way to access all agents in Mesa 3.0+
        self.agents.shuffle_do("step")
        
        # 2. Environment change
        # self.steps is automatically incremented by Mesa's Model
        if self.steps % self.clause_interval == 0:
            self.replace_universal_clause()
            
        # 3. Metrics
        self.calc_performances()
        self.datacollector.collect(self)
