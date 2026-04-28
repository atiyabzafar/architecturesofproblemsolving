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
    def __init__(self, model, unique_id, K):
        # Mesa 3+: Agent.__init__ takes only `model`; unique_id is auto-assigned
        # starting from 1. We override it to keep 0..N-1 indexing that the
        # network code relies on.
        super().__init__(model)
#        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.agent_id = unique_id  # Internal ID reference
        self.K = K
        # Initialize random binary assignment for K variables
        self.x = [random.choice([0, 1]) for _ in range(K)]
        self.kb = [] # list of clauses, oldest first, knowledge base of clauses
        self.true_violations = 0  # Violations against global clause set C
        self.centr = 0.0  # Centrality measure
        self.in_neighbors = None
        self.local_obs_pool = []  # assigned by model after network setup

    def add_clause_to_kb(self, clause):
        """Add a clause to the KB and maintain size constraint M."""
        self.kb.append(clause)
        # bound-knowledgebase-size analogue
        #M = self.model.M
        #while len(self.kb) > M:
        #    self.kb.pop(0)
        #self.kb.append(clause)
        cap = self.model.kb_capacity
        while len(self.kb) > cap:
            self.kb.pop(0)
    
    def clause_violated(self, clause, assign):
        """Check if AND/XOR clause is violated."""
        operator, var_indices = clause
        
        if operator == "AND":
            # ALL variables must be 1 to satisfy
            for var_idx in var_indices:
                if assign[var_idx - 1] == 0:
                    return True
            return False
        
        elif operator == "XOR":
            # Matching Netlogo code: violated if NOT odd parity
            parity_sum = sum(assign[var_idx - 1] for var_idx in var_indices)
            return (parity_sum % 2) != 1
        
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def violation_count(self, kb_list, assign):
        """Count violated clauses."""
        return sum(1 for clause in kb_list if self.clause_violated(clause, assign))

    def cache_neighbors(self):
        """Pre-calculate incoming neighbors for performance."""
        self.in_neighbors = list(self.model.network.predecessors(self.unique_id))

    def step(self):
        """
        Agents learn by communicating and observing.
        Communication is proportional to in-degree.
        """
        my_kb_set = set(self.kb)

        # 1) Private observation
        if random.random() < self.model.obs_prob:
#            obs_clause = random.choice(self.model.C)
            obs_clause = random.choice(self.local_obs_pool)

            # Only update if the clause is actually new
            if obs_clause not in my_kb_set:
                self.add_clause_to_kb(obs_clause)
                self.local_update_around(obs_clause)
                my_kb_set.add(obs_clause)

        # 2) Communication - proportional to in-degree
        in_neighbors = self.in_neighbors
        if in_neighbors:
            comm_scale = self.model.comm_scale
            #comm_scale = 1.0
            for nbr_id in in_neighbors:
                # Get edge weight for probabilistic interaction
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get('weight', 1.0)
                link_probability = min(1.0, comm_scale * link_weight)
                
                if random.random() < link_probability:
                    nbr_agent = self.model.agent_list[nbr_id]  # Fast list lookup
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
        Local optimization (hill-climbing) around a newly learned clause.
        Minimizes violations within the relevant subset of the KB.
        """
        operator, var_indices = clause
        indices = list(var_indices)
        random.shuffle(indices) # Matches NetLogo stochasticity
        
        # Find clauses that mention any of these variables
        related_kb = [cl for cl in self.kb if any(var in cl[1] for var in indices)]
        
        if not related_kb:
            return
        
        # Baseline violations
        Vi_total = self.violation_count(related_kb, self.x)
        
        for j in indices:
            idx = j - 1
            old = self.x[idx]
            
            # Identify clauses specifically affected by flipping index j
            affected_kb = [cl for cl in related_kb if j in cl[1]]
            old_aff = self.violation_count(affected_kb, self.x)
            
            # Tentative flip
            self.x[idx] = 1 - old
            new_aff = self.violation_count(affected_kb, self.x)
            
            # Determine if flip improved consistency
            Vnew_total = Vi_total - old_aff + new_aff
            
            if Vnew_total >= Vi_total:
                self.x[idx] = old  # Revert if no improvement
            else:
                Vi_total = Vnew_total  # Update baseline

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
                 alpha=4,                 # Clause density (M = round(alpha * K))
                 obs_prob=0.1,            # Probability of private observation
                 clause_interval=10,      # Ticks between clause replacements
                 R=1000,                  # Run horizon (number of ticks)
                 setup_source="generate", # "generate", "dataset" or "graph"
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
                 kb_fraction=0.2,         # for fractional size of the knowledge base
                 local_obs_fraction=0.3,  # fraction of C each agent can directly observe
                 run_mode="basic",        # "basic" or "oscillatory or "binary"
                 seed=None):
        
        super().__init__()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
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
        self.run_mode = run_mode
        
        self.connect_prob = connect_prob
        self.n_size = n_size
        self.rewire_prob = rewire_prob
        self.min_deg = min_deg
        self.nlayers = nlayers
        self.intra_layer_connectance = intra_layer_connectance
        self.inter_layer_connectance = inter_layer_connectance
        self.random_layersize = random_layersize
        self.file_path = file_path
        
        self.C = []  # Universal clause set
        
        self.kb_fraction = kb_fraction
        self.kb_capacity = max(1, round(self.kb_fraction * self.M))
        self.local_obs_fraction = local_obs_fraction
        
        self.avg_true_V = 0
        self.min_true_V = 0
        self.homogeneity = 0
        self.comm_scale = 0.0
        self.network = nx.DiGraph()
        self.agent_list = []
        self.sinfreq=0.75
#        self.ANDbias=(1.0-np.sin(self.sinfreq*self.steps))/2.0
        self.ANDbias=1
        self.XORbias=1-self.ANDbias
        
        if self.setup_source == "dataset" and self.file_path is None:
            raise ValueError("file_path must be specified when setup_source='dataset'")

        # Data collector setup
        self.datacollector = DataCollector(
            model_reporters={
                "avg_violations": lambda m: m.avg_true_V,
                "min_violations": lambda m: m.min_true_V,
                "homogeneity": lambda m: m.homogeneity,
                "avg_centrality": lambda m: np.mean([a.centr for a in m.agents]) if len(m.agents) else 0,
            },
            agent_reporters={
                "violations": "true_violations",
                "centrality": "centr",
                "kb_size": lambda a: len(a.kb),
            }
        )
        
        self.generate_clauses()
        self.setup_network()
        if self.N < 5000:
            self.compute_centrality()
        self.calc_performances()
        
    def generate_clauses(self):
        """Generate M random clauses of length 2."""
        self.C = [self.random_clause() for _ in range(self.M)]

    def random_clause(self):
        """Create random AND or XOR clause over 2 variables."""
        indices = random.sample(range(1, self.K + 1), 2)
        operator = "AND" if random.random() < 0.5 else "XOR"
        clause = (operator, tuple(indices))
        return self.canonicalise_clause(clause)

    def clause_probability(self):
        """Get clause bias based on step count"""
        self.ANDbias=(1.0-np.sin(self.sinfreq*self.steps))/2.0
        self.XORbias=1-self.ANDbias

    def random_clause_biased(self):
        """Create random AND or XOR clause over 2 variables."""
        indices = random.sample(range(1, self.K + 1), 2)
        self.clause_probability()
        operator = "AND" if random.random() < self.ANDbias else "XOR"
        clause = (operator, tuple(indices))
        return self.canonicalise_clause(clause)

    def random_clause_binary(self):
        """Create random AND or XOR clause over 2 variables."""
        indices = random.sample(range(1, self.K + 1), 2)
        if random.random() < 0.5: # updates the values
            if self.ANDbias == 1:
                self.ANDbias = 0
            else:
                self.ANDbias = 1 
        self.XORbias = 1 - self.ANDbias
        operator = "AND" if random.random() < self.ANDbias else "XOR"
        clause = (operator, tuple(indices))
        return self.canonicalise_clause(clause)



    def canonicalise_clause(self, clause):
        """Sort variables within clause for deterministic indexing."""
        operator, variables = clause
        return (operator, tuple(sorted(variables)))

    def setup_network(self):
        """Initialize network topology based on parameters."""
        if self.setup_source == "generate":
            self.agent_list = [None] * self.N
            for i in range(self.N):
                agent = Person(self, i, self.K)
                self.agent_list[i] = agent

            if self.type_network == "Random":
                self.setup_random_network()
            elif self.type_network == "Small World":
                self.setup_small_world_network()
            elif self.type_network == "Scale Free":
                self.setup_scale_free_network()
            elif self.type_network == "Hierarchical":
                self.setup_hierarchical_network()
            else:
                raise ValueError(f"Unknown type_network: {self.type_network}")
                
        elif self.setup_source == "dataset":
            self.load_network_from_graphml(self.file_path)
        elif self.setup_source == "graph":
            if self.input_graph is None:
                raise ValueError("input_graph must be provided when setup_source='graph'")
            self.load_network_from_graph(self.input_graph)

        pool_size = max(1, round(self.local_obs_fraction * self.M))
        
        for agent in self.agents:
            agent.cache_neighbors()
            agent.local_obs_pool = random.sample(self.C, pool_size)            

        self.compute_comm_scale()

    def setup_random_network(self):
        self.network.add_nodes_from(range(self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j and random.random() < self.connect_prob:
                    weight = 0.0001 + random.random() * 0.9999
                    self.network.add_edge(i, j, weight=weight)

    def setup_small_world_network(self):
        self.network.add_nodes_from(range(self.N))
        for i in range(self.N):
            for offset in range(1, self.n_size + 1):
                j = (i + offset) % self.N
                self.network.add_edge(i, j, weight=1.0) if random.random() < 0.5 else self.network.add_edge(j, i, weight=1.0)
        
        edges = list(self.network.edges())
        for i, j in edges:
            if random.random() < self.rewire_prob:
                self.network.remove_edge(i, j)
                candidates = [n for n in range(self.N) if n != i and not self.network.has_edge(i, n)]
                if candidates:
                    k = random.choice(candidates)
                    self.network.add_edge(i, k, weight=0.0001 + random.random() * 0.9999)

    def setup_scale_free_network(self):
        self.network.add_nodes_from(range(self.N))
        for i in range(self.min_deg):
            for j in range(self.min_deg):
                if i != j: self.network.add_edge(i, j, weight=1.0)
        
        for i in range(self.min_deg, self.N):
            degrees = dict(self.network.in_degree())
            total = sum(degrees.values()) if degrees else self.min_deg
            targets = []
            while len(targets) < self.min_deg and len(targets) < i:
                ran = random.random() * total
                acc = 0
                target = None
                for node, deg in degrees.items():
                    acc += deg
                    if ran <= acc:
                        target = node
                        break
                if target is None: target = list(degrees.keys())[-1]
                if target not in targets:
                    targets.append(target)
                    w = 0.0001 + random.random() * 0.9999
                    self.network.add_edge(i, target, weight=w) if random.random() < 0.5 else self.network.add_edge(target, i, weight=w)

    def setup_hierarchical_network(self):
        self.network.add_nodes_from(range(self.N))
        base, rem = divmod(self.N, self.nlayers)
        sizes = [base + (1 if i < rem else 0) for i in range(self.nlayers)]
        if self.random_layersize: random.shuffle(sizes)
        
        agents = list(range(self.N))
        random.shuffle(agents)
        layers = []
        curr = 0
        for s in sizes:
            layers.append(agents[curr:curr+s])
            curr += s
            
        for layer in layers:
            for i in layer:
                for j in layer:
                    if i != j and random.random() < self.intra_layer_connectance:
                        self.network.add_edge(i, j, weight=0.0001 + random.random() * 0.9999)
        
        for a in range(len(layers)):
            for b in range(len(layers)):
                if a == b: continue
                for i in layers[a]:
                    for j in layers[b]:
                        if random.random() < self.inter_layer_connectance:
                            self.network.add_edge(i, j, weight=0.0001 + random.random() * 0.9999)

    def convert_undirected_to_asymmetric_directed(self, undirected_graph):
        directed = nx.DiGraph()
        directed.add_nodes_from(undirected_graph.nodes(data=True))
        for u, v in undirected_graph.edges():
            if u == v: continue
            if random.random() < 0.5: directed.add_edge(u, v, weight=1.0)
            else: directed.add_edge(v, u, weight=1.0)
        return directed

    def load_network_from_graphml(self, filepath):
        loaded = nx.read_graphml(filepath)
        self.N = loaded.number_of_nodes()
        if not loaded.is_directed(): loaded = self.convert_undirected_to_asymmetric_directed(loaded)
        mapping = {orig: i for i, orig in enumerate(loaded.nodes())}
        self.network = nx.DiGraph()
        self.network.add_nodes_from(range(self.N))
        self.agent_list = [Person(self, i, self.K) for i in range(self.N)]
        weights = 0.0001 + np.random.rand(loaded.number_of_edges()) * 0.9999
        edges = [(mapping[u], mapping[v], {"weight": w}) for (u, v), w in zip(loaded.edges(), weights)]
        self.network.add_edges_from(edges)

    def load_network_from_graph(self, input_graph):
        import copy
        loaded = copy.deepcopy(input_graph)
        self.N = loaded.number_of_nodes()
        if not loaded.is_directed(): loaded = self.convert_undirected_to_asymmetric_directed(loaded)
        mapping = {orig: i for i, orig in enumerate(loaded.nodes())}
        self.network = nx.DiGraph()
        self.network.add_nodes_from(range(self.N))
        self.agent_list = [Person(self, i, self.K) for i in range(self.N)]
        weights = 0.0001 + np.random.rand(loaded.number_of_edges()) * 0.9999
        edges = [(mapping[u], mapping[v], {"weight": w}) for (u, v), w in zip(loaded.edges(), weights)]
        self.network.add_edges_from(edges)

    def compute_centrality(self):
        uses_weights = any(d.get("weight", 1.0) != 1.0 for _, _, d in self.network.edges(data=True))
        if uses_weights:
            strengths = [sum(d.get('weight', 1.0) for _, _, d in self.network.in_edges(i, data=True)) for i in range(self.N)]
            max_s = max(strengths) if strengths else 1.0
            for i, s in enumerate(strengths): self.agent_list[i].centr = s / max_s
        else:
            degrees = [self.network.in_degree(i) for i in range(self.N)]
            max_d = max(degrees) if degrees else 1.0
            for i, d in enumerate(degrees): self.agent_list[i].centr = d / max_d

    def compute_comm_scale(self):
        uses_weights = any(d.get("weight", 1.0) != 1.0 for _, _, d in self.network.edges(data=True))
        if uses_weights:
            inflow = np.mean([sum(d.get('weight', 1.0) for _, _, d in self.network.in_edges(i, data=True)) for i in range(self.N)])
        else:
            inflow = np.mean([self.network.in_degree(i) for i in range(self.N)])
        self.comm_scale = (1.0 / inflow) if inflow > 0 else 0.0

    def calc_performances(self):
        for a in self.agents: a.true_violations = a.violation_count(self.C, a.x)
        v = [a.true_violations for a in self.agents]
        self.avg_true_V, self.min_true_V = np.mean(v) if v else 0, min(v) if v else 0
        self.homogeneity = self.compute_homogeneity()

    def compute_homogeneity(self):
        if self.N == 0: return 0
        total = sum(max(sum(1 for a in self.agents if a.x[j] == 1), self.N - sum(1 for a in self.agents if a.x[j] == 1)) / self.N for j in range(self.K))
        return total / self.K

    def replace_universal_clause(self):
        u = random.randint(0, self.M - 1)
        if self.run_mode == "basic":
            self.C[u] = self.random_clause()
        elif self.run_mode == "oscillatory":
            self.C[u] = self.random_clause_biased()
        elif self.run_mode == "binary":
            self.C[u] = self.random_clause_binary()
        else:
            raise ValueError(f"Unknown run_mode: {self.run_mode}")


    def step(self):
        self.agents.shuffle_do("step")
        if self.steps % self.clause_interval == 0: self.replace_universal_clause()
        self.calc_performances()
        self.datacollector.collect(self)
