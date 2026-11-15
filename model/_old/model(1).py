import numpy as np
import networkx as nx
from mesa import Agent, Model
from mesa.datacollection import DataCollector
import random


class Person(Agent):
    """Agent with binary assignment and clause knowledge base."""

    def __init__(self, model, unique_id, K):
        super().__init__(model)
        self.unique_id = unique_id
        self.agent_id = unique_id
        self.K = K
        self.x = [random.choice([0, 1]) for _ in range(K)]
        self.kb = []
        self.true_violations = 0
        self.centr = 0.0
        self.in_neighbors = None

    def add_clause_to_kb(self, clause):
        self.kb.append(clause)
        M = self.model.M
        while len(self.kb) > M:
            self.kb.pop(0)

    def clause_violated(self, clause, assign):
        operator, var_indices = clause
        if operator == "AND":
            for var_idx in var_indices:
                if assign[var_idx - 1] == 0:
                    return True
            return False
        elif operator == "XOR":
            parity_sum = sum(assign[var_idx - 1] for var_idx in var_indices)
            return (parity_sum % 2) != 1
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def violation_count(self, kb_list, assign):
        return sum(1 for clause in kb_list if self.clause_violated(clause, assign))

    def cache_neighbors(self):
        self.in_neighbors = list(self.model.network.predecessors(self.unique_id))

    def step(self):
        my_kb_set = set(self.kb)

        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)
            if obs_clause not in my_kb_set:
                self.add_clause_to_kb(obs_clause)
                self.local_update_around(obs_clause)
                my_kb_set.add(obs_clause)

        in_neighbors = self.in_neighbors
        if in_neighbors:
            comm_scale = self.model.comm_scale
            for nbr_id in in_neighbors:
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get("weight", 1.0)
                link_probability = min(1.0, comm_scale * link_weight)
                if random.random() < link_probability:
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
                self.x[idx] = old
            else:
                Vi_total = Vnew_total


class ProblemSolvingModel(Model):
    def __init__(
        self,
        N=50,
        K=20,
        alpha=4,
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
        seed=None,
    ):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.N = N
        self.K = K
        self.alpha = alpha
        self.M = round(alpha * K)
        self.obs_prob = obs_prob
        self.clause_interval = clause_interval
        self.R = R
        self.setup_source = setup_source
        self.type_network = type_network
        self.input_graph = input_graph

        self.connect_prob = connect_prob
        self.n_size = n_size
        self.rewire_prob = rewire_prob
        self.min_deg = min_deg
        self.nlayers = nlayers
        self.intra_layer_connectance = intra_layer_connectance
        self.inter_layer_connectance = inter_layer_connectance
        self.random_layersize = random_layersize
        self.file_path = file_path

        self.C = []
        self.avg_true_V = 0
        self.min_true_V = 0
        self.homogeneity = 0
        self.comm_scale = 0.0
        self.network = nx.DiGraph()
        self.agent_list = []

        if self.setup_source == "dataset" and self.file_path is None:
            raise ValueError("file_path must be specified when setup_source='dataset'")

        self.datacollector = DataCollector(
            model_reporters={
                "avg_violations": lambda m: m.avg_true_V,
                "min_violations": lambda m: m.min_true_V,
                "homogeneity": lambda m: m.homogeneity,
                "avg_centrality": lambda m: np.mean([a.centr for a in m.agents]) if len(self.agents) else 0,
            },
            agent_reporters={
                "violations": "true_violations",
                "centrality": "centr",
                "kb_size": lambda a: len(a.kb),
            },
        )

        self.generate_clauses()
        self.setup_network()
        if self.N < 5000:
            self.compute_centrality()
        self.calc_performances()

    def generate_clauses(self):
        self.C = [self.random_clause() for _ in range(self.M)]

    def random_clause(self):
        indices = random.sample(range(1, self.K + 1), 2)
        operator = "AND" if random.random() < 0.5 else "XOR"
        clause = (operator, tuple(indices))
        return self.canonicalise_clause(clause)

    def canonicalise_clause(self, clause):
        operator, variables = clause
        return (operator, tuple(sorted(variables)))

    def setup_network(self):
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
        else:
            raise ValueError(f"Unknown setup_source: {self.setup_source}")

        for agent in self.agents:
            agent.cache_neighbors()
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
                if random.random() < 0.5:
                    self.network.add_edge(i, j, weight=1.0)
                else:
                    self.network.add_edge(j, i, weight=1.0)
        edges_to_rewire = list(self.network.edges())
        for i, j in edges_to_rewire:
            if random.random() < self.rewire_prob:
                self.network.remove_edge(i, j)
                candidates = [n for n in range(self.N) if n != i and not self.network.has_edge(i, n)]
                if candidates:
                    k = random.choice(candidates)
                    weight = 0.0001 + random.random() * 0.9999
                    self.network.add_edge(i, k, weight=weight)

    def setup_scale_free_network(self):
        self.network.add_nodes_from(range(self.N))
        for i in range(self.min_deg):
            for j in range(self.min_deg):
                if i != j:
                    self.network.add_edge(i, j, weight=1.0)

        for i in range(self.min_deg, self.N):
            degrees = dict(self.network.in_degree())
            total = sum(degrees.values()) if degrees else self.min_deg
            targets = []
            while len(targets) < self.min_deg and len(targets) < i:
                if not degrees or total == 0:
                    target = random.choice(range(i))
                else:
                    ran = random.random() * total
                    acc = 0
                    target = None
                    for node, deg in degrees.items():
                        acc += deg
                        if ran <= acc:
                            target = node
                            break
                    if target is None:
                        target = list(degrees.keys())[-1]
                if target not in targets:
                    targets.append(target)
                    weight = 0.0001 + random.random() * 0.9999
                    if random.random() < 0.5:
                        self.network.add_edge(i, target, weight=weight)
                    else:
                        self.network.add_edge(target, i, weight=weight)

    def setup_hierarchical_network(self):
        self.network.add_nodes_from(range(self.N))
        if self.random_layersize:
            layer_sizes = [0] * self.nlayers
            remaining = self.N
            while remaining > 0:
                i = random.randint(0, self.nlayers - 1)
                layer_sizes[i] += 1
                remaining -= 1
        else:
            base = self.N // self.nlayers
            rem = self.N - base * self.nlayers
            layer_sizes = [base] * self.nlayers
            for j in range(rem):
                layer_sizes[j] += 1

        agents = list(range(self.N))
        random.shuffle(agents)
        layers = []
        start = 0
        for sz in layer_sizes:
            layer = agents[start:start + sz]
            layers.append(layer)
            start += sz

        for layer in layers:
            for i in layer:
                for j in layer:
                    if i != j and random.random() < self.intra_layer_connectance:
                        weight = 0.0001 + random.random() * 0.9999
                        self.network.add_edge(i, j, weight=weight)

        for a in range(len(layers)):
            for b in range(len(layers)):
                if a != b:
                    for i in layers[a]:
                        for j in layers[b]:
                            if random.random() < self.inter_layer_connectance:
                                weight = 0.0001 + random.random() * 0.9999
                                self.network.add_edge(i, j, weight=weight)

    def convert_undirected_to_asymmetric_directed(self, undirected_graph):
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(undirected_graph.nodes(data=True))
        for u, v in undirected_graph.edges():
            if u == v:
                continue
            if random.random() < 0.5:
                directed_graph.add_edge(u, v, weight=1.0)
            else:
                directed_graph.add_edge(v, u, weight=1.0)
        return directed_graph

    def load_network_from_graphml(self, filepath):
        loaded_graph = nx.read_graphml(filepath)
        self.N = loaded_graph.number_of_nodes()
        if not loaded_graph.is_directed():
            loaded_graph = self.convert_undirected_to_asymmetric_directed(loaded_graph)
        original_node_ids = list(loaded_graph.nodes())
        node_mapping = {orig_id: i for i, orig_id in enumerate(original_node_ids)}
        self.network = nx.DiGraph()
        self.network.add_nodes_from(range(self.N))
        self.agent_list = [Person(self, i, self.K) for i in range(self.N)]
        edge_list = [(node_mapping[u], node_mapping[v]) for u, v in loaded_graph.edges()]
        random_weights = 0.0001 + np.random.rand(len(edge_list)) * 0.9999
        edge_tuples = [(u, v, {"weight": w}) for (u, v), w in zip(edge_list, random_weights)]
        self.network.add_edges_from(edge_tuples)

    def load_network_from_graph(self, input_graph):
        import copy
        loaded_graph = copy.deepcopy(input_graph)
        self.N = loaded_graph.number_of_nodes()
        self.network = nx.DiGraph()
        if not loaded_graph.is_directed():
            loaded_graph = self.convert_undirected_to_asymmetric_directed(loaded_graph)
        original_node_ids = list(loaded_graph.nodes())
        node_mapping = {orig_id: i for i, orig_id in enumerate(original_node_ids)}
        self.network.add_nodes_from(range(self.N))
        self.agent_list = [None] * self.N
        for i in range(self.N):
            self.agent_list[i] = Person(self, i, self.K)
        edge_list = [(node_mapping[u], node_mapping[v]) for u, v in loaded_graph.edges()]
        random_weights = 0.0001 + np.random.rand(len(edge_list)) * 0.9999
        edge_tuples = [(u, v, {"weight": w}) for (u, v), w in zip(edge_list, random_weights)]
        self.network.add_edges_from(edge_tuples)

    def compute_centrality(self):
        uses_weights = any(data.get("weight", 1.0) != 1.0 for _, _, data in self.network.edges(data=True))
        if uses_weights:
            in_strengths = []
            for agent_id in range(self.N):
                in_edges = self.network.in_edges(agent_id, data=True)
                in_strength = sum(data.get("weight", 1.0) for _, _, data in in_edges)
                in_strengths.append(in_strength)
            max_strength = max(in_strengths) if in_strengths else 1.0
            max_strength = max(max_strength, 1.0)
            for agent_id, in_strength in enumerate(in_strengths):
                self.agent_list[agent_id].centr = in_strength / max_strength
        else:
            in_degrees = [self.network.in_degree(agent_id) for agent_id in range(self.N)]
            max_degree = max(in_degrees) if in_degrees else 1.0
            max_degree = max(max_degree, 1.0)
            for agent_id, in_degree in enumerate(in_degrees):
                self.agent_list[agent_id].centr = in_degree / max_degree

    def compute_comm_scale(self):
        uses_weights = any(data.get("weight", 1.0) != 1.0 for _, _, data in self.network.edges(data=True))
        if uses_weights:
            in_strengths = []
            for agent_id in range(self.N):
                in_edges = self.network.in_edges(agent_id, data=True)
                in_strength = sum(data.get("weight", 1.0) for _, _, data in in_edges)
                in_strengths.append(in_strength)
            avg_base_inflow = np.mean(in_strengths) if in_strengths else 0
        else:
            in_degrees = [self.network.in_degree(agent_id) for agent_id in range(self.N)]
            avg_base_inflow = np.mean(in_degrees) if in_degrees else 0
        self.comm_scale = (1.0 / avg_base_inflow) if avg_base_inflow > 0 else 0.0

    def calc_performances(self):
        for agent in self.agents:
            agent.true_violations = agent.violation_count(self.C, agent.x)
        violations = [agent.true_violations for agent in self.agents]
        self.avg_true_V = np.mean(violations) if violations else 0
        self.min_true_V = min(violations) if violations else 0
        self.homogeneity = self.compute_homogeneity()

    def compute_homogeneity(self):
        if self.N == 0:
            return 0
        total = 0
        for j in range(self.K):
            ones = sum(1 for agent in self.agents if agent.x[j] == 1)
            zeros = self.N - ones
            total += max(ones, zeros) / self.N
        return total / self.K

    def replace_universal_clause(self):
        u = random.randint(0, self.M - 1)
        old = self.C[u]
        new = self.random_clause()
        self.C[u] = new
        return (u, old, new)

    def step(self):
        self.agents.shuffle_do("step")
        if self.steps % self.clause_interval == 0:
            self.replace_universal_clause()
        self.calc_performances()
        self.datacollector.collect(self)

    def run(self, steps=None):
        if steps is None:
            steps = self.R
        for _ in range(steps):
            if self.steps >= self.R:
                break
            self.step()
