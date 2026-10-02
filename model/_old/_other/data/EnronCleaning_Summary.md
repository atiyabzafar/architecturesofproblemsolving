# Enron Email Network: Data Cleaning Summary

**Dataset:** CMU Enron Email Corpus (May 2015 release)  
**Objective:** Build a clean, analysis-ready directed weighted graph of intra-Enron email communication, with node-level organisational metadata.

---

Step 4 and 5 are not part of the no data graph dataset. 

---
## Pipeline Overview

```
Raw maildir files
      │
      ▼
[1] Graph Creation         →  Raw DiGraph G
      │
      ▼
[2] Garbage Address Filter →  is_garbage_address()
      │
      ▼
[3] Alias Merging          →  buil_dalias3_map()  →  G_clean
      │
      ▼
[4] Distribution List Tags →  node_type ∈ {person, list}
      │
      ▼
[5] Node Enrichment        →  Hardin CSV  →  title, formal_level, notes
      │
      ▼
[6] Hierarchy Assignment   →  BFS from Kenneth Lay  →  bfs_level
      │
      ▼
Final: G_clean  (exported → EnronGraph.graphml / EnronGraph.edges)
```

---

## Step 1 : Graph Creation

- Explores the full CMU maildir corpus (`enron_mail_20150507/maildir/`).
- Parsed every `.` file with Python's `email.message_from_file()`.
- Extracted `From:` and `To:` headers; kept only **intra-Enron edges** (`@enron.com` → `@enron.com`).
- Self-loops (`sender == recipient`) dropped at ingestion time.
- All addresses lowercased and stripped.

| Metric | Value |
|---|---|
| Raw directed edges added | ~2,950,382 send events |
| Unique edges in raw `G` | 234,274 |
| Nodes in raw `G` | 22,174 |

---

## Step 2 : Garbage Address Filtering (`is_garbage_address`)

Removes structurally invalid or synthetic addresses that entered the graph as senders **or** recipients. Rules applied (in order):

| Rule                         | Example caught                             |
| ---------------------------- | ------------------------------------------ |
| Dot-prefix artifact          | `.mark@enron.com`                          |
| Quote-dot artifact           | `".frank"@enron.com`                       |
| Non-ASCII characters         | `?????.?????@enron.com`                    |
| CMU placeholder strings      | `noaddress@enron.com`, `unknown@enron.com` |
| Self-referential local parts | `mark.mark@enron.com`                      |
| Notes routing artifacts      | `oenronounacnrecipients...@enron.com`      |
| Manual removal               | `bodyshop@enron.com,...`                   |

> **Note:** A post-build pass removes residual garbage nodes that entered as *recipients* (not caught at edge-ingestion time):
> ```python
> garbage_nodes = [n for n in G_clean.nodes() if is_garbage_address(n)]
> G_clean.remove_nodes_from(garbage_nodes)
> ```

---

## Step 3 : Alias Merging (`buildalias3map`)

Enron staff had multiple email addresses due to name-reversal conventions and separator variants. This step canonicalises them.

**Detection logic:** For every node pair `(a, b)`, checks if `a`'s local part is a permutation of `b`'s local part (first.last ↔ last.first, dot ↔ underscore ↔ hyphen variants, middle-initial forms).

**392 alias pairs** detected and merged. All edges re-routed to the main address (the one with higher degree).

| Metric | Before | After |
|---|---|---|
| Nodes | 22,174 | 20,265 |
| Edges | 234,274 | 227,898 |

Special case: Kenneth Lay's aliases (`ken.lay`, `kenneth.lay`, `k.lay`, etc.) merged and confirmed as the canonical CEO node (`kenneth.lay@enron.com`, degree in=473, out=1,057).

---

## Step 4 : Distribution List Tagging

Mailing lists are present in the network as individuals in the graph. Nodes whose local part matches broadcast patterns were tagged rather than removed, so they can be filtered per-analysis:

```python
dl_pattern = re.compile(r'^(dl-|all\.|team\.|outlook\.team|body\.shop|...)')
node_type = 'list' if dl_pattern.match(local) else 'person'
```

| node_type | Count |
|---|---|
| `person` | **20,082** |
| `list` | **179** |
| **Total nodes** | **20,261** |
| **Total edges** | **227,891** |
>[!important] Note
>The list nodes were completely removed later. (Kept this in the documentation for historical book keeping)


---

## Step 5 : Node Enrichment (Hardin Employee List)
>This needs more work and verificaiton

Source: Hardin (2008) — *Journal of Statistics Education*, 161-entry Enron employee CSV  
(`EnronEmployees.csv`, columns: `name`, `title`, `notes`).

**Email address generation:** For each name, `name_to_candidate_emails()` generates all plausible email variants (first.last, last.first, f.last, first.l, middle-initial forms) and matches against graph nodes.

| Metric | Value |
|---|---|
| Employees in Hardin list | 161 |
| Matched to graph nodes | **148 (92%)** |
| Unmatched | 13 |

**Attributes set on matched nodes:**

| Attribute | Description |
|---|---|
| `title` | Job title from Hardin CSV (e.g. "Vice President") |
| `formal_level` | Nurek & Michalski (2012) integer hierarchy level (0–4) |
| `notes` | Supplementary notes from Hardin CSV |

**Formal level mapping (Nurek & Michalski scheme):**

| Level | Titles |
|---|---|
| 0 | CEO, Chairman, President |
| 1 | Managing Director, SVP, VP |
| 2 | Director, Manager, In House Lawyer |
| 3 | Specialist, Analyst, Trader, Associate, Employee |

**Manual overrides** added for executives missing from Hardin (e.g. Jeff Skilling → CEO).

---

## Step 6 : BFS Hierarchy Assignment

Organisational depth estimated via **undirected BFS from Kenneth Lay** (`kenneth.lay@enron.com`) as root.

```python
G_undirected = G_clean.to_undirected()
bfs_levels = nx.single_source_shortest_path_length(G_undirected, 'kenneth.lay@enron.com')
```

Unreachable nodes assigned `bfs_level = -1`.

| BFS Level | Node Count | Interpretation |
|---|---|---|
| -1 | 67 | Disconnected / isolated components |
| 0 | 1 | Kenneth Lay (root) |
| 1 | 1,439 | Direct correspondents of Lay |
| 2 | 11,607 | Two hops from Lay |
| 3 | 6,996 | Three hops |
| 4 | 144 | Four hops |
| 5+ | 7 | Peripheral nodes |

---

## Final Graph Statistics

| Property                            | Value       |
| ----------------------------------- | ----------- |
| Nodes (total)                       | **20,261**  |
| Person nodes                        | **20,261**  |
| Distribution list nodes             | **0**       |
| Directed edges                      | **227,891** |
| Enriched nodes (with title)         | **148**     |
| Disconnected nodes (bfs_level = -1) | **67**      |

**Exports:**
- `EnronGraph.edges` — plain edge list
- `EnronGraph.graphml` — full graph with all node attributes (title, formal_level, bfs_level, node_type, notes)

---
