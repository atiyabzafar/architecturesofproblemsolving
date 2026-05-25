# Old outline



Title: Architectures of Problem Solving: How Network Structures Shape Decision Making



1\. Introduction — Motivates the problem (decision-making under uncertainty, role of social networks), reviews related literature (organizational theory, opinion dynamics, Berekmeri et al., Hébert-Dufresne et al.), positions the paper's contribution, and previews findings.



2\. Methods



2.1 Model — Agents on a fixed directed network face a hidden Boolean satisfiability problem (AND/XOR clauses over K binary variables). They learn clauses via private observation and neighbor elicitation, then do local greedy repairs. Covers:

Entities and state (agents, decision variables, universal constraints, knowledge bases, violation counts, homogeneity)

Initialization

Per-period dynamics (4 steps: observe, update, elicit, update)

Environmental change (clause replacement every τ periods)

Termination

2.2 Network decomposability and performance — Rewiring algorithm that converts intra-community edges to inter-community edges while preserving edge count.

3\. Network Data — Describes 5 empirical networks: Congress Twitter, Company Emails, Political Blogs, Conference Attendance (MCL), CGS Interactions. Summary table.



4\. Results



4.1 Network performance — Compares violations and homogeneity across networks; finds dense/reciprocal networks perform best; discusses size effects on minimum violations.

4.2 Node heterogeneity — Centrality vs. violations (exponential decay); deep-dive into company email network using managerial hierarchy data.

4.3 Bridging clusters — Rewiring experiment on Political Blogs; even 0.05% rewiring yields sizable gains; violations plateau around 25% rewiring but homogeneity keeps improving.

5\. Conclusions and further research — Summarizes three main findings (topology drives performance; centrality predicts individual performance; modest cross-community bridging helps). Suggests extensions: behavioral heterogeneity, communication costs, endogenous network formation.



6\. Author contributions



7\. Acknowledgments



Appendices:



A: NetLogo interface

B: More on empirical networks (CGS visualization \& comparison, conference comparison, company email hierarchy deep-dive, comparison of rewiring methods, clause-every-step robustness check)

C: Interactive Shiny app description



# New outline



Title: Does Network Structure Matter for Distributed Decision Making?



1\. Introduction — Motivates the problem (decision-making under uncertainty, role of social networks), reviews related literature (organizational theory, opinion dynamics, Berekmeri et al., Hébert-Dufresne et al.), positions the paper's contribution, and previews findings. New finding: structure doesn't matter in the long run, it matters to withstand shocks.



2.1 Model — Agents on a fixed directed network face a hidden Boolean satisfiability problem (AND/XOR clauses over K binary variables). They learn clauses via private observation and neighbor elicitation, then do local greedy repairs. Covers:

Entities and state (agents, decision variables, universal constraints, knowledge bases, violation counts, homogeneity)

Initialization

Per-period dynamics (4 steps: observe, update, elicit, update)

Environmental change (clause replacement every τ periods)

**A shock every now and then?**

**(Should we still keep the rewiring exercise? what about the empirical data?)**



2.2 Solving the model, long run — in the long run network structure doesn't matter.

2.3 Solving the model, short run — a theory for transient behaviour? What determines convergence speed: network size, density, centralisation etc. What determines the distribution of cognitive labour



3\. Simulations confirming predictions: long-run behaviour (convergence; Centrality vs. violations), short-run behaviour (structures matters).



4 Conclusions and further research

4.1 Main findings: (1) Long-run collective performance in this class of problems is set by problem structure, not network architecture. (2) Network topology does shape transient dynamics, robustness to shocks, and the distribution of cognitive labour.

4.2 Discussion: implications, the importance of transient behaviour under periods of deep uncertainty.

4.3 Future directions: e.g. behavioural heterogeneity; endogenous communication.



5\. Author contributions



6\. Acknowledgments



Appendices:



Potentially the full proof of the long term theory, if not in the text.













