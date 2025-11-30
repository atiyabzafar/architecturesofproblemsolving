from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import gravis as gv
import numpy as np
from model import ProblemSolvingModel
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import asyncio
import tempfile  
import os

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            /* Using the color palette https://coolors.co/606c38-283618-fefae0-dda15e-bc6c25 */
            body {
                font-family: 'Arial', sans-serif;
                background: #fefae0
            }
            
            .header-title {
                background: linear-gradient(135deg, #606c38 0%, #283618 100%);
                color: white;
                padding: 40px;
                border-radius: 15px;
                margin-bottom: 30px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .header-title h1 {
                margin: 0;
                font-size: 32px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            
            .header-title p {
                margin: 10px 0 0 0;
                font-size: 14px;
                opacity: 0.9;
            }
            
            .network-stat-box {
                background: linear-gradient(135deg, #606c38 0%, #283618 100%);
                color: white;
                padding: 25px;
                border-radius: 12px;
                margin: 10px 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
            }
            
            .network-stat-box:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
            }
            
            .network-stat-number {
                font-size: 36px;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .network-stat-label {
                font-size: 13px;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .btn-setup {
                background: linear-gradient(90deg, #FF930F, #FFF95B); /*https://coolors.co/gradient/ff930f-fff95b */
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .btn-setup:hover {
                box-shadow: 0 6px 12px rgba(245, 87, 108, 0.4);
                transform: translateY(-2px);
            }
            
            .btn-run {
                background: linear-gradient(90deg, #FF5858, #FFC8C8); /*https://coolors.co/gradient/ff5858-ffc8c8*/
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .btn-run:hover {
                box-shadow: 0 6px 12px rgba(79, 172, 254, 0.4);
                transform: translateY(-2px);
            }
            
            .btn-stop {
                background: linear-gradient(90deg, #BC1B68, #D3989B);/*https://coolors.co/gradient/bc1b68-d3989b*/
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .btn-stop:hover {
                box-shadow: 0 6px 12px rgba(250, 112, 154, 0.4);
                transform: translateY(-2px);
            }
            
            .status-badge {
                display: inline-block;
                padding: 12px 24px;
                border-radius: 20px;
                font-weight: bold;
                margin: 10px 0;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .status-ready {
                background: #e3f2fd;
                color: #1976d2;
                border: 2px solid #1976d2;
            }
            
            .status-running {
                background: #fff3e0;
                color: #e65100;
                border: 2px solid #e65100;
                animation: pulse 1s infinite;
            }
            
            .status-complete {
                background: #e8f5e9;
                color: #2e7d32;
                border: 2px solid #2e7d32;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            .progress-bar-container {
                background: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
                height: 10px;
                margin: 15px 0;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .progress-bar-fill {
                height: 100%;
                background: linear-gradient(90deg, #606c38 0%, #764ba2 100%);
                transition: width 0.3s ease;
                border-radius: 10px;
            }
            
            .sidebar-section-title {
                color: #606c38;
                font-weight: bold;
                font-size: 16px;
                margin-top: 10px;
                margin-bottom: 10px;
                border-bottom: 2px solid #606c38;
                padding-bottom: 10px;
            }
            .network-container {
                width: 100% !important;
                height: 700px !important;
                min-height: 700px !important;
                overflow: auto !important;
                border-radius: 8px;
                background: #fefae0;
                
                
            }
            
            /* Force Gravis inner divs to respect height */
            .network-container > div {
                height: 100% !important;
                min-height: 700px !important;
            }
            
            /* Force the graph div specifically */
            div[id*="-graph-div"] {
                height: 650px !important;
                min-height: 650px !important;
            }
            /* Colorbar positioning */
            .network-container {
                position: relative !important;
            }
            
            .network-colorbar {
                    position: absolute !important;
                    top: 70px !important;           /* Offset from top */
                    right: 30px !important;
                    /*background: rgba(255, 255, 255, 0.95) !important;*/
                    padding: 15px 12px !important;
                    border-radius: 8px !important;
                    box-shadow: 0 3px 12px rgba(0,0,0,0.2) !important;
                    z-index: 10000 !important;
                    pointer-events: none !important;
                    max-height: 620px !important;   /* Fits within 700px container */
                }
            .irs.irs--shiny .irs-single { /* square with number */
                background-color: #606c38;
                color: white
            }
            .irs-bar.irs-bar--single { /* line */
                background-color: #606c38;
            }
            .irs-handle.single { /* circle */
                background-color: #606c38;
            }
            .form-control {
                background-color: #fefae0;          
            }
            .form-select {
                background-color: #fefae0;          
            }
            .accordion-item{
                background-color: #fefae0;
            }
            .accordion-button collapsed{
                background-color: #fefae0;
            }
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #606c38;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

        """)
    ),
    
    # Header with gradient background
    ui.tags.div(
        ui.row(
            ui.column(6,
            ui.tags.h1("Architectures of Problem Solving"),
            ui.tags.p("An educational interactive app for an agent-based model created to study how different network topologies affect decision making."),
            ui.tags.p("Click on 'How to use?' tab below for information about the model and instructions for using the app"),

            class_="header-title"
            ),
            ui.column(3,
                ui.tags.h3("Configuration", style="color: #606c38; font-weight: bold; margin-top: 0;"),
                ui.hr(),
                ui.row(
                    ui.column(4, ui.input_action_button("setup_model", "Setup", class_="btn-setup", style="width: 100%; margin-bottom: 10px;")),
                    ui.column(4, ui.input_action_button("run_sim", "Run", class_="btn-run", style="width: 100%; margin-bottom: 10px;")),
                    ui.column(4, ui.input_action_button("stop_sim", "Stop", class_="btn-stop", style="width: 100%;")),
                ),
               # Status and Progress Display
               ui.row(
                    ui.column(4,ui.output_ui("status")),
                    ui.column(8,ui.output_ui("progress_bar")),
                )
            ),
            ui.column(3,
                ui.tags.div("Dataset", class_="sidebar-section-title"),
                ui.input_radio_buttons(
                    "setup_source",
                    "Source:",
                    choices={
                        "generate": "Generate Network",
                        "dataset": "Load GraphML",
                        "loaded": "Select from preloaded networks"
                    },
                    selected="generate"
                ),
                ui.panel_conditional(
                    "input.setup_source == 'loaded'",
                    ui.input_select(
                        "loaded_network",
                        "Type",
                        choices={
                            "Email": "Email Network",
                            "Pol Blogs": "Political Blogs",
                            "Congress": "Congress Twitter",
                            "Conference1": "Conference (MCL)",
                            "Conference2": "Conference (AES)",
                            "Conference3": "Conference (CMC)",
                            "Conference4": "Conference (TDA)",
                            "2024CGS": "2024 CGS (Arch)",
                            "2025CGS": "2025 CGS (Arch)",
                        },
                        selected="Conference"
                    ),
                ),
                ui.panel_conditional(
                    "input.setup_source == 'dataset'",
                    ui.input_file(
                        "graphml_file",
                        "Upload GraphML File:",
                        accept=[".graphml", ".xml"],
                        placeholder="Select .graphml file"
                    ),
                    ui.output_text_verbatim("file_info")
                ),
            )
        )
            # Setup and Run Buttons
    ),
    
    #layout of sidebar
    ui.layout_sidebar(
        ui.sidebar(
        ui.accordion(
            # Panel 1: Basic Parameters
            ui.accordion_panel(
                "Basic Parameters",
                ui.row(
                    ui.column(6, ui.input_slider("N_slider", "Agents", min=10, max=500, value=50, step=10)),
                    ui.column(6, ui.input_numeric("N_numeric", "Input", value=50, min=10, max=500)),
                ),
                ui.row(
                    ui.column(6, ui.input_slider("K_slider", "Variables", min=5, max=100, value=50, step=5)),
                    ui.column(6, ui.input_numeric("K_numeric", "Input", value=50, min=5, max=100)),
                ),
                ui.row(
                    ui.column(12, 
                        ui.tags.div(
                            ui.tags.span("Density α ", style="font-weight: bold;"),
                            ui.tags.span("ℹ️", 
                                title="Constraint density = (# of clauses) / (# of variables). Higher values create harder problems.",
                                style="cursor: help; color: #606c38;"
                            )
                        )
                    )
                ),
                ui.row(
                    ui.column(6, ui.input_slider("alpha_slider", "", min=0.1, max=6.0, value=2.0, step=0.1)),
                    ui.column(6, ui.input_numeric("alpha_numeric", "", value=2.0, min=0.1, max=6.0, step=0.1)),
                ),
                ui.row(
                    ui.column(12, 
                        ui.tags.div(
                            ui.tags.span("Obs Prob", style="font-weight: bold;"),
                            ui.tags.span("ℹ️", 
                                title="Probability of obtaining information from the surrounding for an agent",
                                style="cursor: help; color: #606c38;"
                            )
                        )
                    )
                ),   
                ui.row(
                    ui.column(6, ui.input_slider("obs_prob_slider", "", min=0.0, max=1.0, value=0.01, step=0.01)),
                    ui.column(6, ui.input_numeric("obs_prob_numeric", "", value=0.01, min=0.0, max=1.0, step=0.01)),
                ),
                ui.row(
                    ui.column(12, 
                        ui.tags.div(
                            ui.tags.span("Clause Interval", style="font-weight: bold;"),
                            ui.tags.span("ℹ️", 
                                title="Every x step the clauses changes. (constraints)",
                                style="cursor: help; color: #606c38;"
                            )
                        )
                    )
                ),
                ui.row(
                    ui.column(6, ui.input_slider("clause_interval_slider", "", min=1, max=500, value=10, step=1)),
                    ui.column(6, ui.input_numeric("clause_interval_numeric", "", value=10, min=1, max=500)),
                ),
                ui.row(
                    ui.column(6, ui.input_slider("R_slider", "Run Horizon", min=100, max=2000, value=500, step=100)),
                    ui.column(6, ui.input_numeric("R_numeric", "Input", value=500, min=100, max=2000, step=100)),
                ),
                ui.input_numeric("seed", "Random Seed", value=42),
            ),

            # Panel 2: Network Settings
            ui.accordion_panel(
                "Network",
                ui.input_select(
                    "network_type",
                    "Type",
                    choices={
                        "Random": "Random (Erdos Renyi)",
                        "Small World": "Small World (Watts-Strogatz)",
                        "Scale Free": "Scale Free (Barabasi Albert)",
                        "Hierarchical": "Hierarchical (Own Method)"
                    },
                    selected="Random"
                ),
                # Conditional Network Parameters
                ui.panel_conditional(
                    "input.network_type === 'Random'",
                    ui.row(
                        ui.column(6, ui.input_slider("connect_prob_slider", "Conn Prob", min=0.0, max=1.0, value=0.1, step=0.05)),
                        ui.column(6, ui.input_numeric("connect_prob_numeric", "Input", value=0.1, min=0.0, max=1.0, step=0.05)),
                    ),
                ),
                ui.panel_conditional(
                    "input.network_type === 'Small World'",
                    ui.row(
                        ui.column(6, ui.input_slider("n_size_slider", "Neighbor", min=2, max=10, value=4, step=1)),
                        ui.column(6, ui.input_numeric("n_size_numeric", "Input", value=4, min=2, max=10)),
                    ),
                    ui.row(
                        ui.column(6, ui.input_slider("rewire_prob_slider", "Rewire", min=0.0, max=1.0, value=0.1, step=0.05)),
                        ui.column(6, ui.input_numeric("rewire_prob_numeric", "Input", value=0.1, min=0.0, max=1.0, step=0.05)),
                    ),
                ),
                ui.panel_conditional(
                    "input.network_type === 'Scale Free'",
                    ui.row(
                        ui.column(6, ui.input_slider("min_deg_slider", "Min Deg", min=1, max=10, value=2, step=1)),
                        ui.column(6, ui.input_numeric("min_deg_numeric", "Input", value=2, min=1, max=10)),
                    ),
                ),
                ui.panel_conditional(
                    "input.network_type === 'Hierarchical'",
                    ui.row(
                        ui.column(6, ui.input_slider("nlayers_slider", "Layers", min=2, max=5, value=3, step=1)),
                        ui.column(6, ui.input_numeric("nlayers_numeric", "Input", value=3, min=2, max=5)),
                    ),
                    ui.row(
                        ui.column(6, ui.input_slider("intra_layer_slider", "Intra", min=0.0, max=1.0, value=0.5, step=0.05)),
                        ui.column(6, ui.input_numeric("intra_layer_numeric", "Input", value=0.5, min=0.0, max=1.0, step=0.05)),
                    ),
                    ui.row(
                        ui.column(6, ui.input_slider("inter_layer_slider", "Inter", min=0.0, max=1.0, value=0.1, step=0.05)),
                        ui.column(6, ui.input_numeric("inter_layer_numeric", "Input", value=0.1, min=0.0, max=1.0, step=0.05)),
                    ),
                ),
            ),
            open="Basic Parameters" # Keep basic params open by default
        ),
        ui.hr(),
        width=380
    ),
        #ui.output_ui("progress_bar"),  # <--- Place this early in the main panel
        # Main Panel with Tabs
        ui.navset_tab(
            ui.nav_panel(
                "Network",
                ui.row(
#                    ui.column(12, output_widget("network_plot"))
                    ui.column(12, ui.output_ui("network_plot"))
                ),
                ui.row(
                    ui.column(4, 
                        ui.tags.div(
                            ui.tags.div(ui.output_text("num_agents"), class_="network-stat-number"),
                            ui.tags.div("Agents", class_="network-stat-label"),
                            class_="network-stat-box"
                        )
                    ),
                    ui.column(4, 
                        ui.tags.div(
                            ui.tags.div(ui.output_text("num_edges"), class_="network-stat-number"),
                            ui.tags.div("Edges", class_="network-stat-label"),
                            class_="network-stat-box"
                        )
                    ),
                    ui.column(4, 
                        ui.tags.div(
                            ui.tags.div(ui.output_text("net_density"), class_="network-stat-number"),
                            ui.tags.div("Density", class_="network-stat-label"),
                            class_="network-stat-box"
                        )
                    ),
                ),
            ),
            
            ui.nav_panel(
                "Performance",
                ui.row(
                    ui.column(12, output_widget("performance_plot"))
                ),
                ui.row(
                    ui.column(4, ui.value_box("Avg Violations", ui.output_text("avg_violations"), showcase=ui.tags.i(class_="fa fa-chart-line"), theme="info")),
                    ui.column(4, ui.value_box("Min Violations", ui.output_text("min_violations"), showcase=ui.tags.i(class_="fa fa-arrow-down"), theme="success")),
                    ui.column(4, ui.value_box("Homogeneity", ui.output_text("homogeneity"), showcase=ui.tags.i(class_="fa fa-balance-scale"), theme="warning")),
                )
            ),
            
            ui.nav_panel(
                "Distributions",
                ui.row(
                    ui.column(12, output_widget("agent_dist_plot"))
                ),
                ui.hr(), # Separator
                ui.row(
                    ui.column(12, output_widget("violations_vs_centrality_plot")) # <--- NEW PLOT
                )
            ),
            
            ui.nav_panel(
                "Knowledge Base",
                ui.row(
                    ui.column(12, output_widget("kb_analysis_plot"))
                )
            ),
            
            ui.nav_panel(
                "Data",
                ui.download_button("download_data", "📥 Download Results CSV"),
                ui.output_data_frame("data_table")
            ),

            ui.nav_panel(
                "How to Use?",
                ui.markdown(
                    """
                        # User Guide: Architectures of Problem solving

                        Welcome to this **Agent-Based Simulation** tool for exploring collective problem-solving in networks.
                        For more details about how the ABM works. You can go to Acknowledgements below and access full report. 
                        
                        ---

                        ##  Quick Start

                        1. **Configure** parameters in the sidebar (left)
                        2. Click **Setup** to initialize the network
                        3. Click **Run** to start the simulation
                        4. Switch between tabs to explore results

                        ---

                        ##  Key Parameters Explained

                        ### Problem Complexity
                        - **Agents (N)**: Number of decision-makers (10-500)
                        - *Example*: Use N=50 for quick tests, N=200+ for realistic simulations
                        - **Variables (K)**: Problem set dimensions (5-100)
                        - *Example*: K=20 represents a simple agreement task, K=80 is highly complex
                        - **Density (α)**: Constraint ratio `clauses/variables`
                        - *Example*: α=1.0 is easy, α=3.0 is hard, α>5.0 may be unsolvable

                        ### Information Flow
                        - **Obs Prob**: How often agents see "ground truth" directly
                        - *Example*: 0.01 = sparse external info, 0.1 = abundant external info
                        - **Clause Interval**: New constraints added every N steps
                        - *Example*: 10 = dynamic problem, 500 = static problem

                        ---

                        ##  Use Cases

                        ### Scenario 1: Testing Network Structure Impact
                        1. Setup with **Email Network** (real-world topology)
                        2. Run with α=2.0, obs_prob=0.01
                        3. Compare to **Random** network with same parameters
                        4. **Expected**: Real networks solve faster due to community structure

                        ### Scenario 2: Perform Empirical network comparison
                        1. Use Goto the experiments tab
                        2. Click on run simulations and wait for a while.
                        3. **Expected**: Different networks convergint to different rates. (Note this takes a while to complete)

                        ---

                        ##  Interpreting Results

                        ### Performance Tab
                        - **Avg Violations**: System-wide constraint satisfaction (lower = better)
                        - **Min Violations**: Best agent performance (tracks innovation)
                        - **Homogeneity**: Agreement level (1.0 = perfect consensus)

                        ### Network Tab
                        - **Node Color**: Eigenvector centrality 
                        - **Node Size**: Degree (larger = more connected)

                        ### Distributions Tab
                        - Check if violations follow power-law (most agents succeed, few fail)
                        - Knowledge base size shows information accumulation patterns

                        ---

                        ##  Troubleshooting

                        **Simulation too slow?**
                        - Reduce N or R
                        - Use simpler network type (Random instead of Hierarchical)

                        **Results look random?**
                        - Increase R (run longer)
                        - Lower α (make problem easier)
                        - Check seed for reproducibility

                        **Network won't load?**
                        - Ensure GraphML file is valid (test in Gephi/NetworkX first)
                        - Check file permissions

                        ---

                        ##  Scientific Background

                        This tool implements a **constraint satisfaction** model where:
                        1. Agents maintain local beliefs about shared variables
                        2. They communicate with neighbors to reduce conflicts
                        3. External observations provide ground truth signals
                        4. Success = finding assignments that satisfy all constraints
                      
                        ---

                        # About This Tool

                        **Architectures of Agreements** is an agent-based modeling platform for studying distributed consensus and problem-solving in networked systems. 
                        This model was created for a research project for the Complexity Global School 2025. The model was first constructed in NetLogo and later implemented in Python
                        
                        ---

                        ## Research Context

                        This simulation implements a **constraint satisfaction** framework where:
                        - Agents represent decision-makers with partial information
                        - Networks define communication structure
                        - Constraints represent shared objectives or logical rules
                        - Success is measured by collective satisfaction of all constraints

                        **Key Insights:**
                        - Network topology significantly affects solution quality
                        - Information flow (obs_prob) is critical for convergence
                        - Heterogeneous networks often outperform random graphs

                        ---

                        ## Technical Stack

                        - **Frontend**: Shiny for Python
                        - **Visualization**: Plotly, Gravis
                        - **Backend**: NetworkX, Mesa (agent-based modeling)
                        - **Network Analysis**: Eigenvector centrality, clustering coefficients

                        ---

                        ## Authors:
                        Atiyab Zafar, Fausto Gernone, Pia Andres and Ariane Donda
                      
                        ---


                        ---

                        ## Acknowledgments

                        For detailed acknowledgements and references see the report. [Click Here](https://www.github.com/atiyabzafar/architecturesofproblemsolving/blob/main/report/Report.pdf?raw=true) for the pdf. 
                    """
                )
            ),

            ui.nav_panel(
                "Experiment",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.tags.div("Experiment Controls", class_="sidebar-section-title"),
                        
                        # Multi-select for networks
                        ui.input_checkbox_group(
                            "experiment_networks",
                            "Select Networks to Compare:",
                            choices={
                                "Email": "Email Manufacturing",
                                "Pol Blogs": "Political Blogs",
                                "Congress": "Congress Twitter",
                                "Conference1": "Conference (MCL)",
                                "Conference2": "Conference (AES)",
                                "Conference3": "Conference (CMC)",
                                "Conference4": "Conference (TDA)",
                                "2024CGS": "2024 CGS Network",
                                "2025CGS": "2025 CGS Network"
                            },
                            selected=["Email", "Congress"]  # Default selection
                        ),
                        
                        ui.input_numeric("experiment_steps", "Simulation Steps", value=500, min=100, max=2000, step=100),
                        ui.input_numeric("experiment_seed", "Random Seed", value=42),
                        
                        ui.input_action_button("run_experiment", "▶️ Run Experiment", class_="btn-run"),
                        ui.input_action_button("stop_experiment", "⏹️ Stop Experiment", class_="btn-stop"),
                        
                        ui.hr(),
                        ui.output_ui("experiment_status"),
                        width=350,
                    ),
                    ui.tags.h3("Please note this experiment will take some time to run for all networks"),
                    # Main plot area
                    ui.navset_tab(
                        ui.nav_panel("Violations", output_widget("experiment_violations_plot")),
                        ui.nav_panel("Homogeneity", output_widget("experiment_homogeneity_plot")),
                        ui.nav_panel("Summary", output_widget("experiment_summary_plot"))
                    )
                )
            ),


        )
    )
)

def server(input, output, session):
    # Reactive values to store model and data
    model_data = reactive.Value(None)
    network_model = reactive.Value(None)
    is_running = reactive.Value(False)
    current_model = reactive.Value(None)
    simulation_progress = reactive.Value(0)
    network_html_cache = reactive.Value(None)
    sim_step = reactive.Value(0)
    sim_total = reactive.Value(0)
    # Reactive values for experiments
    experiment_results = reactive.Value(pd.DataFrame())  # Stores all results
    experiment_step = reactive.Value(0)
    current_experiment_network = reactive.Value(None)
    experiment_network_queue = reactive.Value([])  # Queue of networks to run
    current_experiment_model = reactive.Value(None)
    is_experiment_running = reactive.Value(False) 

    # Synchronize slider and numeric inputs
    @reactive.Effect
    def sync_N():
        input.N_slider()
        ui.update_numeric("N_numeric", value=input.N_slider())
    
    @reactive.Effect
    def sync_N_numeric():
        input.N_numeric()
        ui.update_slider("N_slider", value=input.N_numeric())
    
    @reactive.Effect
    def sync_K():
        input.K_slider()
        ui.update_numeric("K_numeric", value=input.K_slider())
    
    @reactive.Effect
    def sync_K_numeric():
        input.K_numeric()
        ui.update_slider("K_slider", value=input.K_numeric())
    
    @reactive.Effect
    def sync_alpha():
        input.alpha_slider()
        ui.update_numeric("alpha_numeric", value=round(input.alpha_slider(), 2))
    
    @reactive.Effect
    def sync_alpha_numeric():
        input.alpha_numeric()
        ui.update_slider("alpha_slider", value=input.alpha_numeric())
    
    @reactive.Effect
    def sync_obs_prob():
        input.obs_prob_slider()
        ui.update_numeric("obs_prob_numeric", value=round(input.obs_prob_slider(), 2))
    
    @reactive.Effect
    def sync_obs_prob_numeric():
        input.obs_prob_numeric()
        ui.update_slider("obs_prob_slider", value=input.obs_prob_numeric())
    
    @reactive.Effect
    def sync_clause_interval():
        input.clause_interval_slider()
        ui.update_numeric("clause_interval_numeric", value=input.clause_interval_slider())
    
    @reactive.Effect
    def sync_clause_interval_numeric():
        input.clause_interval_numeric()
        ui.update_slider("clause_interval_slider", value=input.clause_interval_numeric())
    
    @reactive.Effect
    def sync_R():
        input.R_slider()
        ui.update_numeric("R_numeric", value=input.R_slider())
    
    @reactive.Effect
    def sync_R_numeric():
        input.R_numeric()
        ui.update_slider("R_slider", value=input.R_numeric())
    
    # Network-specific synchronization
    @reactive.Effect
    def sync_connect_prob():
        input.connect_prob_slider()
        ui.update_numeric("connect_prob_numeric", value=round(input.connect_prob_slider(), 2))
    
    @reactive.Effect
    def sync_connect_prob_numeric():
        input.connect_prob_numeric()
        ui.update_slider("connect_prob_slider", value=input.connect_prob_numeric())
    
    @reactive.Effect
    def sync_n_size():
        input.n_size_slider()
        ui.update_numeric("n_size_numeric", value=input.n_size_slider())
    
    @reactive.Effect
    def sync_n_size_numeric():
        input.n_size_numeric()
        ui.update_slider("n_size_slider", value=input.n_size_numeric())
    
    @reactive.Effect
    def sync_rewire_prob():
        input.rewire_prob_slider()
        ui.update_numeric("rewire_prob_numeric", value=round(input.rewire_prob_slider(), 2))
    
    @reactive.Effect
    def sync_rewire_prob_numeric():
        input.rewire_prob_numeric()
        ui.update_slider("rewire_prob_slider", value=input.rewire_prob_numeric())
    
    @reactive.Effect
    def sync_min_deg():
        input.min_deg_slider()
        ui.update_numeric("min_deg_numeric", value=input.min_deg_slider())
    
    @reactive.Effect
    def sync_min_deg_numeric():
        input.min_deg_numeric()
        ui.update_slider("min_deg_slider", value=input.min_deg_numeric())
    
    @reactive.Effect
    def sync_nlayers():
        input.nlayers_slider()
        ui.update_numeric("nlayers_numeric", value=input.nlayers_slider())
    
    @reactive.Effect
    def sync_nlayers_numeric():
        input.nlayers_numeric()
        ui.update_slider("nlayers_slider", value=input.nlayers_numeric())
    
    @reactive.Effect
    def sync_intra_layer():
        input.intra_layer_slider()
        ui.update_numeric("intra_layer_numeric", value=round(input.intra_layer_slider(), 2))
    
    @reactive.Effect
    def sync_intra_layer_numeric():
        input.intra_layer_numeric()
        ui.update_slider("intra_layer_slider", value=input.intra_layer_numeric())
    
    @reactive.Effect
    def sync_inter_layer():
        input.inter_layer_slider()
        ui.update_numeric("inter_layer_numeric", value=round(input.inter_layer_slider(), 2))
    
    @reactive.Effect
    def sync_inter_layer_numeric():
        input.inter_layer_numeric()
        ui.update_slider("inter_layer_slider", value=input.inter_layer_numeric())
    
    @reactive.Effect
    def validate_large_inputs():
        N = input.N_numeric() or input.N_slider()
        K = input.K_numeric() or input.K_slider()
        
        if N * K > 50000:
            ui.notification_show(
                "⚠️ Large configuration may be slow!",
                type="warning",
                duration=3
            )

    @reactive.Effect
    @reactive.event(input.setup_model)
    def setup_network():
        # Clear old model explicitly
        old_model = network_model()
        if old_model:
            del old_model  # Help garbage collector
        
        # Reset state
        model_data.set(None)
        # simulation_progress.set(0)
        # network_html_cache.set(None)

        simulation_progress.set(0)
        network_html_cache.set(None)  # Clear cache on new setup
        
        # Base parameters
        params = {
            'N': input.N_numeric() or input.N_slider(),
            'K': input.K_numeric() or input.K_slider(),
            'alpha': input.alpha_numeric() or input.alpha_slider(),
            'obs_prob': input.obs_prob_numeric() or input.obs_prob_slider(),
            'clause_interval': input.clause_interval_numeric() or input.clause_interval_slider(),
            'R': input.R_numeric() or input.R_slider(),
            'setup_source': input.setup_source(),  # Add this!
            'type_network': input.network_type(),
            'seed': input.seed() if input.seed() else None
        }
        
        # Add file path if dataset selected
        if input.setup_source() == "dataset":
            if input.graphml_file() is None:
                print("❌ Please upload a GraphML file")
                return
            
            # Get uploaded file path
            file_info = input.graphml_file()[0]
            params['file_path'] = file_info["datapath"]
            print(f"📁 Loading: {file_info['name']}")

        elif input.setup_source() == "loaded":
            # Define network filename mapping
            network_files = {
                "Email": "data/EmailManufacturing-copy.xml",
                "Pol Blogs": "data/PolBlogsGiant.xml",
                "Congress": "data/congress.graphml",
                "Conference1": "data/fwdscialogdata/networks/Collab_values_MCL_associated.graphml",
                "Conference2": "data/fwdscialogdata/networks/Collab_values_AES_associated.graphml",
                "Conference3": "data/fwdscialogdata/networks/Collab_values_CMC_associated.graphml",
                "Conference4": "data/fwdscialogdata/networks/Collab_values_TDA_associated.graphml",
                "2024CGS": "data/2024ArchMessages_Spaces_etherpad.graphml",
                "2025CGS": "data/2025ArchMessages_Spaces.graphml"
            }
            
            # Get filename with fallback
            filename = network_files.get(
                input.loaded_network(), 
                "data/fwdscialogdata/networks/Collab_values_MCL_associated.graphml"
            )

            params['file_path']=filename
            params['setup_source']="dataset"
#            params[]

        else:
            print("Generated network selected")
            #print(input.seed,"\t",input.N_numeric)
            # Network parameters for generated networks
            if input.network_type() == "Random":
                params['connect_prob'] = input.connect_prob_numeric() or input.connect_prob_slider()
            elif input.network_type() == "Small World":
                params['n_size'] = input.n_size_numeric() or input.n_size_slider()
                params['rewire_prob'] = input.rewire_prob_numeric() or input.rewire_prob_slider()
            elif input.network_type() == "Scale Free":
                params['min_deg'] = input.min_deg_numeric() or input.min_deg_slider()
            elif input.network_type() == "Hierarchical":
                params['nlayers'] = input.nlayers_numeric() or input.nlayers_slider()
                params['intra_layer_connectance'] = input.intra_layer_numeric() or input.intra_layer_slider()
                params['inter_layer_connectance'] = input.inter_layer_numeric() or input.inter_layer_slider()
        
        try:
            model = ProblemSolvingModel(**params)
            model.calc_performances()
            network_model.set(model)
            current_model.set(model)
            print("✅ Setup complete!")
        except Exception as e:
            print(f"❌ Error: {e}")
            # Show user-friendly error in UI
            ui.notification_show(f"Error: File not found", type="error", duration=5)
        except Exception as e:
            print(f"❌ Setup error: {e}")
            ui.notification_show(f"Setup failed: {str(e)}", type="error", duration=5)

    @reactive.Effect
    @reactive.event(input.run_sim)
    def start_simulation():
        if network_model() is None:
            print("⚠️ Setup network first!")
            return
        
        is_running.set(True)
        sim_step.set(0)
        sim_total.set(network_model().R)
        simulation_progress.set(0)
        
        # Trigger first step
        reactive.invalidate_later(0)

    # @reactive.Effect
    # def run_step():
    #     if not is_running():
    #         return
        
    #     model = network_model()
    #     current_step = sim_step()
    #     total = sim_total()
        
    #     if current_step >= total:
    #         # Simulation complete
    #         is_running.set(False)
    #         simulation_progress.set(1.0)
            
    #         # Collect data
    #         model_df = model.datacollector.get_model_vars_dataframe()
    #         agent_df = model.datacollector.get_agent_vars_dataframe()
    #         model_data.set({
    #             'model_df': model_df,
    #             'agent_df': agent_df,
    #             'model': model,
    #         })
    #         return
        
    #     # Run one step
    #     model.step()
        
    #     # Update progress
    #     new_step = current_step + 1
    #     sim_step.set(new_step)
    #     simulation_progress.set(new_step / total)
        
    #     # Schedule next step
    #     reactive.invalidate_later(0.01)  # Run next step after 1ms


    @reactive.Effect
    def run_step():
        if not is_running():
            return
        
        model = network_model()
        if model is None:
            return
        
        current_step = sim_step()
        total = sim_total()
        
        # Check completion
        if current_step >= total:
            is_running.set(False)
            simulation_progress.set(1.0)
            # Final data collection
            model_df = model.datacollector.get_model_vars_dataframe()
            agent_df = model.datacollector.get_agent_vars_dataframe()
            model_data.set({
                'model_df': model_df,
                'agent_df': agent_df,
                'model': model,
            })
            print("Run Completed")
            network_html_cache.set(None)
            return
        
        # ✅ KEY FIX: Run 5 steps per batch (adjust based on speed)
        batch_size = min(5, total - current_step)
        for _ in range(batch_size):
            model.step()
        
        new_step = current_step + batch_size
        sim_step.set(new_step)
        
        # ✅ Update ONLY the progress value (lightweight)
        simulation_progress.set(new_step / total)
        
        
        # ✅ Yield control back to UI with minimal delay
        # Use 0.001 (1ms) to make it feel instant while keeping UI responsive
        reactive.invalidate_later(0.001)


    @reactive.Effect
    @reactive.event(input.stop_sim)
    def stop_simulation():
        is_running.set(False)
    
    # Status and Progress
    @output
    @render.ui
    def status():
        if is_running():
            prog = simulation_progress()
            #print(prog)
            return ui.tags.div(ui.tags.span(f"Running... {int(prog * 100)}%", class_="status-badge status-running"))
        elif network_model() is not None and model_data() is not None:
            return ui.tags.div(ui.tags.span("Complete", class_="status-badge status-complete"))
        elif network_model() is not None:
            return ui.tags.div(ui.tags.span("Ready to Run", class_="status-badge status-ready"))
        else:
            return ui.tags.div(ui.tags.span("Setup First", class_="status-badge status-ready"))
    
    # @output
    # @render.ui
    # def progress_bar():
    #     if is_running() or model_data() is not None:
    #         progress = simulation_progress()
    #         return ui.tags.div(
    #             ui.tags.div(
    #                 ui.tags.div(style=f"width: {progress * 100}%;", class_="progress-bar-fill"),
    #                 class_="progress-bar-container"
    #             )
    #         )
    #     return ui.tags.div()

    # @output
    # @render.ui
    # def progress_bar():
    #     if is_running() or model_data() is not None:
    #         progress = simulation_progress()  # This is 0.0–1.0
    #         progress_pct = progress * 100      # Convert to percentage
    #         print("progress_pct:",progress_pct)
    #         # Add status label
    #         if is_running():
    #             model = network_model()
    #             current_step = int(progress * model.R) if model else 0
    #             total_steps = model.R if model else 0
    #             label = f"Running: Step {current_step}/{total_steps}"
    #             #print(current_step,"\t",total_steps)
    #         else:
    #             label = "Complete"
            
    #         return ui.tags.div(
    #             # Label
    #             ui.tags.div(
    #                 label, 
    #                 style="font-size: 13px; margin-bottom: 5px; color: #555; font-weight: bold;"
    #             ),
    #             # Progress bar container
    #             ui.tags.div(
    #                 ui.tags.div(
    #                     style=f"width: {progress_pct}%;",  # ← Use percentage!
    #                     class_="progress-bar-fill"
    #                 ),
    #                 class_="progress-bar-container"
    #             )
    #         )
        
    #     return ui.tags.div()
    @output
    @render.ui  
    def progress_bar():
        # Always render, even if not running yet
        progress = simulation_progress()  # 0.0 to 1.0
        progress_pct = progress * 100
        
        # Determine label based on state
        if is_running():
            model = network_model()
            if model:
                current_step = int(progress * model.R)
                total_steps = model.R
                label = f"Running: Step {current_step}/{total_steps}"
            else:
                label = "Starting..."
        elif model_data() is not None:
            label = "Complete"
        else:
            label = "Ready to run"  # Show this even before clicking Run
        
        return ui.tags.div(
            # Label
            ui.tags.div(
                label,
                style="font-size: 13px; margin-bottom: 5px; color: #555; font-weight: bold;"
            ),
            # Progress bar container
            ui.tags.div(
                ui.tags.div(
                    style=f"width: {progress_pct}%;",
                    class_="progress-bar-fill"
                ),
                class_="progress-bar-container"
            )
        )

    # @output
    # @render_widget
    # def violations_vs_centrality_plot():
    #     # Ensure model data exists
    #     data = model_data()
    #     if data is None or 'model' not in data:
    #         return go.Figure()

    #     # 1. Get the actual Model object instance
    #     model = data['model']
    #     G = model.network
        
    #     # --- CORRECT ACCESS FOR YOUR MODEL.PY ---
    #     # Your model uses 'agent_list', not 'schedule.agents'
    #     if hasattr(model, 'agent_list'):
    #         agents = model.agent_list
    #     else:
    #         # Fallback in case you revert to an older model version
    #         agents = getattr(model, 'agents', []) 
    #         if not agents and hasattr(model, 'schedule'):
    #             agents = model.schedule.agents
                
    #     if not agents: 
    #         return go.Figure()
    #     # ----------------------------------------

    #     # 2. Create dictionary: Agent ID -> Agent Object
    #     agent_dict = {a.unique_id: a for a in agents}

    #     # 3. Get Centrality (In-Degree)
    #     deg_dict = dict(G.in_degree()) 
    #     max_deg = max(deg_dict.values()) if deg_dict else 1

    #     # 4. Align Data
    #     valid_ids = [i for i in agent_dict.keys() if i in deg_dict]
    #     sorted_ids = sorted(valid_ids)
        
    #     aligned_centrality = []
    #     aligned_violations = []
    #     aligned_kb_size = []
    #     hover_texts = []

    #     for i in sorted_ids:
    #         # X-Axis: Normalized In-Degree Centrality
    #         d = deg_dict.get(i, 0)
    #         norm_centrality = d / max_deg if max_deg > 0 else 0
    #         aligned_centrality.append(norm_centrality)
            
    #         # Y-Axis: Violations
    #         agent = agent_dict[i]
    #         aligned_violations.append(agent.true_violations)
            
    #         # Color/Size dimension: KB Size
    #         kb_len = len(agent.kb) if isinstance(agent.kb, (list, set)) else 0
    #         aligned_kb_size.append(kb_len)
            
    #         # Hover info
    #         hover_texts.append(f"Agent {i}")

    #     # 5. Create Plotly Scatter Plot
    #     fig = go.Figure(data=go.Scatter(
    #         x=aligned_centrality,
    #         y=aligned_violations,
    #         mode='markers',
    #         marker=dict(
    #             size=12,
    #             color=aligned_kb_size,
    #             colorscale='Viridis',
    #             showscale=True,
    #             colorbar=dict(title="KB Size"),
    #             opacity=0.8,
    #             line=dict(width=1, color='DarkSlateGrey')
    #         ),
    #         text=hover_texts,
    #         hovertemplate="<b>%{text}</b><br>" +
    #                     "Centrality: %{x:.2f}<br>" +
    #                     "Violations: %{y}<br>" +
    #                     "KB Size: %{marker.color}<extra></extra>"
    #     ))

    #     fig.update_layout(
    #         title="Agent Performance vs. Network Position",
    #         xaxis_title="Normalized In-Degree Centrality",
    #         yaxis_title="Violations (Unsolved Constraints)",
    #         height=500,
    #         plot_bgcolor='#f8f9fa',
    #         paper_bgcolor='#fefae0',
    #         font=dict(family="Arial", color="#333")
    #     )
        
    #     return fig

    @output
    @render_widget
    def violations_vs_centrality_plot():
        # Ensure model data exists
        data = model_data()
        if data is None or 'model' not in data:
            return go.Figure()

        # 1. Get the actual Model object instance
        model = data['model']
        G = model.network
        
        # --- ROBUST AGENT ACCESS ---
        if hasattr(model, 'agent_list'):
            agents = model.agent_list
        else:
            agents = getattr(model, 'agents', []) 
            if not agents and hasattr(model, 'schedule'):
                agents = model.schedule.agents
                
        if not agents: return go.Figure()
        # ---------------------------

        # 2. Create dictionary for FAST lookups: Agent ID -> Agent Object
        agent_dict = {a.unique_id: a for a in agents}

        # 3. Get Centrality (In-Degree) as a dictionary
        # Using in-degree because it represents information *reception* capacity
        degree_dict = dict(G.in_degree())
        try:
            degree_dict = nx.eigenvector_centrality(G, max_iter=100, tol=1e-4)
        except:
            # Fallback to fast in-degree centrality
            degree_dict = {node: G.in_degree(node) for node in G.nodes()}

        
        if not degree_dict:
            return go.Figure()

        max_deg = max(degree_dict.values()) if degree_dict else 1

        # 4. MATCH DATA BY ID (Crucial Step)
        # We iterate through sorted node IDs to ensure alignment
        sorted_node_ids = sorted([n for n in G.nodes() if n in agent_dict])
        
        aligned_centrality = []
        aligned_violations = []
        aligned_kb_size = []
        hover_texts = []

        for node_id in sorted_node_ids:
            # X-Axis: Normalized In-Degree Centrality
            d = degree_dict.get(node_id, 0)
            norm_centrality = d / max_deg
            aligned_centrality.append(norm_centrality)
            
            # Y-Axis: Violations
            agent = agent_dict[node_id]
            aligned_violations.append(agent.true_violations)
            
            # Color/Size dimension: KB Size
            # Handle difference between list and set implementation of KB
            #kb_len = len(agent.kb) if isinstance(agent.kb, (list, set)) else 0
          #  aligned_kb_size.append(kb_len)
            
            # Hover info
            hover_texts.append(f"Agent {node_id}")

        # 5. Create Plotly Scatter Plot
        fig = go.Figure()

        # Add Scatter Points
        fig.add_trace(go.Scatter(
            x=aligned_centrality,
            y=aligned_violations,
            mode='markers',
            marker=dict(
                size=12,
             #   color=aligned_kb_size,
             #   colorscale='Viridis',
            #    showscale=True,
             #   colorbar=dict(title="KB Size"),
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=hover_texts,
            hovertemplate="<b>%{text}</b><br>" +
                        "Centrality: %{x:.2f}<br>" +
                        "Violations: %{y}<br>" +
                        "KB Size: %{marker.color}<extra></extra>",
            name='Agents'
        ))

        # Add Trendline (Linear Regression)
        if len(aligned_centrality) > 1:
            try:
                import statsmodels.api as sm
                # Add constant for intercept
                X = sm.add_constant(aligned_centrality)
                model_ols = sm.OLS(aligned_violations, X).fit()
                
                # Create trendline points
                x_range = np.linspace(min(aligned_centrality), max(aligned_centrality), 100)
                y_pred = model_ols.params[0] + model_ols.params[1] * x_range
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name=f'Trend (slope={model_ols.params[1]:.2f})',
                    line=dict(color='red', width=2, dash='dash')
                ))
            except ImportError:
                pass # Skip if statsmodels not installed

        fig.update_layout(
            title="Correlation: Centrality vs. Violations",
            xaxis_title="Normalized In-Degree Centrality (Influence)",
            yaxis_title="Violations (Unsolved Constraints)",
            height=500,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#fefae0',
            font=dict(family="Arial", color="#333"),
            showlegend=True
        )
        
        return fig
    
    @reactive.Effect
    @reactive.event(input.refresh_network)
    def refresh_network_viz():
        """Force network visualization to regenerate with updated agent states."""
        network_html_cache.set(None)
        ui.notification_show("Network refreshed with current agent states", type="info", duration=2)


    # Create a reactive calc for the network HTML
    @reactive.Calc
    def network_html():
        """Generate network HTML only when model changes"""
        net_model = network_model()

        sim_results = model_data() 

        if net_model is None:
            return None
        
        # Use updated model if available
        if sim_results is not None and 'model' in sim_results:
            net_model = sim_results['model']
        G = net_model.network
        
        # Get centrality range
        centralities = [a.centr for a in net_model.agents]
        min_cent = min(centralities) if centralities else 0
        max_cent = max(centralities) if centralities else 1
        cent_range = max_cent - min_cent if max_cent > min_cent else 1
        
        # Create matplotlib Plasma colormap
        plasma = plt.cm.plasma
        
        def get_plasma_color(centrality):
            norm_cent = (centrality - min_cent) / cent_range if cent_range > 0 else 0.5
            rgba = plasma(norm_cent)
            return mcolors.rgb2hex(rgba[:3])
        
        # Add node attributes
        for node in G.nodes():
            agent = None
            for a in net_model.agents:
                if a.unique_id == node:
                    agent = a
                    break
            
            if agent:
                centrality = agent.centr
                degree = G.degree(node)
                color = get_plasma_color(centrality)
                max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
                #rescaling size to 2 and 20
                size = 2 + (degree / max_degree) * 20
                
                if centrality > np.percentile([a.centr for a in net_model.agents if a.centr > 0] or [0], 80):
                    shape = 'diamond'
                else:
                    shape = 'circle'
                
                hover = f"Agent {node}\nCentrality: {centrality:.3f}\nDegree: {degree}\nViolations: {agent.true_violations}\nKB Size: {len(agent.kb)}"
                #print(hover)
                G.nodes[node]['color'] = color
                G.nodes[node]['size'] = size
                G.nodes[node]['label'] = str(node)
                G.nodes[node]['hover'] = hover
                G.nodes[node]['shape'] = shape
            else:
                G.nodes[node]['color'] = '#cccccc'
                G.nodes[node]['size'] = 10
                G.nodes[node]['label'] = str(node)
        
#        for edge in G.edges():
 #           G.edges[edge]['color']="5C"
        G.graph['edge_color'] = '#505050'
        G.graph['edge_opacity'] = 0.5
        # Create Gravis visualization
        fig = gv.d3(
            G,
            graph_height=650,
            node_hover_neighborhood=True,
            large_graph_threshold=300,
            node_size_factor=2.0,
            node_drag_fix=False,
            edge_curvature=0.1,
            many_body_force_strength=-100,
            links_force_distance=50,
            use_many_body_force=True,
            use_links_force=True,
            layout_algorithm_active=True,
            zoom_factor=1.0
        )
        
        html_content = fig.to_html()
        return html_content

    # Now the output just renders the cached HTML
    @output
    @render.ui
    def network_plot():
        if is_running():
            return ui.div(ui.div(class_="loading-spinner"), "Computing...")
        html = network_html()  # Uses reactive calc (cached)
        
        if html is None:
            return ui.div(
                "Click '🔧 Setup' to visualize the network",
                class_="network-placeholder"
            )
        
        # Get model for colorbar values
        net_model = network_model()
        if net_model:
            centralities = [a.centr for a in net_model.agents]
            min_cent = min(centralities) if centralities else 0
            max_cent = max(centralities) if centralities else 1
        else:
            min_cent, max_cent = 0, 1
        
        # Create colorbar as separate element
        colorbar = ui.tags.div(
            # Title at top
            ui.tags.div(
                "Centrality", 
                style="font-weight: bold; font-size: 13px; text-align: center; margin-bottom: 10px;"
            ),
            
            # Gradient bar with values
            ui.tags.div(
                # Max value at top
                ui.tags.div(
                    f"{max_cent:.3f}", 
                    style="font-size: 11px; text-align: center; margin-bottom: 3px; font-weight: bold;"
                ),
                
                # Gradient bar
                ui.tags.div(
                    style="""
                        width: 30px; 
                        height: 500px; 
                        background: linear-gradient(to top, 
                            #0d0887 0%, 
                            #46039f 10%, 
                            #7201a8 20%, 
                            #9c179e 30%, 
                            #bd3786 40%, 
                            #d8576b 50%, 
                            #ed7953 60%, 
                            #fb9f3a 70%, 
                            #fdca26 80%, 
                            #f0f921 100%
                        ); 
                        border: 1px solid #999; 
                        border-radius: 4px;
                        margin: 0 auto;
                    """
                ),
                
                # Min value at bottom
                ui.tags.div(
                    f"{min_cent:.3f}", 
                    style="font-size: 11px; text-align: center; margin-top: 3px; font-weight: bold;"
                ),
                
                style="display: flex; flex-direction: column; align-items: center;"
            ),
            
            # Footer text
            ui.tags.div(
                "Node size = degree", 
                style="font-size: 10px; text-align: center; color: #666; margin-top: 10px; font-style: italic;"
            ),
            
            class_="network-colorbar",
            style="background : #fefae0;"
        )

        return ui.tags.div(
            ui.HTML(html),
            colorbar,  # Add colorbar after the HTML
            class_="network-container",
            style="background : #fefae0; height: 700px !important; min-height: 700px !important; overflow: auto !important; position: relative;"
        )



    @reactive.Effect
    @reactive.event(input.run_experiment)
    def start_experiment():
        """Initialize multi-network experiment."""
        if is_experiment_running():
            print("⚠️ Experiment already running.")
            return
        
        selected_networks = list(input.experiment_networks())
        
        if not selected_networks:
            print("⚠️ No networks selected!")
            ui.notification_show("Please select at least one network", type="warning", duration=3)
            return
        
        print(f"🚀 Starting experiment with {len(selected_networks)} networks...")
        
        # Reset state
        experiment_results.set(pd.DataFrame())
        experiment_network_queue.set(selected_networks.copy())
        experiment_step.set(0)
        is_experiment_running.set(True)
        
        # Start first network
        _start_next_network()

    def _start_next_network():
        """Helper: Initialize the next network in queue."""
        queue = experiment_network_queue()
        
        if not queue:
            # All networks complete!
            is_experiment_running.set(False)
            print("🏁 All experiments complete!")
            return
        
        # Pop next network from queue
        next_network = queue.pop(0)
        experiment_network_queue.set(queue)
        current_experiment_network.set(next_network)
        experiment_step.set(0)
        
        # Map network names to file paths
        network_files = {
            "Email": "data/EmailManufacturing-copy.xml",
            "Pol Blogs": "data/PolBlogsGiant.xml",
            "Congress": "data/congress.graphml",
            "Conference1": "data/fwdscialogdata/networks/Collab_values_MCL_associated.graphml",
            "Conference2": "data/fwdscialogdata/networks/Collab_values_AES_associated.graphml",
            "Conference3": "data/fwdscialogdata/networks/Collab_values_CMC_associated.graphml",
            "Conference4": "data/fwdscialogdata/networks/Collab_values_TDA_associated.graphml",
            "2024CGS": "data/2024ArchMessages_Spaces_etherpad.graphml",
            "2025CGS": "data/2025ArchMessages_Spaces.graphml"
        }
        
        filename = network_files.get(next_network, "data/EmailManufacturing-copy.xml")
        
        try:
            print(f"📂 Loading network: {next_network}")
            model = ProblemSolvingModel(
                K=50, alpha=2.0, obs_prob=0.01, clause_interval=10,
                R=input.experiment_steps(), setup_source="dataset",
                file_path=filename, seed=input.experiment_seed()
            )
            current_experiment_model.set(model)
            print(f"✅ {next_network} initialized")
            
            # Trigger simulation loop
            reactive.invalidate_later(0)
            
        except Exception as e:
            print(f"❌ Error loading {next_network}: {e}")
            # Skip this network and try next
            _start_next_network()

    @reactive.Effect
    @reactive.event(input.stop_experiment)
    def stop_experiment():
        """Stop all experiments."""
        is_experiment_running.set(False)
        experiment_network_queue.set([])
        print("⏹️ Experiments stopped by user.")

    @reactive.Effect
    async def experiment_simulation_step():
        """Run simulation steps for current network."""
        if not is_experiment_running():
            return
        
        model = current_experiment_model()
        if model is None:
            return
        
        total_steps = model.R
        current_s = experiment_step()
        
        # Check if current network is complete
        if current_s >= total_steps:
            print(f"✅ {current_experiment_network()} complete")
            
            # Save results for this network
            full_df = model.datacollector.get_model_vars_dataframe().reset_index()
            if 'index' in full_df.columns:
                full_df = full_df.rename(columns={'index': 'Step'})
            full_df['network'] = current_experiment_network()
            
            # Append to results
            current_results = experiment_results()
            updated_results = pd.concat([current_results, full_df], ignore_index=True)
            experiment_results.set(updated_results)
            
            # Start next network
            _start_next_network()
            return
        
        # Run batch of steps
        batch_size = min(10, total_steps - current_s)
        for _ in range(batch_size):
            model.step()
        
        new_step_count = current_s + batch_size
        experiment_step.set(new_step_count)
        
        # Update plot every 50 steps
        if new_step_count % 50 == 0:
            full_df = model.datacollector.get_model_vars_dataframe().reset_index()
            if 'index' in full_df.columns:
                full_df = full_df.rename(columns={'index': 'Step'})
            full_df['network'] = current_experiment_network()
            
            # Append partial results
            current_results = experiment_results()
            updated_results = pd.concat([current_results, full_df], ignore_index=True)
            experiment_results.set(updated_results)
        
        # Yield control
        await asyncio.sleep(0.001)
        reactive.invalidate_later(0)

    @output
    @render.ui
    def experiment_status():
        """Display current experiment status."""
        if is_experiment_running():
            current_net = current_experiment_network()
            queue = experiment_network_queue()
            total_networks = len(input.experiment_networks())
            completed = total_networks - len(queue) - 1
            
            current_step = experiment_step()
            total_steps = input.experiment_steps()
            pct = (current_step / total_steps) * 100 if total_steps > 0 else 0
            
            return ui.div(
                f"Running: {current_net} ({completed + 1}/{total_networks})",
                ui.tags.br(),
                f"Step {current_step}/{total_steps} ({pct:.1f}%)",
                class_="status-badge status-running"
            )
        
        results = experiment_results()
        if not results.empty:
            return ui.div("All Experiments Complete ✓", class_="status-badge status-complete")
        
        return ui.div("Ready to run experiments", class_="status-badge status-ready")

    # --- EXPERIMENT PLOTS ---

    @output
    @render_widget
    def experiment_violations_plot():
        """Plot violations over time for all networks."""
        df = experiment_results()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Click 'Run Experiment' to see results", showarrow=False)
            fig.update_layout(height=600, paper_bgcolor='#fefae0', plot_bgcolor='#f8f9fa')
            return fig
        
        fig = go.Figure()
        colors = ['#606c38', '#283618', '#bc6c25', '#dda15e', '#e07a5f', '#3d405b', '#81b29a', '#f2cc8f']
        
        # ✅ FIX: Group by network AND step, take last value for each step
        # This removes duplicates
        df_clean = df.groupby(['network', 'Step']).last().reset_index()
        
        for idx, network in enumerate(df_clean['network'].unique()):
            network_data = df_clean[df_clean['network'] == network]
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=network_data['Step'],
                y=network_data['avg_violations'],
                mode='lines',
                name=network,
                line=dict(color=color, width=3)
            ))
        
        fig.update_layout(
            title="Average Violations Over Time",
            xaxis_title="Step",
            yaxis_title="Average Violations",
            height=600,
            showlegend=True,
            legend=dict(x=1.05, y=1),
            paper_bgcolor='#fefae0',
            plot_bgcolor='#f8f9fa',
            hovermode='x unified'
        )
        
        return fig

    @output
    @render_widget
    def experiment_homogeneity_plot():
        """Plot homogeneity over time for all networks."""
        df = experiment_results()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Click 'Run Experiment' to see results", showarrow=False)
            fig.update_layout(height=600, paper_bgcolor='#fefae0', plot_bgcolor='#f8f9fa')
            return fig
        
        fig = go.Figure()
        colors = ['#606c38', '#283618', '#bc6c25', '#dda15e', '#e07a5f', '#3d405b', '#81b29a', '#f2cc8f']
        
        # ✅ FIX: Remove duplicates
        df_clean = df.groupby(['network', 'Step']).last().reset_index()
        
        for idx, network in enumerate(df_clean['network'].unique()):
            network_data = df_clean[df_clean['network'] == network]
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=network_data['Step'],
                y=network_data['homogeneity'],
                mode='lines',
                name=network,
                line=dict(color=color, width=3)
            ))
        
        fig.update_layout(
            title="Homogeneity Over Time",
            xaxis_title="Step",
            yaxis_title="Homogeneity",
            height=600,
            showlegend=True,
            legend=dict(x=1.05, y=1),
            paper_bgcolor='#fefae0',
            plot_bgcolor='#f8f9fa',
            hovermode='x unified'
        )
        
        return fig


    @output
    @render_widget
    def experiment_summary_plot():
        """Summary comparison of final values."""
        df = experiment_results()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="Click 'Run Experiment' to see results", showarrow=False)
            fig.update_layout(height=600, paper_bgcolor='#fefae0', plot_bgcolor='#f8f9fa')
            return fig
        
        # Get final values for each network
        final_values = df.groupby('network').last().reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Final Average Violations", "Final Homogeneity")
        )
        
        # Violations bar chart
        fig.add_trace(
            go.Bar(
                x=final_values['network'],
                y=final_values['avg_violations'],
                marker_color='#606c38',
                name='Violations'
            ),
            row=1, col=1
        )
        
        # Homogeneity bar chart
        fig.add_trace(
            go.Bar(
                x=final_values['network'],
                y=final_values['homogeneity'],
                marker_color='#bc6c25',
                name='Homogeneity'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='#fefae0',
            plot_bgcolor='#f8f9fa'
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig


    
    # Network Statistics
    @output
    @render.text
    def num_agents():
        net_model = network_model()
        return str(net_model.network.number_of_nodes()) if net_model else "0"
    
    @output
    @render.text
    def num_edges():
        net_model = network_model()
        return str(net_model.network.number_of_edges()) if net_model else "0"
    
    @output
    @render.text
    def net_density():
        net_model = network_model()
        if net_model and net_model.network.number_of_nodes() > 1:
            N = net_model.network.number_of_nodes()
            M = net_model.network.number_of_edges()
            return f"{M / (N * (N - 1)):.3f}"
        return "0.000"
    
    # Performance Metrics
    @output
    @render.text
    def avg_violations():
        data = model_data()
        return f"{data['model_df']['avg_violations'].iloc[-1]:.2f}" if data else "N/A"
    
    @output
    @render.text
    def min_violations():
        data = model_data()
        return f"{int(data['model_df']['min_violations'].iloc[-1])}" if data else "N/A"
    
    @output
    @render.text
    def homogeneity():
        data = model_data()
        return f"{data['model_df']['homogeneity'].iloc[-1]:.3f}" if data else "N/A"
    
    @output
    @render.text
    def file_info():
        if input.graphml_file() is None:
            return ""
        
        file_data = input.graphml_file()[0]
        return f"File: {file_data['name']} ({file_data['size']} bytes)"

    @render.download(
        filename=lambda: f"simulation_results_{input.seed() or 'default'}.csv"
    )
    def download_data():
        print("DEBUG: download_data called")
        
        data = model_data()
        print(f"DEBUG: data is None? {data is None}")
        
        if data is None or 'model_df' not in data:
            # Create a temporary file with error message
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                tmp.write("error,message\nno_data,Please run simulation first\n")
                return tmp.name
        
        # Create a temporary file with the CSV data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            data['model_df'].to_csv(tmp, index=True)
            temp_path = tmp.name
        
        print(f"DEBUG: Created temp file at: {temp_path}")
        
        # Return the file path (NOT the content!)
        return temp_path



    @output
    @render_widget
    def performance_plot():
        data = model_data()
        if data is None:
            fig = go.Figure()
            fig.add_annotation(text="Run simulation to see results", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=800)
            fig.update_layout( paper_bgcolor='#fefae0' )
            return fig
        
        df = data['model_df']
        fig = make_subplots(rows=3, cols=1, subplot_titles=("📉 Average Violations", "📊 Minimum Violations", "📈 Homogeneity"), vertical_spacing=0.12)
        
        fig.add_trace(go.Scatter(y=df['avg_violations'], mode='lines', name='Avg', line=dict(color='#606c38', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['min_violations'], mode='lines', name='Min', line=dict(color='#f5576c', width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(y=df['homogeneity'], mode='lines', name='Homogeneity', line=dict(color='#4facfe', width=3)), row=3, col=1)
        
        fig.update_xaxes(title_text="Step", row=3, col=1)
        fig.update_yaxes(title_text="Violations", row=1, col=1)
        fig.update_yaxes(title_text="Violations", row=2, col=1)
        fig.update_yaxes(title_text="Homogeneity", row=3, col=1)
        fig.update_layout(height=800, showlegend=True, plot_bgcolor='#f8f9fa', font=dict(family="Arial", color="#333"))
        fig.update_layout( paper_bgcolor='#fefae0' )
        return fig
    
    @output
    @render_widget
    def agent_dist_plot():
        data = model_data()
        if data is None:
            fig = go.Figure()
            fig.add_annotation(text="Run simulation to see results", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=400)
            fig.update_layout( paper_bgcolor='#fefae0' )
            return fig
        
        agent_df = data['agent_df']
        final_step = agent_df.index.get_level_values('Step').max()
        final_data = agent_df.xs(final_step, level='Step')
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Violations", "KB Size", "Centrality"))
        fig.add_trace(go.Histogram(x=final_data['violations'], marker_color='#606c38', nbinsx=20, showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(x=final_data['kb_size'], marker_color='#f5576c', nbinsx=20, showlegend=False), row=1, col=2)
        fig.add_trace(go.Histogram(x=final_data['centrality'], marker_color='#4facfe', nbinsx=20, showlegend=False), row=1, col=3)
        
        fig.update_layout(height=400, plot_bgcolor='#f8f9fa', font=dict(family="Arial", color="#333"))
        fig.update_layout( paper_bgcolor='#fefae0' )
        return fig
    
    @output
    @render_widget
    def kb_analysis_plot():
        data = model_data()
        if data is None:
            fig = go.Figure()
            fig.add_annotation(text="Run simulation to see results", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=500)
            return fig
        
        agent_df = data['agent_df']
        kb_over_time = agent_df.groupby(level='Step')['kb_size'].agg(['mean', 'std', 'min', 'max'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=kb_over_time['mean'], mode='lines', name='Mean', line=dict(color='#606c38', width=3)))
        fig.add_trace(go.Scatter(y=kb_over_time['mean'] + kb_over_time['std'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(y=kb_over_time['mean'] - kb_over_time['std'], mode='lines', line=dict(width=0), fillcolor='rgba(102, 126, 234, 0.2)', fill='tonexty', name='±1 Std', hoverinfo='skip'))
        fig.add_trace(go.Scatter(y=kb_over_time['max'], mode='lines', name='Max', line=dict(color='#f5576c', width=1, dash='dash')))
        fig.add_trace(go.Scatter(y=kb_over_time['min'], mode='lines', name='Min', line=dict(color='#4facfe', width=1, dash='dash')))
        
        fig.update_layout(title="Knowledge Base Evolution", xaxis_title="Step", yaxis_title="KB Size", height=500, hovermode='x unified', plot_bgcolor='#f8f9fa', font=dict(family="Arial", color="#333"))
        fig.update_layout( paper_bgcolor='#fefae0' )
        return fig
    
    @output
    @render.data_frame
    def data_table():
        data = model_data()
        if data is None:
            return render.DataGrid(pd.DataFrame({'Message': ['Run simulation to see data']}), width="100%", height="500px")
        df = data['model_df'].reset_index()
        return render.DataGrid(df, width="100%", height="500px")

app = App(app_ui, server)
