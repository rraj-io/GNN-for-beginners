# Graph Neural Networks (GNNs) from Scratch

### From graph fundamentals to scientific machine learning and engineering applications

**Python · NumPy · PyTorch · PyTorch Geometric · Jupyter Notebook**

---

## Introduction

Most deep learning tutorials work with images, sequences, or tabular data. But what happens when data points are not arranged in a regular grid? What if the relationships between individual data points are just as important as the data itself?

This is where **Graph Neural Networks (GNNs)** come in.

Graphs provide a natural way to represent interconnected systems. From social networks and molecular structures to transportation networks and computational meshes, many real-world problems can be described using nodes, edges, and their associated features.

This repository is a progressive, hands-on tutorial series designed to develop a deeper understanding of Graph Neural Networks — starting with the mathematical foundations of graphs, implementing graph convolution from scratch, training models using PyTorch, and gradually exploring modern GNN architectures and scientific applications.

The objective is not simply to learn how to use a GNN library. It is to understand **how information travels through a graph, how neighbouring nodes interact, how graph structures become learnable representations, and how these ideas can be applied to real engineering problems.**

---

## Learning Philosophy

The series follows a build-it-from-scratch approach.

1. **Understand before implementing:** Start with graph theory, mathematical representations, and the intuition behind each operation.
2. **Derive before using libraries:** Implement fundamental operations using NumPy and basic PyTorch wherever practical.
3. **Progressively introduce architectures:** Move from simple graph convolution to message-passing networks and more advanced GNN architectures.
4. **Learn through implementation:** Build models, inspect intermediate results, train networks, and evaluate their performance.
5. **Work with real datasets:** Move from small illustrative graphs to established graph datasets and practical applications.
6. **Connect theory to engineering:** Explore how GNNs can represent computational meshes, physical systems, and scientific data.
7. **Understand limitations:** Discuss computational costs, generalisation, model limitations, and physical consistency.

Each tutorial builds upon the previous one while remaining focused on a specific concept or architecture.

---

## Series Roadmap

| # | Tutorial | Main Topics | Status |
|---|---|---|---|
| 01 | **Graph Fundamentals and Representations** | Nodes, edges, graph types, node features, edge features, adjacency matrices, degree matrices | Available |
| 02 | **Graph Convolutional Networks from Scratch** | Self-loops, degree matrices, normalization, graph propagation, NumPy implementation | Available |
| 03 | **Training a GCN with PyTorch** | Tensors, trainable parameters, autograd, backpropagation, loss functions, node classification | Available |
| 04 | **PyTorch Geometric and the Cora Dataset** | PyG, `Data`, `edge_index`, Cora dataset, GCNConv, training and evaluation | Available |
| 05 | **Message Passing Neural Networks** | Message, aggregate, update, custom message-passing layers, GraphSAGE-style implementation | In Progress |
| 06 | **Graph Attention Networks (GAT)** | Attention mechanisms, attention coefficients, weighted neighbourhood aggregation | Planned |
| 07 | **GraphSAGE and Advanced Aggregation** | Neighbourhood sampling, aggregation strategies, inductive learning, scalability | Planned |
| 08 | **Graph-Level Learning** | Graph classification, graph regression, pooling and graph representations | Planned |
| 09 | **GNNs for Mesh-Based Systems** | Mesh connectivity, geometric features, graph representations of computational domains | Planned |
| 10 | **GNNs for Scientific Machine Learning** | Surrogate modelling, physical systems, engineering applications and limitations | Planned |

The roadmap may evolve as the series develops, but the overall progression will remain:

**Graph Fundamentals → Graph Convolution → Training → Message Passing → Advanced Architectures → Scientific Applications**

---

## Repository Structure

The repository is organised by tutorial. Each tutorial contains a Jupyter notebook with explanations, mathematical derivations, code implementations, experiments, and exercises.

```text
GNN-Tutorials/
│
├── README.md
│
├── tutorial_01_graph_fundamentals/
│   └── GNN_Tutorial_01_Graph_Fundamentals.ipynb
│
├── tutorial_02_gcn_from_scratch/
│   └── GNN_Tutorial_02_GCN_From_Scratch.ipynb
│
├── tutorial_03_training_gcn_pytorch/
│   └── GNN_Tutorial_03_Training_GCN_with_PyTorch.ipynb
│
├── tutorial_04_pytorch_geometric_cora/
│   └── GNN_Tutorial_04_PyTorch_Geometric_Cora.ipynb
│
├── tutorial_05_message_passing/
│   └── GNN_Tutorial_05_Message_Passing_Networks.ipynb
│
├── tutorial_06_graph_attention/
│   └── GNN_Tutorial_06_Graph_Attention_Networks.ipynb
│
├── tutorial_07_graphsage/
│   └── GNN_Tutorial_07_GraphSAGE.ipynb
│
├── tutorial_08_graph_level_learning/
│   └── GNN_Tutorial_08_Graph_Level_Learning.ipynb
│
├── tutorial_09_gnns_for_meshes/
│   └── GNN_Tutorial_09_GNNs_for_Mesh_Based_Systems.ipynb
│
└── tutorial_10_gnns_for_sciml/
    └── GNN_Tutorial_10_GNNs_for_Scientific_Machine_Learning.ipynb
```

*Note: The directory and notebook names are suggested conventions. Keep them consistent when adding new tutorials.*

---

## Tutorial Structure and Teaching Style

To maintain consistency throughout the series, every tutorial should follow a common teaching structure.

### 1. Introduction and Motivation

Start with a short explanation of:
- What problem are we trying to solve?
- Why is this concept or architecture important?
- Where is it used in practical applications?
- What will we implement in this tutorial?

The introduction should establish the engineering or mathematical motivation without becoming unnecessarily lengthy.

### 2. Learning Objectives

List the specific concepts and skills the reader should gain.

Focus on practical outcomes, such as implementing a graph operation, understanding an equation, building a model, or evaluating predictions.

### 3. Mathematical Foundations

Introduce the mathematical formulation before using a library implementation.

For every important equation:
- Define all variables and symbols.
- Explain the physical or mathematical meaning.
- Connect the equation to the graph representation.
- Discuss relevant assumptions.

Use consistent mathematical notation across the series.

### 4. Small Worked Examples

Before moving to a full dataset or model, introduce a small example that can be understood manually.

Examples should include:
- Small graphs.
- Adjacency matrices.
- Node and edge features.
- Intermediate tensor operations.
- Neighbourhood aggregation.

The reader should understand what the code is doing, not just execute it.

### 5. Implementation from Scratch

Whenever it helps understanding, implement the central mathematical operation using NumPy or basic PyTorch.

Gradually introduce PyTorch Geometric and other libraries after explaining the underlying idea.

Code should be:
- Modular and readable.
- Well commented.
- Divided into manageable sections.
- Accompanied by explanations of important operations.
- Explicit about tensor dimensions and transformations.

The objective is understanding the underlying algorithm, not unnecessarily recreating every library feature.

### 6. Model Training and Evaluation

For tutorials involving trainable models, explain:
- Input and output representations.
- Model architecture.
- Loss functions.
- Optimizers.
- Forward and backward propagation.
- Training, validation, and testing.
- Evaluation metrics.

Include training curves and relevant visualisations where they help interpret the results.

### 7. Hands-on Exercises

Every tutorial should include practical exercises that encourage experimentation.

Use the following consistent format:

**🧪 Hands-on Exercise**

Exercises should encourage readers to modify the graph, change model parameters, implement an operation, or compare different architectures.

Include hints and expected qualitative observations where appropriate.

### 8. Limitations and Practical Considerations

Discuss the limitations of the method being introduced.

Depending on the topic, this may include:
- Computational and memory requirements.
- Graph construction choices.
- Scalability.
- Oversmoothing and oversquashing.
- Generalisation to unseen graphs.
- Data leakage and evaluation design.
- Physical consistency in engineering applications.

### 9. Final Challenge

Conclude with a small project that combines the concepts introduced in the tutorial.

The challenge should encourage independent implementation and experimentation.

### 10. Summary and Next Tutorial

Finish with:
- A concise summary of the concepts covered.
- A checklist of learning outcomes.
- A preview of the next tutorial.

---

## Technology Stack

The series primarily uses the following tools:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Mathematical foundations and graph operations |
| Matplotlib | Graph and training visualisations |
| PyTorch | Neural networks, automatic differentiation and training |
| PyTorch Geometric | Graph datasets, graph layers and message passing |
| Jupyter Notebook | Interactive tutorials and experiments |

### Environment Setup

Create a Python virtual environment:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the basic dependencies:

```bash
python -m pip install --upgrade pip

pip install numpy matplotlib jupyter
```

Install PyTorch according to your system configuration:

[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

Install PyTorch Geometric following its official installation instructions:

[PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

*Note: PyTorch and PyTorch Geometric versions should be compatible with your Python version, operating system, and CUDA configuration.*

---

## Who Is This Series For?

This series is intended for:

- Students and researchers interested in graph representation learning.
- Engineers who want to explore deep learning beyond conventional neural networks.
- Python users interested in implementing GNN architectures.
- Researchers working in scientific machine learning.
- People interested in computational physics, CFD, computational mechanics, and surrogate modelling.

You do not need previous experience with Graph Neural Networks.

Basic Python programming, linear algebra, and familiarity with neural networks will make the learning process easier.

---

## Why GNNs for Engineering and Scientific Machine Learning?

Many engineering problems naturally involve irregular and interconnected structures.

Consider an unstructured computational mesh used in CFD.

We can represent it as a graph:

| Graph Concept | Possible Engineering Interpretation |
|---|---|
| Node | Cell centre, mesh vertex, or control volume |
| Edge | Cell adjacency or mesh connectivity |
| Node features | Pressure, velocity, temperature, turbulence quantities |
| Edge features | Distance, relative position, face area, orientation |
| Message | Learned interaction between neighbouring entities |
| Aggregation | Combination of information from neighbouring cells |
| Node embedding | Learned representation of the local physical environment |

This creates interesting opportunities for:
- Mesh-based surrogate modelling.
- Reduced-order modelling.
- Learning physical interactions.
- Fast approximations of computational simulations.
- Scientific machine learning on irregular domains.

However, a learned message-passing operation is not automatically a physically conservative numerical method.

Conservation laws, boundary conditions, geometric consistency, numerical stability, and generalisation must be considered explicitly when applying GNNs to physical systems.

The final tutorials will explore these connections and demonstrate how graph learning can become a useful component of engineering workflows.

---

## How to Use This Repository

1. Start with Tutorial 1 to understand the fundamentals.
2. Run each notebook sequentially.
3. Read the mathematical explanations before executing the implementation.
4. Modify the examples and complete the exercises.
5. Experiment with different graph structures and model parameters.
6. Record and compare your results.
7. Move to the next tutorial once you understand the concepts introduced.

These notebooks are designed for active learning. Running the code is only the beginning; experimenting with it and understanding the results is where the real learning happens.

---

## Contributions and Feedback

Suggestions, corrections, and improvements are welcome.

If you find an issue or have an idea for improving a tutorial:
- Open a GitHub issue.
- Mention the relevant tutorial number and section.
- Include error messages and environment information for technical issues.
- Submit a pull request if you would like to contribute directly.

---

## Author

Created by **Rohit Raj** as part of *The Learning Stack* educational initiative.

The series explores graph learning, scientific machine learning, computational physics, and engineering applications.

- **GitHub:** [rraj-io](https://github.com/rraj-io)
- **Substack:** The Learning Stack — add publication URL
- **YouTube:** Add channel URL

---

## License

The repository license will be specified in the `LICENSE` file.

Please check the license before reusing or redistributing the notebooks and code.

---

**Learn the graph. Understand the message passing. Build the model. Connect it to the physical world.** 