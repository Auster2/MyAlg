\documentclass{article}
\usepackage{amsmath, amssymb, amsthm, mathtools}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{cite}  % 引入 citation 包

\title{Structure-Constrained Multi-Objective Optimization Problems (SCMOP)}
\author{}
\date{}

\begin{document}

\maketitle

\section{Structure-Constrained Multi-Objective Optimization Problems}
    \subsection{Traditional Multi-objective Optimization Problems (MOPs)} 
    Multi-objective optimization problems (MOPs) involve the simultaneous optimization of multiple conflicting objectives. Given a decision space $\mathcal{X} \subseteq \mathbb{R}^n$ and an objective function $\mathbf{f} : \mathcal{X} \to \mathbb{R}^m$, a standard MOP is formulated as:
    \begin{equation}
    \min_{x \in \mathcal{X}} \; \mathbf{f}(x) = [f_1(x), f_2(x), \dots, f_m(x)]^\top,
    \label{eq:standard_moo}
    \end{equation}
    Due to the inherent conflicts among the objectives, a single globally optimal solution rarely exists. Instead, the solution is characterized by a set of non-dominated solutions, known as the Pareto-optimal set, denoted by $\mathcal{P}^* \subseteq \mathcal{X}$. The image of this set under the mapping $\mathbf{f}$ is referred to as the Pareto front, denoted by $\mathcal{F}^* = \{ \mathbf{f}(x) \mid x \in \mathcal{P}^* \}$.

    Classical multi-objective optimization primarily focuses on approximating $\mathcal{F}^*$ using a set of solutions that are diverse and convergent. However, in many practical applications, the structure of the solution set itself plays an important role and may be subject to additional constraints.

    \subsection{Structure-Constrained Multi-objective Optimization Problems (SCMOP)}
    In a variety of domains, such as product design, autonomous systems, or neural architecture search, solutions are not deployed in isolation. Rather, a set of solutions is selected and used jointly. In such contexts, it is often desirable for the selected solution set to exhibit certain structural properties, such as common components, coordinated variable patterns, or functional consistency. These requirements motivate the study of structure-constrained multi-objective optimization, in which the optimization objective is to identify a high-quality set of solutions that also adheres to domain-specific structural constraints.

    Let $\mathcal{S} \subseteq \mathcal{X}$ denote a candidate solution set with fixed cardinality $|\mathcal{S}| = K$. The objective is to select such a set $\mathcal{S}$ that achieves high performance in the objective space while satisfying predefined structure constraints. This leads to the following formulation:
    \begin{equation}
    \begin{aligned}
    \max_{\mathcal{S} \subseteq \mathcal{X}} \quad & Q(\mathcal{S}) \\
    \text{subject to} \quad & |\mathcal{S}| = K, \\
                            & g(\mathcal{S}) \leq 0,\\
                            & h(\mathcal{S}) = 0, \\
    \end{aligned}
    \label{eq:scmoo}
    \end{equation}
    where $Q(\mathcal{S})$ is a performance indicator measuring the quality of the solution set $\mathcal{S}$ in the objective space. Common choices include hypervolume (HV), inverted generational distance (IGD), or other performance metrics. $g(\mathcal{S})$ and $h(\mathcal{S})$ denote the structure inequality and equality constraints, respectively. 

    Clearly, this formulation generalizes classical MOP by shifting the focus from the optimization of individual solutions to the selection of structurally consistent solution sets.

    \subsection{Examples of Structure Constraints}
    The structure constraints imposed on the solution set $\mathcal{S}$ can take various forms depending on the characteristics of the application domain. These constraints typically reflect desired regularities, dependencies, or shared features among the solutions. Below, we outline two representative types of structure constraints.

    \begin{itemize}
        \item \textbf{Shared Variable Constraint:}  
        In certain applications, it is required that all solutions in the set $\mathcal{S}$ share common values for specific decision variables. For example, in modular product design or manufacturing, a common component or configuration may be preferred across all selected solutions. Let $k \in \{1, \dots, n\}$ denote the index of a decision variable to be shared. The variable shared structure constraint is expressed as:
        \begin{equation}
            h(\mathcal{S}) = \sum_{x \in \mathcal{S}} \left| x_k - \bar{x}_k \right|=0, 
        \end{equation}
        where $\bar{x}_k = \frac{1}{K} \sum_{x' \in \mathcal{S}} x_k'$ is the mean value of variable $x_k$ across all solutions in the set. The constraint can be extended to multiple dimensions by aggregating across all designated indices.

        \item \textbf{Functional Dependency Constraint:}  
        In some problem domains, specific decision variables are expected to be determined by a functional relationship with others. Let $x_k$ be a decision variable that follows a deterministic mapping $x_k = g(x_1, x_2, \dots, x_{k-1}; \theta)$, where $g$ is a known function and $\theta$ is a fixed parameter shared within the set. The corresponding structure constraint can be defined as:
        \begin{equation}
            h(\mathcal{S}) = \sum_{x \in \mathcal{S}} \left| x_k - g(x_1, \dots, x_{k-1}; \theta) \right|=0,
            \label{eq:csshared}
        \end{equation}
        This constraint enforces functional coherence within the solution set by ensuring that all solutions respect a common generative rule. It is particularly relevant in settings where part of the design space is governed by physical laws, domain knowledge, or predefined architectural rules.
    \end{itemize}

    \subsection{Implications and Challenges of SCMOP}
    The introduction of structure constraints fundamentally transforms the nature of multi-objective optimization. Unlike classical MOP, where solutions are evaluated independently, SCMOP must consider interactions and dependencies among solutions within a set. This shift induces several significant challenges:
    \begin{itemize}
        \item \textit{Non-separability of evaluation:} The quality of an individual solution depends on the collective behavior of the entire set. This violates the assumption of independent evaluation and renders traditional selection and update strategies less effective.    
        \item \textit{Increased combinatorial complexity:} The optimization process must simultaneously ensure Pareto-optimality and structural feasibility at the set level. This results in a substantially larger and more intricate search space, particularly when the constraints induce coupling among decision variables.   
        \item \textit{Constraint handling:} Structural constraints are often non-convex, discontinuous, or implicitly defined. As a result, they are difficult to encode and enforce using standard constraint-handling techniques, necessitating the development of specialized structure constraint handling techniques.  
        \item \textit{Lack of benchmark instances:} Existing MOP benchmarks typically focus on convergence and diversity in the objective space without considering structural properties. The absence of widely accepted SCMOP benchmarks limits the evaluation and comparison of algorithmic approaches.   
        \item \textit{Limited algorithmic support:} Currently, there are few optimization algorithms specifically designed to address the dual requirements of objective performance and structural consistency. Most existing MOO methods are not directly applicable or effective under structure constraints, highlighting a significant gap in the literature.
    \end{itemize}
    These challenges suggest that SCMOP is a non-trivial generalization of classical MOP and demands novel theoretical models, algorithmic frameworks, and evaluation protocols.

    \subsection{SCMOP Benchmark Instances}
    Structure-constrained multi-objective optimization problems (SCMOPs) present significant challenges in both formulation and solution. The lack of widely accepted benchmark instances for SCMOPs has hindered the development and evaluation of algorithms designed to solve them. Recently, work on addressing structure constraints in evolutionary Pareto set learning has suggested the importance of benchmarks specifically designed for this type of optimization problem \cite{dealing_structure_constraints}. These instances focus on structure-constrained optimization and provide a foundation for further exploration of SCMOP algorithms.

\bibliographystyle{plain}
\bibliography{references}
\end{document}
