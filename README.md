# Combinatorial Optimization for All

![alt image](static/images/abstract.png)

**Abstract:** Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills.

Homepage: [https://camilochs.github.io/comb-opt-for-all/](https://camilochs.github.io/comb-opt-for-all/)

## Contents of This Repository

### 1. Prompts and Generated Algorithms
- The [`prompt_template`](prompts/) and the prompts used to generate the 10 algorithms are located in the [`prompts/`](prompts/) directory.
- The algorithms generated for each LLM (Claude-3.5-Sonnet, GPT-O1, Llama-3.3-70b, Gemini-exp-1206, DeepSeek-R1) are stored in [`algorithms_improved_LLMs/`](algorithms_improved_LLMs/).  
  - **Note:** If an LLM failed to generate an algorithm (see Table 2 of the paper), the corresponding prompt has been omitted, and only the final algorithm is included.

### 2. Original Algorithms
- The 10 original algorithms were extracted from the **[pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)** framework (version from late February 2025).  
  - Special thanks to **Valdecy Pereira**, the creator of the framework, for kindly addressing our inquiries.

### 3. Experimental Results
- The results of all experiments, for each algorithm, are available in the [`results/`](results/) directory.

### 4. Algorithm Instances
- The instances for each algorithm are stored in the [`instances/`](instances/) directory.  
- Specific instances for the *branch and bound* algorithm can be found in [`instances/branch_and_bound/`](instances/branch_and_bound/).

> LLMs can assist not only researchers in enhancing their optimization algorithms but also non-experts seeking quick and efficient solutions, as well as those using them for educational purposes.

## Research Findings

Our research is detailed in the paper:
**[Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms](https://www.alphaxiv.org/abs/2503.10968)** – Check it out!

## Cite

```
@misc{sartori2025combinatorialoptimizationallusing,
          title={Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms}, 
          author={Camilo Chacón Sartori and Christian Blum},
          year={2025},
          eprint={2503.10968},
          archivePrefix={arXiv},
          primaryClass={cs.AI},
          url={https://arxiv.org/abs/2503.10968}, 
        }
```
