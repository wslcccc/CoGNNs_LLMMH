# AI4DSE: Leveraging Graph Neural Networks and Large Language Models for Optimizing High-Level Synthesis Design Space  

## Content
- [About the project](#jump1)
- [Project File Tree](#jump2)
- [Required environment](#jump3)

## <span id="jump1">About the project</span>

This project is an automated framework. By using GNNs as an alternative model to the HLS tool, it predicts the QoR results and utilizes the meta-heuristic algorithms (GA, SA, ACO) assisted by LLMs to identify the Pareto optimal solution with higher quality. Inspired by the works proposed in [CoGNN](https://github.com/benfinkelshtein/CoGNN) and [LMEA](https://github.com/cschen1205/LMEA), we propose a LLM-driven metaheuristic framework for HLS DSE, which takes advantage of LLMs' powerful information comprehension capabilities to interpret the functional roles of various pragmas and their impacts on final routing outcomes.

![The framework of CoGNNs_LLMMH](framework.jpeg)

### Contribution
- We adopt a novel graph neural network architecture (CoGNNs) with an adaptive message-passing mechanism as our core prediction models for accurate QoR prediction. Unlike traditional GNNs with fixed message-passing rules, CoGNNs dynamically reconstructs graph topologies based on task objectives, enabling targeted feature propagation and mitigating over-smoothing. The adapted CoGNNs can be tuned to predict both post-HLS and post-implementation QoR metrics with the highest accuracy compared to SOTA works.
- We propose a LLM-driven metaheuristic (LLMMH) framework for efficient DSE. The targeting meta-heuristic includes GA, SA and ACO. By leveraging LLMs’ learning and reasoning capabilities with carefully engineered prompts, LLMMH significantly reduces domain expertise dependency and achieves prominent improvement in Pareto front quality compared to baseline algorithms.
- To the best of our knowledge, the proposed LLMMH is the first framework that integrates LLM with meta-heuristic algorithms for HLS DSE. This approach represents a significant shift from prior methods in HLS DSE. Notably, it has demonstrated promising ability to enhance the quality of DSE outcomes. We hope that our findings will inspire further exploration of LLM-based techniques to tackle challenges in HLS DSE.

## <span id="jump2">Project File Tree</span>
```
|--CoGNNs_LLMMH
|--baseline                        
|--baseline_dataset    
|   +--CoGNN                       # components of CoGNNs
|   +--action_gumbel_layer.py
|   +--layer.py
|   +--model_parse.py
|--dse_database                    # database, graphs, codes to generate/analyze them
|--LLM_MH                          # the algorithm details of the LLMMH framework and the LLM call code
|   +--dse.py
|   +--LLM.py
|--src                             # the source codes for defining and training the model and running the DSE
|   +--config.py
|   +--config_ds.py
|   +--main.py
|   +--model.py
|   +--nn_att.py
|   +--parallel_run_tool_dse.py
|   +--parameter.py
|   +--programl_data.py
|   +--result.py
|   +--saver.py
|   +--train.py
|   +--utils.py
```

## <span id="jump3">Required environment</span>
- os Linux
- python 3.9.22
- torch 1.12.1+cu118
- torch_geometric 2.2.0
- openai 0.28.1
- numpy 1.23.5
- langchain 0.1.16
