---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

Hi there! 

Welcome to Xinyin Ma(马欣尹)'s website!
I am currently a Ph.D candidate @ [xML-Lab](https://sites.google.com/view/xml-nus), National University of Singapore from August 2022, advised by [Prof.Xinchao Wang](https://sites.google.com/site/sitexinchaowang/). Previously I obtained my master degree in computer science from Zhejiang University, advised by [Prof.Weiming Lu](https://person.zju.edu.cn/en/lwm). I obtained my bachelor degree in software engineering also in Zhejiang University and got the honor degree from Chu Kochen College. 
I'm so honored to receive the [***Google PhD Fellowship***](https://research.google/programs-and-events/phd-fellowship/recipients/) in 2024.

Currently, I'm conducting research in efficient deep learning ([Google Scholar](https://scholar.google.com/citations?user=jFUKS0oAAAAJ&hl=en)), including:  

🌲 Efficient Large Language Models, Reasoning Models and Diffusion Language Models

🌱 Efficient Diffusion Models, mainly for the cache inference paradigm

🌿 Data-centric Compression, e.g., Data-free Distillation, Dataset Distillation

<div style="background-color:rgb(236, 236, 236); padding: 10px; border-left: 5px solid rgb(2, 47, 92); margin-top: 15px; margin-bottom: 15px;">
    <span style="color:rgb(34, 75, 141)"><strong>
    I'm expected to graduate before June 2026 and am currently on the job market (for both academic and industrial opportunities). 
    I would greatly appreciate it if you could email me about any available opportunities!
    </strong>
    </span>
</div>


# 🔥 News
- *2025.05*: &nbsp;We release [dKV-Cache](https://arxiv.org/abs/2505.15781) for the first KV-Cache algorithm for diffusion language models! 
- *2025.05*: &nbsp;[CoT-Valve](https://arxiv.org/abs/2502.09601) is accepted by ACL'25! See you in Vienna!
- *2025.02*: &nbsp;Three papers ([SSD](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Diffusion_Model_is_Effectively_Its_Own_Teacher_CVPR_2025_paper.pdf), [CoDe](https://arxiv.org/abs/2411.17787) and [TinyFusion](https://arxiv.org/abs/2412.01199)) accepted by CVPR'25.
- *2025.02*: &nbsp;Two new papers released! Check [CoT-Valve](https://arxiv.org/abs/2502.09601) for controllable and compressible CoT and [VPT](https://arxiv.org/abs/2502.17425) for multimodal reasoning!
- *2025.02*: &nbsp;Co-organize the [2nd workshop on Efficient Large Vision Models](https://sites.google.com/view/elvm/home), CVPR'25.
- *2025.01*: &nbsp;Invited talk at KAUST Rising Stars in AI Symposium 2025, April 7 - 10. 
- *2024.11*: &nbsp;🥳 Awarded Google PhD Fellowship
- *2024.09*: &nbsp;Four papers ([Learning-to-Cache](https://arxiv.org/abs/2406.01733), [AsyncDiff](https://arxiv.org/abs/2406.06911), [SlimSAM](https://arxiv.org/abs/2312.05284) and [RemixDiT](https://openreview.net/forum?id=vo5LONGAdo)) accepted by NeurIPS'24! See you in Vancouver!
- *2024.02*: &nbsp;DeepCache is accepted by CVPR'24! 
- *2023.12*: &nbsp;🌟 Our new work, DeepCache, accelerates Diffusion Models for FREE! Check our [paper](https://arxiv.org/abs/2312.00858) and [code](https://github.com/horseee/DeepCache)! 
- *2023.06*: &nbsp;🎉 Release LLM-Pruner🐏, the first structural pruning work of LLM. See our [paper](https://arxiv.org/abs/2305.11627) and [code](https://github.com/horseee/LLM-Pruner)! 
- *2022.08*: &nbsp;⛵ Start my Ph.D. journey in NUS!
- *2022.04*: &nbsp; One paper ‘Prompting to distill: Boosting Data-Free Knowledge Distillation via Reinforced Prompt’ accepted by IJCAI’22.
- *2022.04*: &nbsp; Got my master degree from ZJU! Thanks to my supervisor and all my friends in ZJU!

# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Preprint</div><img src='https://github.com/horseee/dKV-Cache/blob/main/assets/teaser.gif?raw=true' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**dKV-Cache: The Cache for Diffusion Language Models**](https://arxiv.org/abs/2505.15781) <img src='https://img.shields.io/github/stars/horseee/dKV-Cache.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Runpeng Yu, Gongfan Fang, Xinchao Wang 

- Delayed Caching Mechanism: dKV-Cache delays the caching of keys and values.
- Two Variants: (1) dKV-Cache-Decode for high-performance inference. (2) dKV-Cache-Greedy for potentially faster decoding with trade-off in performance.
- Applied to LLaDA and Dream, dKV-Cache achieves 2x to 10x speedups.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2505.15781"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/dKV-Cache"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Diffusion Language Models (DLMs) have been seen as a promising competitor for autoregressive language models. However, diffusion language models have long been constrained by slow inference. A core challenge is that their non-autoregressive architecture and bidirectional attention preclude the key-value cache that accelerates decoding. We address this bottleneck by proposing a KV-cache-like mechanism, delayed KV-Cache, for the denoising process of DLMs. Our approach is motivated by the observation that different tokens have distinct representation dynamics throughout the diffusion process. Accordingly, we propose a delayed and conditioned caching strategy for key and value states. We design two complementary variants to cache key and value step-by-step: (1) dKV-Cache-Decode, which provides almost lossless acceleration, and even improves performance on long sequences, suggesting that existing DLMs may under-utilise contextual information during inference. (2) dKV-Cache-Greedy, which has aggressive caching with reduced lifespan, achieving higher speed-ups with quadratic time complexity at the cost of some performance degradation. dKV-Cache, in final, achieves from 2-10x speedup in inference, largely narrowing the gap between ARs and DLMs. We evaluate our dKV-Cache on several benchmarks, delivering acceleration across general language understanding, mathematical, and code-generation benchmarks. Experiments demonstrate that cache can also be used in DLMs, even in a training-free manner from current DLMs. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2025</div><img src='images/papers/cot-valve.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**CoT-Valve: Length-Compressible Chain-of-Thought Tuning**](https://arxiv.org/abs/2502.09601) <img src='https://img.shields.io/github/stars/horseee/CoT-Valve.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma\***, Guangnian Wan\*, Runpeng Yu, Gongfan Fang, Xinchao Wang 

(*Equal Contribution)

- A tuning and inference strategy that elastically controls CoT length within a single model
- GSM8K: 741 → 225 tokens with only 0.15% accuracy drop. AIME: 6827 → 4629 tokens (32% reduction) while preserving accuracy

<div style="display: inline">
    <a href="https://arxiv.org/abs/2502.09601"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/CoT-Valve"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Chain-of-Thought significantly enhances a model's reasoning capability, but it also comes with a considerable increase in inference costs due to long chains. With the observation that the reasoning path can be easily compressed under easy tasks but struggle on hard tasks, we explore the feasibility of elastically controlling the length of reasoning paths with only one model, thereby reducing the inference overhead of reasoning models dynamically based on task difficulty. We introduce a new tuning and inference strategy named CoT-Valve, designed to allow models to generate reasoning chains of varying lengths. To achieve this, we propose to identify a direction in the parameter space that, when manipulated, can effectively control the length of generated CoT. Moreover, we show that this property is valuable for compressing the reasoning chain. We construct datasets with chains from long to short for the same questions and explore two enhanced strategies for CoT-Valve: (1) a precise length-compressible CoT tuning method, and (2) a progressive chain length compression approach. Our experiments show that CoT-Valve successfully enables controllability and compressibility of the chain and shows better performance than the prompt-based control. We applied this method to QwQ-32B-Preview, reducing reasoning chains on GSM8K from 741 to 225 tokens with a minor performance drop (95.07% to 94.92%) and on AIME from 6827 to 4629 tokens, with only one additional incorrect answer. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2024</div><img src='https://github.com/horseee/learning-to-cache/raw/main/assets/teaser.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching**](https://arxiv.org/abs/2406.01733) <img src='https://img.shields.io/github/stars/horseee/Learning-to-Cache.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Gongfan Fang, Michael Bi Mi, Xinchao Wang

- A novel scheme that learns to conduct caching in a dynamic manner for diffusion transformers.
- A large proportion of layers in the diffusion transformer can be removed, without updating the model parameters.
- Learning-to-Cache largely outperforms samplers such as DDIM and DPM-Solver.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2406.01733"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/learning-to-cache"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Diffusion Transformers have recently demonstrated unprecedented generative capabilities for various tasks. The encouraging results, however, come with the cost of slow inference, since each denoising step requires inference on a transformer model with a large scale of parameters. In this study, we make an interesting and somehow surprising observation: the computation of a large proportion of layers in the diffusion transformer, through introducing a caching mechanism, can be readily removed even without updating the model parameters. In the case of U-ViT-H/2, for example, we may remove up to 93.68% of the computation in the cache steps (46.84% for all steps), with less than 0.01 drop in FID. To achieve this, we introduce a novel scheme, named Learning-to-Cache (L2C), that learns to conduct caching in a dynamic manner for diffusion transformers. Specifically, by leveraging the identical structure of layers in transformers and the sequential nature of diffusion, we explore redundant computations between timesteps by treating each layer as the fundamental unit for caching. To address the challenge of the exponential search space in deep models for identifying layers to cache and remove, we propose a novel differentiable optimization objective. An input-invariant yet timestep-variant router is then optimized, which can finally produce a static computation graph. Experimental results show that L2C largely outperforms samplers such as DDIM and DPM-Solver, alongside prior cache-based methods at the same inference speed. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024</div><img src='https://github.com/horseee/DeepCache/blob/master/assets/intro.png?raw=true' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**DeepCache: Accelerating Diffusion Models for Free**](https://arxiv.org/abs/2312.00858) <img src='https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Gongfan Fang, Xinchao Wang

- A training-free paradigm that accelerates diffusion models
- Utilizes the U-Net's properties to efficiently reuse high-level features and update low-level features
- 2.3× speedup for Stable Diffusion v1.5 and a 4.1× speedup for LDM-4-G, based upon DDIM/PLMS

<div style="display: inline">
    <a href="https://arxiv.org/abs/2312.00858"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/DeepCache"> <strong>[code]</strong></a>
    <a href="https://horseee.github.io/Diffusion_DeepCache/"> <strong>[Project Page]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3× for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1× for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023</div><img src='images/papers/llm-pruner.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LLM-Pruner: On the Structural Pruning of Large Language Models**](https://arxiv.org/abs/2305.11627) <img src='https://img.shields.io/github/stars/horseee/LLM-Pruner.svg?style=social&label=Star' alt="sym" height="100%">

**Xinyin Ma**, Gongfan Fang, Xinchao Wang

- Task-agnostic Compression: The compressed LLM retain its multi-task ability.
- Less Training Corpus: We use only 50k samples to post-train the LLM.
- Efficient Compression: 3 minutes for pruning and 3 hours for post-training. 
- Automatic Structural Pruning: Pruning new LLMs with minimal human effort.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2305.11627"> <strong>[paper]</strong></a>
    <a href="https://github.com/horseee/LLM-Pruner"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-Pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality. To this end, the performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data. We validate the LLM-Pruner on three LLMs, including LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still exhibit satisfactory capabilities in zero-shot classification and generation. </p>
    </div>
</div>

</div>
</div>

<ul>
  <li>
    <strong> Diffusion Model is Effectively Its Own Teacher. CVPR 2025. </strong>
    <div style="display: inline">
        <a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Diffusion_Model_is_Effectively_Its_Own_Teacher_CVPR_2025_paper.pdf"> [paper]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> In this paper, we introduce a novel self-distillation paradigm for improving the performance of diffusion models. Previous studies have shown that introducing a teacher to distill the diffusion model can enhance its sampling efficiency. We raise an intriguing question: can the diffusion model itself serve as its teacher to further improve the performance of itself? To this end, we propose a new paradigm called Self Step-Distillation (SSD). The core idea of SSD is to integrate the predictions or the intermediate activations of the diffusion model at each timestep with its preceding timestep through a fusion mechanism. We propose two forms, explicit SSD and implicit SSD (iSSD), to perform N-step to N-step distillation from the diffusion model itself to achieve improved image quality. We further elucidate the relationship between SSD and high-order solver, highlighting their underlying relationship. The effectiveness of SSD is validated through extensive experiments on diffusion transformers of various sizes and across different sampling steps. Our results show that this novel self-distillation paradigm can significantly enhance performance. Additionally, our method is compatible with the distillation method designed for few-step inference. Notably, with iSSD trained less than one epoch, we obtain a 32-step DiT-XL/2 achieving an FID of 1.99, outperforming the original 250-step DiT-XL/2 with an FID of 2.26. We further validate the effectiveness of our method on text-to-image diffusion models, such as Stable Diffusion, and also observe notable improvement in image quality. </p>
        </div>
    </div>
    <div><i><strong>Xinyin Ma</strong>, Runpeng Yu, Songhua Liu, Gongfan Fang, Xinchao Wang. </i></div>
  </li>

  <li>
    <strong> Prompting to distill: Boosting Data-Free Knowledge Distillation via Reinforced Prompt. IJCAI 2022. </strong>
    <div style="display: inline">
        <a href="https://www.ijcai.org/proceedings/2022/0596.pdf"> [paper]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Data-free knowledge distillation (DFKD) conducts knowledge distillation via eliminating the dependence of original training data, and has recently achieved impressive results in accelerating pre-trained language models. At the heart of DFKD is toreconstruct a synthetic dataset by invertingthe parameters of the uncompressed model. Prior DFKD approaches, however, havelargely relied on hand-crafted priors of the target data distribution for the reconstruction, which can be inevitably biased and often incompetent to capture the intrinsic distributions. To address this problem, we propose a prompt-based method, termed as PromptDFD, that allows us to take advantage of learned language priors, which effectively harmonizes the synthetic sentences to be semantically and grammatically correct. Specifically, PromptDFD leverages a pre-trained generative model to provide language priors and introduces a reinforced topic prompter to control data synthesis, making the generated samples thematically relevant and semantically plausible, and thus friendly to downstream tasks. As shown in our experiments, the proposed method substantially improves the synthesis quality and achieves considerable improvements on distillation performance. In some cases, PromptDFD even gives rise to results on par with those from the data-driven knowledge distillation with access to the original training data. </p>
        </div>
    </div>
    <div><i><strong>Xinyin Ma</strong>, Xinchao Wang, Gongfan Fang, Yongliang Shen, Weiming Lu. </i></div>
  </li>
  
  <li>
   <strong> MuVER: Improving First-Stage Entity Retrieval with Multi-View Entity Representations. EMNLP 2021 Short. </strong>
    <div style="display: inline">
        <a href="https://aclanthology.org/2021.emnlp-main.205.pdf"> [paper]</a>
        <a href="https://github.com/alibaba-nlp/muver"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Entity retrieval, which aims at disambiguating mentions to canonical entities from massive KBs, is essential for many tasks in natural language processing. Recent progress in entity retrieval shows that the dual-encoder structure is a powerful and efficient framework to nominate candidates if entities are only identified by descriptions. However, they ignore the property that meanings of entity mentions diverge in different contexts and are related to various portions of descriptions, which are treated equally in previous works. In this work, we propose Multi-View Entity Representations (MuVER), a novel approach for entity retrieval that constructs multi-view representations for entity descriptions and approximates the optimal view for mentions via a heuristic searching method. Our method achieves the state-of-the-art performance on ZESHEL and improves the quality of candidates on three standard Entity Linking datasets. </p>
        </div>
    </div>
    <div><i><strong>Xinyin Ma</strong>, Yong Jiang, Nguyen Bach, Tao Wang, Zhongqiang Huang, Fei Huang, Weiming Lu.</i></div>
  </li>

  <li>
   <strong> Adversarial Self-Supervised Data-Free Distillation for Text Classification. EMNLP 2020. </strong>
    <div style="display: inline">
        <a href="https://aclanthology.org/2020.emnlp-main.499.pdf"> [paper]</a>
        <a href="https://slideslive.com/38938706/adversarial-selfsupervised-datafree-distillation-for-text-classification"> [video]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Large pre-trained transformer-based language models have achieved impressive results on a wide range of NLP tasks. In the past few years, Knowledge Distillation(KD) has become a popular paradigm to compress a computationally expensive model to a resource-efficient lightweight model. However, most KD algorithms, especially in NLP, rely on the accessibility of the original training dataset, which may be unavailable due to privacy issues. To tackle this problem, we propose a novel two-stage data-free distillation method, named Adversarial self-Supervised Data-Free Distillation (AS-DFD), which is designed for compressing large-scale transformer-based models (e.g., BERT). To avoid text generation in discrete space, we introduce a Plug & Play Embedding Guessing method to craft pseudo embeddings from the teacher’s hidden knowledge. Meanwhile, with a self-supervised module to quantify the student’s ability, we adapt the difficulty of pseudo embeddings in an adversarial training manner. To the best of our knowledge, our framework is the first data-free distillation framework designed for NLP tasks. We verify the effectiveness of our method on several text classification datasets. </p>
        </div>
    </div>
    <div><i><strong>Xinyin Ma</strong>, Yongliang Shen, Gongfan Fang, Chen Chen, Chenghao Jia, Weiming Lu.</i></div>
  </li>
  <li>
    <strong> Introducing Visual Perception Token into Multimodal Large Language Model. ICCV 2025</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2502.17425"> [paper]</a>
        <a href="https://github.com/yu-rp/VisualPerceptionToken"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> To utilize visual information, Multimodal Large Language Model (MLLM) relies on the perception process of its vision encoder. The completeness and accuracy of visual perception significantly influence the precision of spatial reasoning, fine-grained understanding, and other tasks. However, MLLM still lacks the autonomous capability to control its own visual perception processes, for example, selectively reviewing specific regions of an image or focusing on information related to specific object categories. In this work, we propose the concept of Visual Perception Token, aiming to empower MLLM with a mechanism to control its visual perception processes. We design two types of Visual Perception Tokens, termed the Region Selection Token and the Vision Re-Encoding Token. MLLMs autonomously generate these tokens, just as they generate text, and use them to trigger additional visual perception actions. The Region Selection Token explicitly identifies specific regions in an image that require further perception, while the Vision Re-Encoding Token uses its hidden states as control signals to guide additional visual perception processes. Extensive experiments demonstrate the advantages of these tokens in handling spatial reasoning, improving fine-grained understanding, and other tasks. On average, the introduction of Visual Perception Tokens improves the performance of a 2B model by 23.6%, increasing its score from 0.572 to 0.708, and even outperforms a 7B parameter model by 13.4% (from 0.624).  </p>
        </div>
    </div>
    <div><i>Runpeng Yu*, <strong>Xinyin Ma*</strong>, Xinchao Wang (*Equal Contribution) </i></div>
  </li>
  <li>
    <strong> Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient. CVPR 2025</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2411.17787"> [paper]</a>
        <a href="https://github.com/czg1225/CoDe"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> In the rapidly advancing field of image generation, Visual Auto-Regressive (VAR) modeling has garnered considerable attention for its innovative next-scale prediction approach. This paradigm offers substantial improvements in efficiency, scalability, and zero-shot generalization. Yet, the inherently coarse-to-fine nature of VAR introduces a prolonged token sequence, leading to prohibitive memory consumption and computational redundancies. To address these bottlenecks, we propose Collaborative Decoding (CoDe), a novel efficient decoding strategy tailored for the VAR framework. CoDe capitalizes on two critical observations: the substantially reduced parameter demands at larger scales and the exclusive generation patterns across different scales. Based on these insights, we partition the multi-scale inference process into a seamless collaboration between a large model and a small model. The large model serves as the 'drafter', specializing in generating low-frequency content at smaller scales, while the smaller model serves as the 'refiner', solely focusing on predicting high-frequency details at larger scales. This collaboration yields remarkable efficiency with minimal impact on quality: CoDe achieves a 1.7x speedup, slashes memory usage by around 50%, and preserves image quality with only a negligible FID increase from 1.95 to 1.98. When drafting steps are further decreased, CoDe can achieve an impressive 2.9x acceleration ratio, reaching 41 images/s at 256x256 resolution on a single NVIDIA 4090 GPU, while preserving a commendable FID of 2.27 </p>
        </div>
    </div>
    <div><i>Zigeng Chen, <strong>Xinyin Ma</strong>, Gongfan Fang, Xinchao Wang.</i></div>
  </li>
  <li>
    <strong> TinyFusion: Diffusion Transformers Learned Shallow. CVPR 2025</strong>. 
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2412.01199"> [paper]</a>
        <a href="https://github.com/VainF/TinyFusion"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Diffusion Transformers have demonstrated remarkable capabilities in image generation but often come with excessive parameterization, resulting in considerable inference overhead in real-world applications. In this work, we present TinyFusion, a depth pruning method designed to remove redundant layers from diffusion transformers via end-to-end learning. The core principle of our approach is to create a pruned model with high recoverability, allowing it to regain strong performance after fine-tuning. To accomplish this, we introduce a differentiable sampling technique to make pruning learnable, paired with a co-optimized parameter to simulate future fine-tuning. While prior works focus on minimizing loss or error after pruning, our method explicitly models and optimizes the post-fine-tuning performance of pruned models. Experimental results indicate that this learnable paradigm offers substantial benefits for layer pruning of diffusion transformers, surpassing existing importance-based and error-based methods. Additionally, TinyFusion exhibits strong generalization across diverse architectures, such as DiTs, MARs, and SiTs. Experiments with DiT-XL show that TinyFusion can craft a shallow diffusion transformer at less than 7% of the pre-training cost, achieving a 2× speedup with an FID score of 2.86, outperforming competitors with comparable efficiency </p>
        </div>
    </div>
    <div><i>Gongfan Fang, Kunjun Li, <strong>Xinyin Ma</strong>, Xinchao Wang.</i></div>
  </li>
  <li>
    <strong> AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising. NeurIPS 2024. </strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2406.06911"> [paper]</a>
        <a href="https://github.com/czg1225/AsyncDiff"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Diffusion models have garnered significant interest from the community for their great generative ability across various applications. However, their typical multi-step sequential-denoising nature gives rise to high cumulative latency, thereby precluding the possibilities of parallel computation. To address this, we introduce AsyncDiff, a universal and plug-and-play acceleration scheme that enables model parallelism across multiple devices. Our approach divides the cumbersome noise prediction model into multiple components, assigning each to a different device. To break the dependency chain between these components, it transforms the conventional sequential denoising into an asynchronous process by exploiting the high similarity between hidden states in consecutive diffusion steps. Consequently, each component is facilitated to compute in parallel on separate devices. The proposed strategy significantly reduces inference latency while minimally impacting the generative quality. Specifically, for the Stable Diffusion v2.1, AsyncDiff achieves a 2.7x speedup with negligible degradation and a 4.0x speedup with only a slight reduction of 0.38 in CLIP Score, on four NVIDIA A5000 GPUs. Our experiments also demonstrate that AsyncDiff can be readily applied to video diffusion models with encouraging performances. </p>
        </div>
    </div>
    <div><i>Zigeng Chen, <strong>Xinyin Ma</strong>, Gongfan Fang, Zhenxiong Tan, Xinchao Wang. </i></div>
  </li>

  <li>
    <strong> SlimSAM: 0.1% Data Makes Segment Anything Slim. NeurIPS 2024.</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2312.05284"> [paper]</a>
        <a href="https://github.com/czg1225/SlimSAM"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Current approaches for compressing the Segment Anything Model (SAM) yield commendable results, yet necessitate extensive data to train a new network from scratch. Employing conventional pruning techniques can remarkably reduce data requirements but would suffer from a degradation in performance. To address this challenging trade-off, we introduce SlimSAM, a novel data-efficient SAM compression method that achieves superior performance with extremely less training data. The essence of SlimSAM is encapsulated in the alternate slimming framework which effectively enhances knowledge inheritance under severely limited training data availability and exceptional pruning ratio. Diverging from prior techniques, our framework progressively compresses the model by alternately pruning and distilling distinct, decoupled sub-structures. Disturbed Taylor pruning is also proposed to address the misalignment between the pruning objective and training target, thereby boosting the post-distillation after pruning. SlimSAM yields significant performance improvements while demanding over 10 times less training data than any other existing compression methods. Even when compared to the original SAM, SlimSAM achieves approaching performance while reducing parameter counts to merely 1.4% (9.1M), MACs to 0.8% (23G), and requiring only 0.1% (10k) of the SAM training data. </p>
        </div>
    </div>
    <div><i>Zigeng Chen, Gongfan Fang, <strong>Xinyin Ma</strong>, Xinchao Wang. </i></div>
  </li>

  <li>
    <strong> Remix-DiT: Mixing Diffusion Transformers for Multi-Expert Denoising. NeurIPS 2024. </strong>
    <div style="display: inline">
        <a href="https://openreview.net/forum?id=vo5LONGAdo"> [paper]</a>
        <a href=""> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Transformer-based diffusion models have achieved significant advancements across a variety of generative tasks. However, producing high-quality outputs typically necessitates large transformer models, which result in substantial training and inference overhead. In this work, we investigate an alternative approach involving multiple experts for denoising, and introduce RemixDiT, a novel method designed to enhance output quality at a low cost. The goal of RemixDiT is to craft N diffusion experts for different denoising timesteps, yet without the need for expensive training of N independent models. To achieve this, RemixDiT employs K basis models (where K < N) and utilizes learnable mixing coefficients to adaptively craft expert models. This design offers two significant advantages: first, although the total model size is increased, the model produced by the mixing operation shares the same architecture as a plain model, making the overall model as efficient as a standard diffusion transformer. Second, the learnable mixing adaptively allocates model capacity across timesteps, thereby effectively improving generation quality. Experiments conducted on the ImageNet dataset demonstrate that RemixDiT achieves promising results compared to standard diffusion transformers and other multiple-expert methods. </p>
        </div>
    </div>
    <div><i>Gongfan Fang, <strong>Xinyin Ma</strong>,  Xinchao Wang. </i></div> 
  </li>

  <li>
    ``AAAI 2024``
    <strong>Isomorphic Pruning for Vision Models. ECCV 2024. </strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2407.04616"> [paper]</a>
        <a href="https://github.com/VainF/Isomorphic-Pruning"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Structured pruning reduces the computational overhead of deep neural networks by removing redundant sub-structures. However, assessing the relative importance of different sub-structures remains a significant challenge, particularly in advanced vision models featuring novel mechanisms and architectures like self-attention, depth-wise convolutions, or residual connections. These heterogeneous substructures usually exhibit diverged parameter scales, weight distributions, and computational topology, introducing considerable difficulty to importance comparison. To overcome this, we present Isomorphic Pruning, a simple approach that demonstrates effectiveness across a range of network architectures such as Vision Transformers and CNNs, and delivers competitive performance across different model sizes. Isomorphic Pruning originates from an observation that, when evaluated under a pre-defined importance criterion, heterogeneous sub-structures demonstrate significant divergence in their importance distribution, as opposed to isomorphic structures that present similar importance patterns. This inspires us to perform isolated ranking and comparison on different types of sub-structures for more reliable pruning. Our empirical results on ImageNet-1K demonstrate that Isomorphic Pruning surpasses several pruning baselines dedicatedly designed for Transformers or CNNs. For instance, we improve the accuracy of DeiT-Tiny from 74.52% to 77.50% by pruning an off-the-shelf DeiT-Base model. And for ConvNext-Tiny, we enhanced performance from 82.06% to 82.18%, while reducing the number of parameters and memory usage.  </p>
        </div>
    </div>
    <div><i>Gongfan Fang, <strong>Xinyin Ma</strong>, Michael Bi Mi, Xinchao Wang. </i></div>
  </li>

  <li>
    <strong> LiteFocus: Accelerated Diffusion Inference for Long Audio Synthesis. Interspeech 2024.</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2407.10468"> [paper]</a>
        <a href="https://github.com/Yuanshi9815/LiteFocus"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Latent diffusion models have shown promising results in audio generation, making notable advancements over traditional methods. However, their performance, while impressive with short audio clips, faces challenges when extended to longer audio sequences. These challenges are due to model's self-attention mechanism and training predominantly on 10-second clips, which complicates the extension to longer audio without adaptation. In response to these issues, we introduce a novel approach, LiteFocus that enhances the inference of existing audio latent diffusion models in long audio synthesis. Observed the attention pattern in self-attention, we employ a dual sparse form for attention calculation, designated as same-frequency focus and cross-frequency compensation, which curtails the attention computation under same-frequency constraints, while enhancing audio quality through cross-frequency refillment. LiteFocus demonstrates substantial reduction on inference time with diffusion-based TTA model by 1.99x in synthesizing 80-second audio clips while also obtaining improved audio quality. </p>
        </div>
    </div>
    <div><i>Zhenxiong Tan, <strong>Xinyin Ma</strong>, Gongfan Fang, Xinchao Wang. </i></div>
  </li>

  <li>
    <strong> DepGraph: Towards Any Structural Pruning. CVPR 2023. </strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2301.12900"> [paper]</a>
        <a href="https://github.com/VainF/Torch-Pruning"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Structural pruning enables model acceleration by removing structurally-grouped parameters from neural networks. However, the parameter-grouping patterns vary widely across different models, making architecture-specific pruners, which rely on manually-designed grouping schemes, non-generalizable to new architectures. In this work, we study a highly-challenging yet barely-explored task, any structural pruning, to tackle general structural pruning of arbitrary architecture like CNNs, RNNs, GNNs and Transformers. The most prominent obstacle towards this goal lies in the structural coupling, which not only forces different layers to be pruned simultaneously, but also expects all removed parameters to be consistently unimportant, thereby avoiding structural issues and significant performance degradation after pruning. To address this problem, we propose a general and fully automatic method, Dependency Graph(DepGraph), to explicitly model the dependency between layers and comprehensively group coupled parameters for pruning. In this work, we extensively evaluate our method on several architectures and tasks, including ResNe(X)t, DenseNet, MobileNet and Vision transformer for images, GAT for graph, DGCNN for 3D point cloud, alongside LSTM for language, and demonstrate that, even with a simple norm-based criterion, the proposed method consistently yields gratifying performances. </p>
        </div>
    </div>
    <img src='https://img.shields.io/github/stars/VainF/Torch-Pruning.svg?style=social&label=Star' alt="sym" height="100%">
    <div><i>Gongfan Fang, <strong>Xinyin Ma</strong>, Mingli Song, Michael Bi Mi, Xinchao Wang. </i></div>
  </li>

  <li>
    <strong> Structural Pruning for Diffusion Models. NeurIPS 2023. </strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2305.10924"> [paper]</a>
        <a href="https://github.com/VainF/Diff-Pruning"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across four diverse datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained progenitors. </p>
        </div>
    </div>
    <div><i>Gongfan Fang, <strong>Xinyin Ma</strong>, Xinchao Wang.</i></div>
  </li>

  <li>
   <strong> A Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition. ACL2021. </strong>
    <div style="display: inline">
        <a href="https://aclanthology.org/2021.acl-long.216.pdf"> [paper]</a>
        <a href="https://github.com/tricktreat/locate-and-label"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Named entity recognition (NER) is a well-studied task in natural language processing. Traditional NER research only deals with flat entities and ignores nested entities. The span-based methods treat entity recognition as a span classification task. Although these methods have the innate ability to handle nested NER, they suffer from high computational cost, ignorance of boundary information, under-utilization of the spans that partially match with entities, and difficulties in long entity recognition. To tackle these issues, we propose a two-stage entity identifier. First we generate span proposals by filtering and boundary regression on the seed spans to locate the entities, and then label the boundary-adjusted span proposals with the corresponding categories. Our method effectively utilizes the boundary information of entities and partially matched spans during training. Through boundary regression, entities of any length can be covered theoretically, which improves the ability to recognize long entities. In addition, many low-quality seed spans are filtered out in the first stage, which reduces the time complexity of inference. Experiments on nested NER datasets demonstrate that our proposed method outperforms previous state-of-the-art models. </p>
        </div>
    </div>
    <div><i>Yongliang Shen, <strong>Xinyin Ma</strong>, Zeqi Tan, Shuai Zhang, Wen Wang, Weiming Lu.</i></div>
  </li>

  <li>
   <strong> A Trigger-Sense Memory Flow Framework for Joint Entity and Relation Extraction. WWW 2021</strong>. 
    <div style="display: inline">
        <a href="https://dl.acm.org/doi/abs/10.1145/3442381.3449895"> [paper]</a>
        <a href="https://github.com/tricktreat/trimf"> [code]</a>
        <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" >[abstract]</a>
        <div class="abstract"  style="overflow: hidden; display: none;">  
            <p> Joint entity and relation extraction framework constructs a unified model to perform entity recognition and relation extraction simultaneously, which can exploit the dependency between the two tasks to mitigate the error propagation problem suffered by the pipeline model. Current efforts on joint entity and relation extraction focus on enhancing the interaction between entity recognition and relation extraction through parameter sharing, joint decoding, or other ad-hoc tricks (e.g., modeled as a semi-Markov decision process, cast as a multi-round reading comprehension task). However, there are still two issues on the table. First, the interaction utilized by most methods is still weak and uni-directional, which is unable to model the mutual dependency between the two tasks. Second, relation triggers are ignored by most methods, which can help explain why humans would extract a relation in the sentence. They’re essential for relation extraction but overlooked. To this end, we present a Trigger-Sense Memory Flow Framework (TriMF) for joint entity and relation extraction. We build a memory module to remember category representations learned in entity recognition and relation extraction tasks. And based on it, we design a multi-level memory flow attention mechanism to enhance the bi-directional interaction between entity recognition and relation extraction. Moreover, without any human annotations, our model can enhance relation trigger information in a sentence through a trigger sensor module, which improves the model performance and makes model predictions with better interpretation. Experiment results show that our proposed framework achieves state-of-the-art results by improves the relation F1 to 52.44% (+3.2%) on SciERC, 66.49% (+4.9%) on ACE05, 72.35% (+0.6%) on CoNLL04 and 80.66% (+2.3%) on ADE. </p>
        </div>
    </div>
    <div><i>Yongliang Shen, <strong>Xinyin Ma</strong>, Yechun Tang, Weiming Lu.</i></div>
  </li>  
</ul>





# 🎖 Honors and Awards 
- *2025.06*: 清源潜力学者 by CAAI
- *2025.01*: KAUST AI Rising Stars
- *2024.11*: Google PhD Fellowship
- *2024.10*: NeurIPS'24 Ourstanding Reviewer
- *2019-2022(M.Eng.)*: Outstanding Graduate(2022), Tencent Scholarship(2021), Award of Honor for Graduate(2021, 2020)
- *2015-2019(B.Eng.)*: Outstanding Engineer Scholarship (2018), Outstanding Student of Zhejiang University (2018, 2017, 2016), Second-Class Academic Scholarship of Zhejiang University (2017, 2016)

# 🎩 Educations
- *2022.08 - (now)*, Ph.D. Student in College of Design and Engineering, National University of Singapore
- *2019.08 - 2022.04*, M.Eng. in Computer Science, College of Computer Science and Technology, Zhejiang University
- *2015.09 - 2019.06*, B.Eng. in Software Engineering, Chu Kochen Honors College, Zhejiang University

# 📋 Academic Service
- Workshop: Co-organizor of 2nd workshop on Efficient Large Vision Models, CVPR'25
- Conference: NeurIPS (25, 24, 23), EMNLP (25, 24, 23, 22, 21), ICML (25, 24, 23), ACL (25, 24, 23, 22, 21), ICCV (25), CVPR (25), ICLR (25, 24), AAAI (25, 24), ICASSP (25), ECCV (24), IJCAI (24), NAACL (24)
- Journal: TPAMI, JVCI, TIP, TMLR

# ☃️ Internships
- *2020.12 - 2021.6*, Alibaba DAMO Academy, Research Intern. Mentor: [Yong Jiang](https://jiangyong.site).
- *2018.07 - 2018.11*, Netease Thunderfire UX, Data Analyst Intern. Mentor: Lei Xia.

# 🎙️ Invited Talk 
- May 29, 2025: IVUL @ KAUST. Topic: Efficient Generative Models via Caching
- June 18, 2025: Multimodal Interation Group @ Bytedance Seed. Topic: Efficient and Hybrid Reasoning Models 

# 🍞 Teaching Experience
- Fall 2024, Fall 2023, Spring 2023. TA for EE2211, Introduction to Machine Learning, NUS.
