---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<span class='anchor' id='about-me'></span>
Hi there!

I am Xuanlei Zhao, a second-year PhD student in Computer Science at National University of Singapore advised by [Yang You](https://www.comp.nus.edu.sg/~youy/), where I also completed my master's studies. I obtained my bachelor's degree in CS & EE from Huazhong University of Science and Technology. I currently intern at [Adobe Research](https://www.adobe.com/) with [Yan Kang](https://research.adobe.com/person/yan-kang/). Previously, I collaborated at [Pika](https://pika.art/about) with [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and interned at [Colossal-AI](https://github.com/hpcaitech/ColossalAI) with [Jiarui Fang](https://fangjiarui.github.io/).

My current research mainly focuses on efficient AI, including:

- Efficient diffusion and autoregressive models, e.g., for video generation.
- Efficient machine learning system, with parallelism and low-level optimization.
- Co-optimization of algorithm and infrastructure.


# 🔥 News
- *2025.07*: Join [Adobe Research](https://www.adobe.com/) as research intern in Seattle.
- *2025.05*: [DSP](https://arxiv.org/abs/2403.10266) accepted by ICML 2025!
- *2025.01*: [PAB](https://arxiv.org/abs/2408.12588) accepted by ICLR 2025 and integrated into [Diffusers](https://huggingface.co/docs/diffusers/en/api/cache#diffusers.PyramidAttentionBroadcastConfig)!
- *2024.03*: Release [VideoSys (OpenDiT)](https://github.com/NUS-HPC-AI-Lab/VideoSys) for efficient training and inference of video models.
- *2024.02*: [HeteGen](https://arxiv.org/abs/2403.01164) accepted by MLSys 2024!
- *2024.01*: [AutoChunk](https://arxiv.org/abs/2401.10652) accepted by ICLR 2024!
- *2024.01*: Start my PhD journey!


<span class='anchor' id='publications'></span>

# 📝 Selected Publications ([all](https://scholar.google.com/citations?user=I5NBOacAAAAJ))
## 📽️ Efficient Video Generation
<ul>
  <li>
    <code class="language-plaintext highlighter-rouge">ICLR 2025</code> <strong>Real-Time Video Generation with Pyramid Attention Broadcast</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2408.12588"> [paper]</a>
        <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys"> [code]</a>
        <a href="https://oahzxl.github.io/PAB/"> [blog]</a>
    </div>
    <img src='https://img.shields.io/github/stars/NUS-HPC-AI-Lab/VideoSys.svg?style=social&label=Star' alt="sym" height="100%">
    <div><i><u>Xuanlei Zhao</u><b><sup>*</sup></b>, Xiaolong Jin<b><sup>*</sup></b>, Kai Wang<b><sup>*</sup></b>, Yang You</i></div>
  </li>
  <li>
    <code class="language-plaintext highlighter-rouge">ICML 2025</code> <strong>DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2403.10266"> [paper]</a>
        <a href="https://github.com/NUS-HPC-AI-Lab/VideoSys"> [code]</a>
    </div>
    <div><i><u>Xuanlei Zhao</u>, Shenggan Cheng, Chang Chen, Zangwei Zheng, Ziming Liu, Zheming Yang, Yang You</i></div>
  </li>
</ul>

<hr>

## 🧹 Efficient Memory Cost
<ul>
  <li>
    <code class="language-plaintext highlighter-rouge">ICLR 2024</code> <strong>AutoChunk: Automated Activation Chunk for Memory-Efficient Long Sequence Inference</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2401.10652"> [paper]</a>
        <a href="https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/autochunk"> [code]</a>
    </div>
    <div><i><u>Xuanlei Zhao</u>, Shenggan Cheng, Guangyang Lu, Jiarui Fang, Haotian Zhou, Bin Jia, Ziming Liu, Yang You</i></div>
  </li>
  <li>
    <code class="language-plaintext highlighter-rouge">MLSys 2024</code> <strong>HeteGen: Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices</strong>
    <div style="display: inline">
        <a href="https://arxiv.org/abs/2403.01164"> [paper]</a>
    </div>
    <div><i><u>Xuanlei Zhao</u><b><sup>*</sup></b>, Bin Jia<b><sup>*</sup></b>, Haotian Zhou<b><sup>*</sup></b>, Ziming Liu, Shenggan Cheng, Yang You</i></div>
  </li>
</ul>

<hr>

## 🔬 Efficient AI for Science
<ul>
  <li>
    <code class="language-plaintext highlighter-rouge">PPoPP 2024</code> <strong>FastFold: Optimizing AlphaFold Training and Inference on GPU Clusters</strong>
    <div style="display: inline">
        <a href="https://dl.acm.org/doi/10.1145/3627535.3638465"> [paper]</a>
        <a href="https://github.com/hpcaitech/FastFold"> [code]</a>
    </div>
    <img src='https://img.shields.io/github/stars/hpcaitech/FastFold.svg?style=social&label=Star' alt="sym" height="100%">
    <div><i>Shenggan Cheng, <u>Xuanlei Zhao</u>, Guangyang Lu, Jiarui Fang, Tian Zheng, Ruidong Wu, Xiwen Zhang, Jian Peng, Yang You</i></div>
  </li>
</ul>


# 💡 Open-Source Projects
- **[VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)** (Project Lead): An Easy and Efficient System for Video Generation <img src='https://img.shields.io/github/stars/NUS-HPC-AI-Lab/VideoSys.svg?style=social&label=Star' alt="sym" height="100%">
- **[Colossal-AI](https://github.com/hpcaitech/ColossalAI)** (Top Contributor): Making large AI models cheaper, faster and more accessible <img src='https://img.shields.io/github/stars/hpcaitech/ColossalAI.svg?style=social&label=Star' alt="sym" height="100%">
- **[FastFold](https://github.com/hpcaitech/FastFold)** (Top Contributor): Optimizing AlphaFold Training and Inference on GPU Clusters <img src='https://img.shields.io/github/stars/hpcaitech/FastFold.svg?style=social&label=Star' alt="sym" height="100%">


# 💻 Internships
- *2025.07 - now*, [Adobe Research](https://www.adobe.com/) with [Yan Kang](https://research.adobe.com/person/yan-kang/).
- *2024 - 2024*, [Pika](https://pika.art/) with [Chenlin Meng](https://cs.stanford.edu/~chenlin/).
- *2022.07 - 2023.12*, [Colossal-AI](https://github.com/hpcaitech/ColossalAI) with [Jiarui Fang](https://fangjiarui.github.io/) and [Shenggui Li](https://franklee.xyz/).


# 📖 Educations
- *2024.01 - now*, PhD in Computer Science, National University of Singapore
- *2022.08 - 2023.12*, Master in Computer Science, National University of Singapore
- *2018.09 - 2022.06*, Bachelor in Computer Science & Electrical Information, Huazhong University of Science and Technology


# 💬 Invited Talks
- *2024.07*, Real-Time Video Generation with Pyramid Attention Broadcast, Ventures [\[video\]](https://www.techbeat.net/talk-info?id=892)
- *2024.07*, Speedup for Video Generation, Bytedance internal talk
