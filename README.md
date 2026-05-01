# Deep Multi-Task Learning in Connected Autonomous Vehicles — Short Story Assignment

> **Course:** NLP / Foundation Models  
> **Paper:** [A Survey on Deep Multi-Task Learning in Connected Autonomous Vehicles](https://arxiv.org/abs/2508.00917)  
> **Authors:** Jiayuan Wang, Farhad Pourpanah, Q. M. Jonathan Wu, Ning Zhang (University of Windsor & Queen's University, 2025)

---

## Overview

This repository contains all deliverables for the short story assignment based on the survey paper above. The paper is the **first comprehensive review** focused on Multi-Task Learning (MTL) within the context of Connected Autonomous Vehicles (CAVs). It covers how a single unified deep learning model can simultaneously handle perception, prediction, planning, control, and multi-agent collaboration — making self-driving systems faster, cheaper, and more reliable.

---

## Repository Structure

```
├── README.md                    ← You are here
├── medium_article.md            ← Full Medium article content (published separately)
├── slides/
│   └── short_story_slides.pptx  ← Presentation slide deck
├── code/
│   └── mtl_reproduction.py      ← Reproduction experiment (autoresearch template)
│   └── requirements.txt         ← Python dependencies
├── video_script.md              ← Script for the YouTube video walkthrough
└── assets/
    └── (figures referenced in article)
```

---

## Key Links

| Deliverable | Link |
|---|---|
| 📄 Medium Article | *(paste your Medium URL here after publishing)* |
| 🎥 YouTube Video | *(paste your YouTube URL here after uploading)* |
| 📖 Original Paper | https://arxiv.org/abs/2508.00917 |

---

## What is This Paper About?

Self-driving cars must do many things at once — detect pedestrians, estimate depth, predict where a car is going, plan a route, and control the steering — all in real time. Traditionally, each of these tasks had its own separate deep learning model, which is expensive and slow.

**Multi-Task Learning (MTL)** solves this by training **one model to do all tasks simultaneously**, sharing knowledge across them. When combined with **Vehicle-to-Everything (V2X) communication** (cars talking to other cars, traffic lights, and cloud servers), you get Connected Autonomous Vehicles (CAVs) that are significantly smarter and more efficient.

This survey is the first to comprehensively map how MTL is being applied across every layer of the CAV software stack.

---

## Paper Structure Summary

| Section | Topic |
|---|---|
| Section I | Introduction — Why MTL for CAVs? |
| Section II | CAV Systems — Hardware, Software, V2X |
| Section III | MTL Overview — Architectures & Optimization |
| Section IV | MTL in CAVs — Perception, Prediction, Planning, Control |
| Section V | Multi-Agent Collaboration via V2X |
| Section VI | Open Challenges & Future Directions |

---

## Reproduction Experiment

The code in `/code/mtl_reproduction.py` reproduces a simplified version of a **multi-task perception model** inspired by the paper, using the [autoresearch template](https://dlmastery.github.io/autoresearch/). It trains a shared-encoder model on the **BDD100K** dataset to simultaneously perform:
- Object detection
- Drivable area segmentation
- Lane detection

See `/code/` for instructions on running the experiment.

---

## References

- Wang et al. (2025). *A Survey on Deep Multi-Task Learning in Connected Autonomous Vehicles*. arXiv:2508.00917
- Vandenhende et al. (2021). *Multi-Task Learning for Dense Prediction Tasks: A Survey*. TPAMI.
- Crawshaw (2020). *Multi-Task Learning with Deep Neural Networks: A Survey*. arXiv:2009.09796.
- BDD100K Dataset: https://www.bdd100k.com/
- AutoResearch Template: https://dlmastery.github.io/autoresearch/
