# Autonomous GenAI Drone Reconnaissance Framework

This repository contains a modular, three-layer architecture for an autonomous drone reconnaissance system, integrating **Generative AI (LLM)** for high-level reasoning and **RAG (Retrieval-Augmented Generation)** for tactical knowledge grounding.

## 🚀 Overview

The framework is designed to move beyond simple automation by giving drones "cognitive" capabilities:
- **Spatial Reasoning**: Interprets complex satellite imagery and maintains a 2.5D awareness of buildings and obstacles.
- **Dynamic Tactics**: Generates Behavior Trees and Kinematic profiles on-the-fly based on RAG knowledge.
- **Closed-Loop Learning**: Uses Episodic Memory to learn from past mission failures and avoid repeating mistakes.

## 📂 Repository Structure

- **[Layer 1: Cognitive Brain](./layer1_test)**  
  The "Decision Layer." Handles natural language processing, RAG retrieval from military SOPs, and episodic memory management. Contains the knowledge databases (`db/`).
- **[Layer 2: Behavior Planner](./layer2_test)**  
  The "Tactical Layer." Translates high-level mission goals into waypoint sequences, manages the FSM (Finite State Machine), and maintains 3D-aware geofencing.
- **[Layer 3: Physical Controller](./layer3_test)**  
  The "Execution Layer." Low-latency control for waypoint following using PID and obstacle avoidance via Artificial Potential Fields (APF).
- **[Memory DB](./memory_db)**  
  Contains the episodic memory store for closed-loop learning across mission cycles.

## 🧠 Key Technologies

- **LLM Context Injection**: Dynamic prompt building using ChromaDB + HNSW.
- **Multi-Modality**: Support for local LLMs (Ollama) and cloud-based models (GPT-4).
- **Pseudo-3D Perception**: Height estimation from 2D imagery for autonomous "Auto-Climb" decisions.

## 🛠️ Getting Started

1. **Prerequisites**: Python 3.10+, `numpy`, `opencv-python`, `sentence-transformers`, `chromadb`.
2. **Environment**: Install dependencies via `pip install -r requirements.txt`.
3. **Simulation**: Run `python phase4_continuous_recon.py` in the root (ensure all layers are in Path) to see a full search-and-detect loop.

---
*Developed as part of an advanced exploration in agentic drone controls and cognitive simulation.*
