# AI-PM: AI-Powered Project Management & Optimization System

**AI-PM** is a sophisticated project management assistant that combines **Combinatorial Optimization (CP-SAT)**, **Large Language Models (Groq)**, and **Interactive Human-in-the-Loop Reviews** to solve complex resource allocation and scheduling problems. It is designed to handle real-world constraints like skill matching, capacity planning, and deadline adherence, while learning from user preferences over time via coactive learning.

---

## 📂 Project Architecture

The repository is structured to separate the frontend (Streamlit), core logic (Optimization, LLM, Skills), and configuration/data persistence.

### `ai_pm/` (Root Package)

#### `app_streamlit/` — Frontend Interface
Interactive web application built with Streamlit.
*   **`00_home.py`**: Landing page. Manages navigation and displays quickstart commands if the sidebar links fail.
*   **`01_ingest.py`**: **Data Ingestion & Normalization**.
    *   Accepts `team.csv` and `skills.csv` (plus optional history/holidays).
    *   **Skill Normalization (Groq)**: Identifies unknown skills in the team data and uses an LLM to map them to a canonical ontology (e.g., "ReactJS" → "React").
    *   Validates schemas and snapshots inputs to `runs/<Project>/<Timestamp>`.
*   **`02_task_builder.py`**: **AI Task Generation**.
    *   Takes a raw text brief (paste or upload).
    *   Uses **Groq (LLM)** to extracting a structured `task_graph.json`, inferring dependencies and required skills.
    *   Provides an editable grid for human refinement before saving.
*   **`03_optimizer.py`**: **Plan Generation (Plan A)**.
    *   Configures optimization weights ($\theta$) for skill fit, fairness, continuity, and risk.
    *   Runs the **CP-SAT Solver** to generate an initial assignments plan.
    *   Visualizes results: Gantt chart, workload distribution, and KPIs.
*   **`04_review.py`**: **Interactive Review & Plan B**.
    *   **Side-by-side comparison** of Plan A vs. Plan B.
    *   **Quick Fix**: One-click resolution for tasks violating deadlines (adds `end_before` constraints).
    *   **Locks**: Allows users to lock specific tasks to members or set strict date windows.
    *   **Coactive Learning**: "Prefer B over A" button updates the global preference weights ($\theta$) based on the differences between the plans.
*   **`05_analytics.py`**: **Deep Dive Analytics**.
    *   Utilization histograms (capacity vs. load).
    *   Deadline risk heatmaps (slack analysis).
    *   Critical path visualization.
*   **`06_admin.py`**: **System Administration**.
    *   View/Reset preference weights.
    *   Export full run packages (ZIP/PDF) for external reporting.
*   **`_state.py`**: Shared state management, environment loading, and path resolution.

#### `core/` — Backend Logic
*   **`optimizer.py`**: The mathematical engine.
    *   **`solve_assignments()`**: Uses **Google OR-Tools (CP-SAT)** to assign tasks. Enforces hard constraints (1 owner/task, skill requirements, capacity limits) and optimizes for soft objectives (skill fit, fairness, continuity).
    *   **`build_schedule()`**: A deterministic greedy scheduler that places tasks on a timeline, respecting dependencies, workweeks (Mon-Fri), and work hours.
*   **`skills.py`**: Skills ontology manager. Handles normalization (lowercase, strip, mapping synonyms) and persistence.
*   **`skills_groq.py`**: Interface for LLM-based skill suggestion.
*   **`llm_extractor.py`**: Logic for parsing text briefs into JSON task graphs.
*   **`kpis.py`**: Analytics library. Calculates:
    *   **Coverage**: % of tasks scheduled.
    *   **Skill Fit**: How well assignments match member capabilities (0-1).
    *   **Utilization StdDev**: Metric for workload fairness (lower is better).
    *   **Critical Path**: Longest dependency chain duration.
    *   **Slack**: Time buffer before a task's deadline.
*   **`schemas.py`**: Pydantic models defining strict data contracts for CSV inputs (Team, Skills, Holidays, History).
*   **`explain.py`**: Rationale generation engine. Produces human-readable explanations for *why* a specific assignment was made (e.g., "Member A selected for high skill fit despite higher workload").
*   **`storage.py`**: Utilities for file I/O, directory management, and input snapshotting.

#### `config/` & `runs/`
*   **`config/settings.yaml`**: Global settings (Timezone `Asia/Kolkata`, Workday 9-5, etc.).
*   **`runs/`**: The "Database". Every execution creates a timestamped folder containing inputs, logs, the generated plan (`plan.json`), and KPIs.

---

## 📊 Data Schemas

The system relies on structured CSV inputs. Key columns include:

*   **`team.csv`**:
    *   `member_id` (str): Unique identifier.
    *   `weekly_capacity_hours` (float): Total hours available per week.
    *   `skills` (JSON): `[{"name": "python", "level": 4}, ...]` (Levels 1-5).
*   **`skills.csv`** (Ontology):
    *   `canonical_skill`: The standard name (e.g., "Python").
    *   `synonyms`: Pipe-delimited or JSON list (e.g., "py|python3").
*   **`history.csv`** (Optional):
    *   `task_id`, `member_id`, `review_score` (1-5): Used to calculate continuity bonuses.

---

## 🚀 Key Features & Methodologies

### 1. Constraint Programming (CP-SAT)
Unlike simple heuristics, AI-PM uses a solver to find the mathematically optimal assignment.
*   **Hard Constraints**:
    *   Every task must have exactly one owner.
    *   Assigned owner *must* possess all required skills at the minimum level.
    *   Total assigned hours $\le$ Member Capacity * Horizon Weeks.
*   **Soft Objectives** (Weighted by $\theta$):
    *   Maximize **Skill Fit**: Reward high-level skills on relevant tasks.
    *   Maximize **Continuity**: Reward assigning tasks to members with good history.
    *   Minimize **Inequity**: Penalize deviation from average workload (Fairness).

### 2. Coactive Learning (Human-in-the-Loop)
The system learns from your corrections.
1.  **Generate Plan A** using current weights ($\theta$).
2.  **User Review**: You notice Plan A overloads a senior dev. You lock a task to a junior dev and re-solve (**Plan B**).
3.  **Feedback**: You click "Prefer B over A".
4.  **Update**: The system calculates the difference in feature vectors (Fairness_B - Fairness_A) and nudges $\theta$ to value Fairness more in future runs.

### 3. LLM Integration (Groq)
*   **Ingestion**: Normalizes messy human data ("ReactJS", "react.js") into a clean ontology.
*   **Extraction**: Converts unstructured briefs ("Build a login page using OAuth") into structured dependency graphs (`task_graph.json`).

---

## 🛠️ Setup Guide

### Prerequisites
*   Python 3.10+
*   [Groq API Key](https://console.groq.com/) (for LLM features)

### Installation
1.  **Clone & Install**:
    ```bash
    git clone https://github.com/your-repo/ai-pm.git
    cd ai-pm
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    *   Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
    *   Edit `ai_pm/config/settings.yaml` to set your timezone and work hours.

3.  **Run the App**:
    ```bash
    streamlit run app_streamlit/00_home.py
    ```

---

## 📈 Usage Example

1.  **Ingest**: Upload `team.csv` and `skills.csv`. Click "Validate & Snapshot".
2.  **Build**: Paste a brief: *"Create a CRUD API for products using FastAPI."* Click "Generate".
3.  **Optimize**: Go to **03 - Optimizer**. Select inputs. Click "Generate Plan".
4.  **Review**: See that the API task is late. Go to **04 - Review**.
    *   The "Quick Fix" table warns: "Task T5 finishes Oct 25, due Oct 23".
    *   Click "Apply Quick Fix" (adds `end_before=2025-10-23`).
    *   Click "Re-solve". The solver re-shuffles resources to meet the deadline.
    *   Click "Prefer B over A" to teach the system to prioritize deadlines higher next time.
