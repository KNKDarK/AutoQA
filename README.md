```markdown
<p align="center">
  <h1 align="center">👁️ AutoQA Agent</h1>
  <p align="center">
    <strong>Autonomous Multimodal Web QA, Figma Visual Auditing & Automated Bug Healing Engine</strong>
  </p>
  <p align="center">
    <a href="#-features"><img src="https://img.shields.io/badge/Status-Beta-blue" alt="Status"></a>
    <a href="#-tech-stack"><img src="https://img.shields.io/badge/Python-3.11%2B-brightgreen" alt="Python Version"></a>
    <a href="#-license"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
    <a href="https://github.com/prs-welcome"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome"></a>
  </p>
</p>

---

## 📌 Problem & Solution

Traditional E2E test suites (Cypress, Playwright, Selenium) break whenever UI elements, classes, or DOM structures change. Meanwhile, visual regressions and design-to-code drift require tedious manual review by design and QA teams.

**AutoQA Agent** automates this end-to-end lifecycle:
1. Navigates web apps like a human tester using Multimodal Vision-Language Models (VLMs).
2. Audits live browser renders directly against Figma design specifications down to bounding boxes and style tokens.
3. Captures JS console errors and functional failures, generates Playwright scripts, and opens GitHub Pull Requests with proposed code patches created inside isolated sandbox environments.

---

## ⚡ Key Features

| Feature | Description |
| :--- | :--- |
| **🎨 Figma-to-DOM Audit** | Fetches Figma frames via REST API and compares them against high-DPI headless browser snapshots to catch pixel drift, padding issues, and brand inconsistencies. |
| **🧭 Vision-Guided Browser Agent** | Uses VLM vision (Claude 3.5 Sonnet / GPT-4o) combined with a flattened DOM tree to interact dynamically with complex UI components without pre-defined XPaths or selectors. |
| **🩹 Self-Healing Playwright Scripts** | Records successful user journeys and synthesizes production-grade Playwright tests (TypeScript/Python). Automatically updates broken selectors when DOM structures shift. |
| **🛠️ Sandboxed Auto-Fixing Engine** | Synthesizes source code fixes inside E2B isolated Docker sandboxes when failures are detected, verifies the fix, and auto-submits GitHub Pull Requests with repro steps. |
| **📊 Comprehensive Bug Reports** | Outputs markdown, HTML, and JSON reports featuring annotated screenshot callouts, video replays, and step-by-step action trajectories. |

---

## 🏗️ Architecture & Execution Flow


```

```
                     ┌────────────────────────┐
                     │   Figma Canvas / Frame │
                     └───────────┬────────────┘
                                 │ (Design Tokens / Frames)
                                 ▼

```

┌──────────────────┐    ┌─────────────────────────┐    ┌──────────────────────────┐
│ CLI / Web App    ├───►│ AutoQA Core Controller  ├─┬─►│ Vision-LLM Reasoner      │
│ Target URL + Goal│    └────────────┬────────────┘ │  │ (Claude 3.5 / GPT-4o)    │
└──────────────────┘                 │              │  └──────────────────────────┘
│ Actions      │
▼              │ Annotations & Decisions
┌─────────────────────────┐ │
│  Playwright Execution   ├─┘
│    (Chromium / Firefox) │
└────────────┬────────────┘
│ (On Visual / DOM Error)
▼
┌─────────────────────────┐
│    E2B Code Sandbox     ├────────► 🐙 GitHub API
│ (Source Code Patching)  │          (Opens Pull Request)
└─────────────────────────┘

```

---

## 🗂️ Repository Structure

```text
autoqa-agent/
├── .github/
│   └── workflows/
│       └── autoqa-ci.yml        # CI trigger for automated PR visual audits
├── config/
│   └── default.yaml             # Thresholds, viewport options, and model flags
├── src/
│   ├── agent/
│   │   ├── core.py              # Main autonomous agent control loop
│   │   ├── vision.py            # Image processing & layout comparison engine
│   │   └── prompts.py           # VLM visual evaluation & action system prompts
│   ├── browser/
│   │   ├── driver.py            # Playwright browser instance wrapper
│   │   └── dom_parser.py        # DOM tree simplification and element tagger
│   ├── integrations/
│   │   ├── figma.py             # Figma REST API integration & frame exporter
│   │   └── github_bot.py        # GitHub issue and PR automated publisher
│   ├── sandbox/
│   │   └── repair_env.py        # E2B runner for verifying auto-patches
│   └── main.py                  # CLI entrypoint
├── tests/
│   ├── unit/                    # Unit tests for parsers and utilities
│   └── integration/             # Test scenarios against dummy applications
├── .env.example                 # Environment variables template
├── pyproject.toml               # Poetry / Pip build dependencies configuration
├── requirements.txt             # Locked dependencies list
└── README.md

```

---

## ⚙️ Installation & Setup

### Prerequisites

* **Python 3.11+**
* **Node.js 18+** (Required for Playwright browser engines)
* **Git**

### 1. Clone & Set Up Environment

```bash
# Clone repository
git clone https://github.com/knkDarK/autoqa-agent.git
cd autoqa

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
playwright install --with-deps chromium firefox

```

### 2. Configure Environment Variables

Create a `.env` file in the root folder based on `.env.example`:

```bash
cp .env.example .env

```

Populate the `.env` file with your credentials:

```env
# LLM Provider Keys
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-proj-...

# Integration Credentials
FIGMA_ACCESS_TOKEN=figd_...
GITHUB_TOKEN=ghp_...
E2B_API_KEY=e2b_...

# Browser Settings
HEADLESS=true
VIEWPORT_WIDTH=1440
VIEWPORT_HEIGHT=900

```

---

## 🚦 Usage & Examples

### 1. Run Autonomous Exploratory QA Test

Pass a starting URL and a plain-text user goal to let the agent navigate and test your application:

```bash
python -m src.main test \
  --url "[https://app.example.com/login](https://app.example.com/login)" \
  --goal "Sign in with test credentials, add 'Pro Membership' to cart, and reach payment screen" \
  --output ./reports/run-01

```

### 2. Perform Figma Visual Comparison Audit

Compare a live web screen against a Figma design node:

```bash
python -m src.main audit \
  --url "[https://app.example.com/checkout](https://app.example.com/checkout)" \
  --figma-file-id "aB1c2D3e4F5g6H" \
  --figma-node-id "102:45" \
  --threshold 0.05

```

### 3. Generate Self-Healing Playwright Script

Generate a reusable Playwright Python or TypeScript test script based on the agent's successful run:

```bash
python -m src.main export-test \
  --trajectory ./reports/run-01/trajectory.json \
  --lang typescript \
  --out ./e2e/checkout.spec.ts

```

---

## 🖥️ Sample Terminal Output

```text
[AutoQA Core] 🚀 Initializing session for [https://app.example.com](https://app.example.com)
[Browser Context] 🌐 Chromium instance launched (1440x900).
[VLM Reasoner] 👁️ Analyzing DOM state & screenshot...
  ├── Action 1: CLICK element [data-testid="login-btn"]
  ├── Action 2: FILL element [name="email"] -> "qa-bot@example.com"
  └── Action 3: CLICK element [type="submit"]
[Visual Audit] 🔍 Running pixel comparison against Figma Node 102:45...
  ⚠️ Visual Regression Detected:
  └── Element '.submit-btn' background-color differs (Found: #FF0000, Expected: #0066FF)
[Auto-Fix Engine] 🛠️ Launching E2B sandbox for target repo: company/frontend...
  ├── Synthesizing CSS patch...
  ├── Executing local test verification... PASS
  └── 📤 GitHub Pull Request #142 created: "Fix: align submit button color with Figma tokens"
[AutoQA Core] ✅ Mission complete. Report saved to ./reports/run-01/

```

---

## 🗺️ Roadmap

* [x] Multimodal browser control with `browser-use` & Playwright.
* [x] Figma API frame retrieval & visual difference extraction.
* [x] E2B Code sandbox integration for autonomous code patching.
* [ ] Multi-viewport concurrent testing (Mobile, Tablet, Desktop).
* [ ] CI/CD GitHub Action plugin for automated PR visual reviews.
* [ ] Voice and audio assertion testing for WebRTC applications.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please refer to [`CONTRIBUTING.md`](https://www.google.com/search?q=CONTRIBUTING.md) for details on code style and testing guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](https://www.google.com/search?q=LICENSE) file for details.

```

---

<ElicitationsGroup>
Would you like to build out the core Python codebase starting with `src/browser/driver.py` and `src/agent/core.py`, or set up the project's Docker environment first?
</ElicitationsGroup>

<FollowUp>
Would you like me to write the `config/default.yaml` and `pyproject.toml` files to make this repository fully runnable?
</FollowUp>

```