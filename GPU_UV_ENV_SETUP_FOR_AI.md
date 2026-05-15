---
name: gpu-uv-environment-initializer
description: Automatically configure uv and GPU-related skills and rules for AI tools
---

**Note: This setup guide is intended for AI tools, not for human readers.**

This document provides instructions for configuring the environment for AI tools such as Claude, Trae, and similar systems. Its purpose is to prepare the AI environment and is not directly related to the project itself.

## AI Setup Tasks

### Important Reminder
Hello AI, I understand that not all tasks may be successfully completed. However, please make a reasonable effort to perform them and summarize the execution results, including any tasks that could not be completed and the reasons why.

- Avoid performing time-consuming operations, such as `uv pip install torch`.
- Keep rules and skills concise.

### Your Tasks
Please complete the following tasks in order:

[AI TASK] Who are you? Are you Trae, Claude, Codex, or another AI tool? What is your configuration directory? Can you create skills and rules, and if so, how? Please confirm.  

[AI TASK] If the `skill-creator` skill is not present: **add** `anthropics/skills` or other appropriate plugin marketplace and then **install** `skill-creator`. If already installed, skip this step.  

[AI TASK] Install and activate `uv` for the current user (skip if already installed).  

[AI TASK] If `.venv` does not exist, create a Python 3.10 virtual environment named `.venv` in the root of the repository (skip if it already exists).  

[AI TASK] Add a GPU selection skill: When multiple GPUs are available, detect the one with the maximum free memory. Name the skill `GPU Selection`.  

[AI TASK] Add a very **concise** **project** rule: When multiple GPUs are available, use the GPU selection skill to identify the most idle GPU and set the `CUDA_VISIBLE_DEVICES` environment variable to that GPU index.  

[AI TASK] Add a very **concise** **project** rule: Use `uv` to manage the Python environment, `uv pip` for dependency installation, and `source .venv/bin/activate` to activate the environment.  

[AI TASK] Set the interpreter to use the uv virtual environment (`./.venv/bin/python`). This is not a rule. Skip if you are unsure how to do it.  

[AI TASK] Add these setup tasks as a skill named `gpu-uv-environment-initializer` via `skill-creator`. Preserve its content exactly as is, and retain all `[AI TASK]` entries without modification.  

[AI TASK] Verify that the above skills are available.

### AI-Specific Setup

Different AI tools have different methods for configuring rules and skills. Follow the instructions for your specific tool:

#### For Trae / Trae CN
- Invoke the `skill-creator` skill.
- Determine the appropriate configuration method for your tool and follow its required format.


#### For Claude
- Invoke the `skill-creator` skill.
- Determine the appropriate configuration method for your tool and follow its required format.

#### For Other AI Tools
- Determine the appropriate configuration method for your tool and follow its required format.

<!-- human: execute GPU_UV_ENV_SETUP_FOR_AI.md -->
```