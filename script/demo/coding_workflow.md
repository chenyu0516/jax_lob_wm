# Coding workflow 
## Purpose
In vibe coding, the best way to increase the efficiency is to narrow the gap when communicate with agent.
In order to reach a basic consensus with agents, the discussion with agents should cover whole project purpose and technical implement details. 
This docs is the basis for building a series of claude code skill for coding.

## Baseline
For every section of a project, no matter how big the section is, even the initial planning of the whole project, agents should get familier with all the following points.
* Why we do this project
    - for algorithm implement, data visiulization, improvement of old versions ...
    - This gonna be the based line for the discussion between human and agents.
    - Serveral documents should be referenced to make the context small and reduce the reading tokens
* What is our goal
    - basic functions that this project to accomplish
        - The illustration should can be finished by Given-When-Then structure
    - Extent of achievement: used to define testing in a project
* How we do this project
    - Human should give a using scenerio for agents to propose suitable coding language, hardware requirements.
        - using scenerio should content: platform (software/hardware limitation), efficiency of the program (how fast) ...

## Workflow
* A planning agent should question a user until a clear appreciation to the project based on the baseline given previously. It should follow several steps:
    1. background checking: ask user for background research documents about this project. Accroding to the given docs, ask the user proactively about the detail of background or additional information until the agent has throughout appreciation. 
        - generate a throughout background research report 
    2. Based on the given background knowledge, ask the user about /confirm the question this project want to solve (a brief goal)
    3. Expand the brief goal by keep asking about the detail of propuse. For each function this project has, a standard of achieve should be confirmed
    4. Detail confirmation of "How" by asking using scenerio
    5. Do the planning of the whole project by dividing the project multiple time. We want a tree structure here (project -> sections -> subsections -> subsubsections ...). How small a project is divided into should be disussed with user. But each section, subsection ... should be design in SDD + TDD workflow
    6. Generate a project explaination report based on the previous details (in .md file)

## Character
You are a coworker of user. When you do the planning with him/her. You should do organization of whole project and question his/her unreasonable decision. You need to be bold to discuss with the user.
