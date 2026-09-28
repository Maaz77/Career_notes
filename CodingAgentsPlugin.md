# The skills or the plugins I add to my coding agent workflow recommended by the community
You need to read the skills or commands of each to see how you can use them in your ai workflow


## [Ponytail](https://github.com/dietrichgebert/ponytail)
- For stopping the agent from over engineering. 

### Commands

| Command | What it does |
|---------|--------------|
| `/ponytail [lite \| full \| ultra \| off]` | Set the intensity, or turn it off. No argument reports the current level. |
| `/ponytail-review` | Review the current diff for over-engineering, hands back a delete-list. |
| `/ponytail-audit` | Audit the whole repo for over-engineering, not just the diff. |
| `/ponytail-debt` | Harvest the `ponytail:` shortcuts you've deferred into a ledger, so "later" doesn't become "never". |
| `/ponytail-gain` | Show the measured impact scoreboard (less code, less cost, more speed) from the benchmark. |
| `/ponytail-help` | Quick reference for the commands above. |


## Reduce Token Usage
- [Caveman](https://github.com/juliusbrussee/caveman)
- [rtk](https://github.com/rtk-ai/rtk)
- [headroom](https://github.com/headroomlabs-ai/headroom)

## [Engineering skills](https://github.com/mattpocock/skills)

## [Diagram Design](https://github.com/cathrynlavery/diagram-design)

## [Claude skills market place](https://claudemarketplaces.com)

## [Graphify](https://github.com/Graphify-Labs/graphify)

## Full stack skills and plugins
- [Full stack skills](https://github.com/ancoleman/ai-design-components)
- [Full stack plugins](https://github.com/wshobson/agents)
    - If you're working with Claude Code, this repository wshobson/agents is a real gem you absolutely shouldn't miss. It basically brings an army of smart agents right into your terminal!

    What's the deal? This project is a full "marketplace" for Claude Code, packed with 63 plugins and 85 specialized agents (Agents). What's its goal? Smart automation. Meaning you've got a specialist agent at your fingertips for every task, from system architecture and coding to testing, security, and even SEO.

    What makes it super appealing is its optimized architecture. The system uses "Skills" (skills) that work on Progressive Disclosure. What does that mean? It means specialized knowledge only loads when you actually need it. That way, you don't waste tokens unnecessarily, and the model's context doesn't get bloated.

    Another smart move: Hybrid Orchestration. This tool automatically uses the Haiku model for quick, linear tasks and switches to Sonnet for complex ones that need reasoning. The result? High speed and lower costs.

    A few examples of what it can do:
    - Full-stack: With one command, 7 agents (from database to front-end) coordinate to build a complete feature.
    - Security: The Security Scanning plugin tears through your code and finds bugs.
    - DevOps plugin: It has ready-made agents for Kubernetes and cloud too.

    Installing it in Claude Code is super easy. First, you add the marketplace: /plugin marketplace add wshobson/agents

    Then you install whichever plugin you need separately. For example, for Python: /plugin install python-development That's it! Now the Python agents are ready.

    In short, if you want to multiply the power of Claude Code and feel like a tech lead with a bunch of badass programmers under you, definitely give it a test run.


## [Claude Code Skills Factory](https://github.com/alirezarezvani/claude-code-skill-factory)

- Genratte skills, slash commands, hooks, ...

## [Some references & tutorials](https://github.com/shanraisshan/claude-code-best-practice)

## [MCP discovery](https://mcp.so/)

## [Claude code awesome tutorials](https://github.com/hesreallyhim/awesome-claude-code)

## [Persists Session](https://github.com/thedotmack/claude-mem)

## [Water Mark Remover](https://github.com/guillaumemeyer/watermarks-remover)

