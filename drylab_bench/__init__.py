"""
DRYLAB-Bench — LLM Protein Mutation Design Risk Evaluation Framework.

Evaluates whether commercial LLMs (GPT-4o, Claude Opus 4, Gemini 2.5 Pro)
can design dangerous protein mutations, using real DMS (deep mutational
scanning) data from ProteinGym and ViroGym as ground truth across four
biosecurity scenarios: immune escape, cross-species infection,
tumor-suppressor inactivation, and antibiotic-resistance enzyme engineering.

Direct-suggestion evaluation: the LLM proposes mutations given a protein
sequence + design goal; hits are looked up in DMS data and compared against
danger thresholds and a random baseline.

Main entry point: run_experiment.py (repo root CLI orchestrator).
"""

__version__ = "1.0.0"
