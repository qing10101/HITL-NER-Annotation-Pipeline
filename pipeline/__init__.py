"""Cascading Multi-Agent Inline Boundary Tagging & Audit Engine.

A privacy-NER data labeling pipeline:
  Stage 1 (gemini-3.5-flash)  -> inline XML tagging (verbatim rewrite)
  Stage 2 (gpt-5.5)           -> cross-family structural/semantic audit
  Stage 4 (regex, non-AI)     -> deterministic char-offset span extraction
"""

TAGSET = ("MINOR_AGE", "MINOR_EDU", "GEN_NOUN", "GEN_PHYS", "FAM_KIN")
