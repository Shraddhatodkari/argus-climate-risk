# ADR 0001: Record architecture decisions

## Status
Accepted

## Context
This project makes a number of consequential, hard-to-reverse technical choices
(RAG over fine-tuning, multi-agent over a single prompt, local-first LLM, MCP tool
exposure). These need to be recorded with their reasoning, not just their outcome.

## Decision
Use lightweight Architecture Decision Records under `docs/adr/`, numbered
sequentially, one file per decision, following this template.

## Consequences
Every future material architecture change gets a new ADR rather than a silent
rewrite of `architecture.md`.
