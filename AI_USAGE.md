# AI Usage Protocol & Log

Project: VRScout dissertation (University of Manchester, Department of Computer Science)
Author: Jess Price
Tool: Claude Code (Anthropic, model: Claude Fable 5)
Protocol adopted: 2026-07-18, ahead of dissertation start (September 2026)

## Working protocol (strict tutor mode)

The AI assistant is used in the same capacity as a lab demonstrator or supervisor
office hours: it explains, locates, and critiques — it does not author.

The assistant MAY:

1. Explain concepts, architectures, and inherited code (e.g. the ACT/DETR-VAE codebase).
2. Produce maps/checklists of *where* changes are needed (file/line pointers), without
   writing the changes.
3. Review code I have written and flag problems — pointing at the issue, not supplying
   the corrected code.
4. Help interpret error messages and suggest *what to investigate* when debugging.
5. Point me to relevant documentation, papers, and official guidance.

The assistant MUST NOT:

1. Write, edit, or generate project source code (including "small fixes").
2. Make or pre-empt design decisions — it presents trade-offs; I decide and record why.
3. Write any prose that will appear in the dissertation document.

All design decisions and all code in this repository are my own work.

## Audit trail

- Claude Code session transcripts are retained (Claude Code stores full session logs
  locally; sessions are also summarised in the log table below).
- This file is version-controlled with the project, so the protocol and log history
  are verifiable via git.

## Usage log

| Date | Session summary | Protocol category (1–5 above) |
|------|-----------------|-------------------------------|
| 2026-07-18 | Explained UoM AI guidance; produced change-point checklist for converting CNN screen input to game-state MLP (no code written by AI) | 1, 2, 5 |

## Draft declaration (for dissertation front matter — confirm wording with supervisor)

> During this project I used Claude Code (Anthropic) as a study and review aid, under a
> fixed protocol recorded in the project repository (AI_USAGE.md). Its use was limited to:
> explaining the inherited codebase and background concepts, identifying locations in the
> code affected by my planned changes, reviewing code I had written and flagging potential
> issues, and assisting with the interpretation of error messages. All source code, design
> decisions, experiments, and written content in this dissertation are my own work. A
> per-session usage log and full transcripts are retained as an audit trail.

## To do at dissertation start (September 2026)

- [ ] First supervisor meeting: show this protocol, confirm the dissertation's AI-use
      category and the declaration wording. Record supervisor's answer here.
- [ ] Check the final-year project handbook / unit brief for any AI clause that is
      stricter than the university-level guidance.

## References

- UoM AI Guidelines: https://www.staffnet.manchester.ac.uk/ai-hub/ai-guidelines-and-policies/ai-guidelines-overview/guidelines/
- AI Guidance Summary for Students: https://documents.manchester.ac.uk/DocuInfo.aspx?DocID=71256
- FSE Teaching College — AI and Teaching & Learning: https://www.teachingcollege.fse.manchester.ac.uk/artificial-intelligence-and-teaching-and-learning/
