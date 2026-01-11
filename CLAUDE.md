# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational lecture series web application for GWU ECE 6125 Parallel Computer Architecture (Spring 2026). A static site hosting interactive Reveal.js presentations with no build dependencies.

**Branch:** `spring-2026` - This is the Spring 2026 semester offering.

## Development Commands

```bash
# Start local development server (port 8000)
./start.sh

# Start with custom port
./start.sh 8080

# Alternative: Python server directly
python3 serve.py
```

Access at `http://localhost:8000`

## Architecture

**Static Site Structure:**
- `index.html` - Landing page with responsive lecture card grid
- `css/styles.css` - Shared styling with GWU brand colors (#004065 blue, #FFD200 gold)
- `lectures/` - Self-contained Reveal.js presentations
- `source-pdfs/` - PDF slides pending conversion to Reveal.js format

**Lecture Module Pattern:**
Each lecture folder contains:
- `index.html` - Reveal.js presentation wrapper (loads slides.md via reveal-markdown plugin)
- `slides.md` - Markdown source for slide content
- `images/` - SVG diagrams and illustrations

**Deployment:**
- Push to `spring-2026` triggers GitHub Actions (`.github/workflows/deploy.yml`)
- Auto-deploys to GitHub Pages via `gh-pages` branch

## Technology Stack

- Reveal.js 4.3.1 (CDN-hosted) for slide presentations
- Pure HTML/CSS/JS with no build step
- Python stdlib http.server for local development

## PDF Conversion Workflow

Lectures 02-04 need to be converted from PDF to Reveal.js format:

1. PDFs are in `source-pdfs/`
2. For each PDF, create `lectures/XX-topic-name/` with:
   - `index.html` (copy structure from existing lecture)
   - `slides.md` (convert PDF content to markdown)
   - `images/` (recreate diagrams as SVG)
3. Add lecture card to root `index.html`

## Key Patterns

**Slide Markdown Format:**
```markdown
## Slide Title
<!-- .slide: data-auto-animate -->

Content here

---

## Next Slide
```

**Image References in slides.md:**
```markdown
![Alt text](images/diagram.svg)
```
