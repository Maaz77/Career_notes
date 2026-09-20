# Job Application Cowork — Complete Setup Guide

This is the full setup for your 3-stage pipeline: a Cowork project, one shared folder structure, and three scheduled tasks (Discovery, Resume Prep, Apply). Paths below match your actual `Job_Finder_And_Applier` folder. Follow the steps in order — the rollout checklist at the end tells you when to test each piece before trusting it to run on its own.

---

## 1. Folder structure

This is your existing structure — nothing needs to move.

```
Job_Finder_And_Applier/                   ← give Cowork access to this folder
├── Applications/
│   └── <Company>_<JobTitle>/             ← created by Stage 1 as it finds matches
│       ├── job.md
│       └── <resume>.pdf                  ← added by Stage 1 (existing fit) or Stage 2 (custom build)
├── Default_Resume_Files/                 ← your existing pre-made resumes
│   ├── Resume-Amin-AI.pdf
│   ├── Resume-Amin-CV.pdf
│   ├── Resume-Amin-DS.pdf
│   ├── Resume-Amin-ML.pdf
│   ├── Resume-Amin-ML_V2.pdf
│   ├── Resume-Amin-ML:AI.pdf
│   └── Resume-Amin-TinyML.pdf
├── Needs-Attention/                      ← Stage 2 or 3 move blocked jobs here
├── Pending-Review/                       ← Stage 3 moves finished, ready-to-submit jobs here
├── Profile.md                            ← your search parameters + fit preferences (Section 2)
└── Resume/
    ├── build/
    ├── compile-resume.sh
    ├── fonts/
    ├── maaz-resume.cls
    ├── poetry.lock
    ├── pyproject.toml
    ├── resume.yaml
    ├── src/
    │   └── build_resume.py
    └── templates/
        └── resume.tex.j2
```

A couple of things about `Default_Resume_Files/` worth knowing, since matching happens dynamically rather than through a fixed table: there's still no Backend Engineer resume in there, so that role will always go through the custom-build path in Stage 2. And with four ML-adjacent files (`ML`, `ML_V2`, `ML:AI`, `TinyML`), Stage 1 will read all of them and use judgment case by case — more flexible than a fixed table, but its pick might not be perfectly consistent run to run unless the files themselves make their purpose clear.

---

## 2. Profile.md

This drives what Stage 1 searches for and how it judges fit — nothing about your search criteria lives in the scheduled task itself, so you can change titles, location, filters, or preferences anytime just by editing this file.

```markdown
# Profile

## Search Parameters
*(Stage 1 reads this section every run — edit freely, no need to touch the scheduled task itself)*

- **Job titles / keywords:** AI Engineer, Machine Learning Engineer, Backend Engineer, Computer Vision Engineer, Data Scientist (or close equivalents)
- **Location:** Milan, Italy
- **Posted within:** last 24 hours
- **Exclude:** (optional — e.g. specific companies, staffing agencies, on-site-only roles)

## Fit Preferences
*(used for judging whether a specific JD is genuinely a good match, beyond title/location)*

- **Seniority / experience level:**
- **What I'm looking for:**
- **Dealbreakers:**
- **Nice-to-haves:**
- **Anything else worth weighing when judging fit:**
```

Fill in Fit Preferences yourself — Search Parameters is pre-filled with what you've been using so far.

---

## 3. Cowork project setup

1. Point your Cowork project at `Job_Finder_And_Applier/`.
2. Make sure the **Claude in Chrome** connector is enabled for this project.
3. Paste this into the project's **Description** field:

```
Job Application Assistant — a 3-stage pipeline automating my daily job search on LinkedIn (roles: AI Engineer, Machine Learning Engineer, Backend Engineer, Computer Vision Engineer, Data Scientist; location: Milan, Italy). Stage 1 (scheduled) finds new listings, scores them, and uses an existing resume when one's a strong fit. Stage 2 (scheduled) custom-builds a one-page resume for anything that needs it. Stage 3 (manual, triggered by me) fills out application forms up to the final submit step, which I always do myself.
```

4. Paste this into the project's **Instructions** field (this is the operative field — it shapes behavior in every session under this project, not just a summary):

```
This project automates my job search pipeline, run across three tasks: Job Discovery & Scoring, Resume Preparation, and Application Assistant (see Scheduled).

Folder reference: Applications/ — one subfolder per job in progress. Default_Resume_Files/ — my pre-made resumes; copy from here, never edit. Resume/ — the LaTeX build project for custom resumes; the only file ever edited is resume.yaml. Pending-Review/ — finished applications waiting on me. Needs-Attention/ — anything blocked. Profile.md — my search parameters (titles, location, filters) and fit preferences; read this every run rather than relying on any hardcoded criteria.

Standing rule, above anything else in this project: never click Submit, Send Application, Send, Confirm, or any equivalent final-action button on a job application, in any task, under any circumstances. I always do that myself after reviewing.

Keep outputs concise — this project runs largely unattended.
```

---

## 4. Scheduled Task 1 — Job Discovery & Scoring

| Setting | Value |
|---|---|
| Name | `Job Discovery & Scoring` |
| Description | `Searches LinkedIn daily for new job postings matching my criteria in Profile.md, scores each for fit, and either attaches an existing resume or flags it for a custom build.` |
| Folder | `Job_Finder_And_Applier/` |
| Approval mode | **Auto** |
| Schedule | Daily, weekdays, 8:00 (adjust to taste) |

Prompt:

```
Before searching, read Profile.md in full — its Search Parameters section tells you which job titles, location, freshness window, and any exclusions to use for this run. Also read every file in Default_Resume_Files/, and Profile.md's Fit Preferences section — together these are your full picture of my background, skills, and preferences for judging fit.

Use the Claude in Chrome connector to open LinkedIn and search using exactly the titles, location, and freshness window from Profile.md's Search Parameters. Apply any exclusions listed there too.

For each listing: open it and read the full job description. Judge whether it's a genuinely strong match against what you read in Default_Resume_Files/ and Profile.md's Fit Preferences. Skip anything marginal.

For each strong match, create a folder at Applications/<Company>_<JobTitle>/ containing a job.md file with: job title, company, full job description text, link to the posting, date found, application type (LinkedIn Easy Apply vs external site), and your match reasoning.

Then decide on a resume for this job:
- If one of the files in Default_Resume_Files/ is already a strong, direct fit for this specific JD, copy it into the job's folder as-is. Note in job.md which file you used and why.
- If none of them fit well, don't copy anything — just note in job.md that this job needs a custom-built resume.

Only ever copy from Default_Resume_Files/, never edit or move the originals. Do not create a duplicate folder for a job that's already been captured on a previous run.

If Profile.md's Search Parameters section is missing or empty, stop and report that rather than guessing at criteria.
```

---

## 5. Scheduled Task 2 — Resume Preparation

| Setting | Value |
|---|---|
| Name | `Resume Preparation` |
| Description | `Builds a custom one-page resume for any job that didn't get a strong match from my existing resumes, using the LaTeX project in Resume/.` |
| Folder | `Job_Finder_And_Applier/` |
| Approval mode | **Auto** |
| Schedule | Daily, weekdays, 8:40 — after Stage 1, adjust once you see how long discovery actually takes |

Prompt:

```
Look inside Applications/ for job folders that don't yet contain a resume PDF — that's the signal a job needs a custom-built resume. (Jobs that already had a good fit in Default_Resume_Files/ were handled during discovery and already have one.)

For each one, read its job.md for the job description, then customize the resume:
- The only file you ever edit is Resume/resume.yaml. Do not modify, move, or delete any other file in the Resume/ project.
- Reset it first: from Resume/, run `git checkout -- resume.yaml`, so a previous job's edits can't bleed into this one.
- Edit only the content fields (summary, skills emphasis, bullet selection/order) to match the JD.
- Build with `./compile-resume.sh <CompanyName_JobTitle>` from the Resume/ folder. Never compile manually or invent a different command.
- The resume must fit on one page, same as the current resume's length. After every build, check the actual page count of the output PDF — don't assume it fits. If it's over, shorten content and rebuild — never shrink font size, margins, or spacing to force a fit. If two attempts don't get it to one page, stop trying.

Copy the resulting PDF into that job's Applications/<Company>_<JobTitle>/ folder. If it couldn't be built successfully, add a note to job.md explaining what went wrong, move the folder into Needs-Attention/, and move on to the next job.

When finished, send me a Telegram message with how many resumes were built and how many need attention, by navigating to:
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/sendMessage?chat_id=<YOUR_CHAT_ID>&text=<your summary, URL-encoded>
```

Fill in your actual bot token and chat ID before scheduling this.

---

## 6. Scheduled Task 3 — Application Assistant

| Setting | Value |
|---|---|
| Name | `Application Assistant` |
| Description | `Fills out job applications for matched roles up to the final submit step, then leaves each one open for my review — I always submit myself.` |
| Folder | `Job_Finder_And_Applier/` |
| Approval mode | **Auto** (not Manual — too many prompts; not Skip — removes your review gate) |
| Schedule | **None — manual trigger only.** Run this on demand, only when you're at your desk and can stay with it. |

Prompt:

```
Look inside Applications/ for job folders that already contain a resume PDF — these are ready to apply to.

For each one, in turn: open a new browser tab via the Claude in Chrome connector, navigate to the job posting, and begin the application (LinkedIn Easy Apply or the external site, whichever applies), using the resume PDF from that folder and my Profile.md for any screening questions. Fill in every step of the application as completely and accurately as you can.

Stop at the final step, immediately before the submit/send action — never click Submit, Send Application, or any equivalent button. Leave the tab open exactly as it is; don't close it or navigate away.

Once a job's application is filled and parked at that final step, move its folder into Pending-Review/.

If you hit something you can't get past — a CAPTCHA, a required account sign-up, a broken form — stop working on that one, add a note to its job.md explaining what blocked you, and move its folder into Needs-Attention/ instead. Then continue to the next ready job.

Keep going through all ready jobs in this run, one at a time, until none are left.

This rule applies for the whole task, including once every ready job is done and you're finished: do not close any tab you opened, do not call tabs_close_mcp or any other tab-closing action, and do not perform any tidy-up or cleanup step at the end of this run. Every tab must stay open exactly as you left it — I will review and close them myself.
```

---

## 7. Rollout order — don't schedule everything at once

1. Fill in Profile.md (Section 2) — both Search Parameters and Fit Preferences — before running anything.
2. Create Stage 1, but hit **Run now** instead of waiting for the schedule. Check the `job.md` files it produces — are the matches good, is the JD text complete, does the reasoning make sense, and are its resume picks (existing file vs. flagged for custom build) sensible?
3. Once satisfied, turn on Stage 1's schedule.
4. Create Stage 2, **Run now** against whatever Stage 1 left needing a custom build. Check the PDFs — page count, content quality, correct folder placement.
5. Turn on Stage 2's schedule.
6. Create Stage 3, but test it manually with just one or two ready jobs before ever running it against a full day's batch.
