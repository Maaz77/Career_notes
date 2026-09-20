JOB APPLICATION AUTOMATION PROJECT — CONTEXT EXPORT
Exported from a Claude conversation for continuation on another AI service.

═══════════════════════════════════════════════════════════
BACKGROUND
═══════════════════════════════════════════════════════════

User is an AI/ML-focused job seeker based in Milan, Italy, targeting AI
Engineer, Machine Learning Engineer, Backend Engineer, Computer Vision
Engineer, and Data Scientist roles. Actively job searching, building a
structured, largely automated LinkedIn application workflow using Claude
Desktop (Cowork + Claude in Chrome), inside a project/folder called
"Job_Finder_And_Applier."

═══════════════════════════════════════════════════════════
HOW WE GOT HERE — DECISIONS AND WHY (read before changing anything)
═══════════════════════════════════════════════════════════

1. ORIGINAL IDEA (rejected): A fully custom agentic system — Python,
   LangGraph for orchestration, LangSmith for observability, Browser-Use
   as the browser agent — to autonomously search LinkedIn, score jobs,
   fill applications, and use Telegram for human-in-the-loop CAPTCHA
   handling and pre-submit approval.

2. FEASIBILITY RESEARCH (done before pivoting away from the custom build):
   - Automation reliability is uneven by branch: LinkedIn Easy Apply and
     template ATSs (Greenhouse, Lever) are reasonably automatable
     (~75-90%); Workday and any ATS requiring account creation are
     unreliable (~30-50%) and need heavy human intervention.
   - LinkedIn's User Agreement prohibits automation, and enforcement/ban
     risk is real (documented takedowns of automation vendors). Verdict:
     keep this a personal/local tool, never a multi-user product, and
     keep "reading job postings" decoupled from "automating a logged-in
     session" wherever possible.
   - Browser-Use was assessed as a reasonable open-source agentic core
     but slow; recommended a hybrid of deterministic scripts for known
     ATS templates + agentic fallback for unknown sites.
   - Alibaba's "page-agent" tool was investigated and rejected as
     unusable here: it injects a script into the page itself, and
     LinkedIn's Content-Security-Policy blocks exactly that kind of
     externally-injected script.

3. PIVOT: Given that research, decided NOT to build a custom agent.
   Instead, use Claude Desktop (Cowork) with the Claude in Chrome
   connector directly. Key platform constraints that shaped everything
   downstream:
   - Claude cannot create accounts, under any permission setting.
   - Claude cannot bypass CAPTCHAs.
   - Claude requires explicit human approval before entering sensitive
     data into a form field, and before any irreversible action like
     clicking Submit — even in "Auto" approval mode.
   - Conclusion: read-only work and local file edits can run scheduled
     and unattended; anything touching a real form or a submit button
     cannot be truly unattended and must be a human-supervised session.

4. FINAL ARCHITECTURE — a 3-stage pipeline:
   - Stage 1 "Job Discovery & Scoring" (scheduled, unattended): searches
     LinkedIn, scores matches, and does the easy resume-selection case.
   - Stage 2 "Resume Preparation" (scheduled, unattended): handles only
     the harder case — building a custom resume via the LaTeX project
     when no pre-made resume is a good fit.
   - Stage 3 "Application Assistant" (NOT scheduled — manually triggered
     only, attended): fills out real application forms, stops before
     the final submit action, leaves tabs open. User reviews and submits
     every application personally. This is a hard, non-negotiable rule.

5. OTHER KEY DECISIONS AND WHY (don't relitigate these without reason):
   - No static "job title → resume file" lookup table. Rejected because
     it goes stale as resume files are added/renamed/removed. Instead,
     Stage 1 reads the live contents of Default_Resume_Files/ and
     Profile.md every run and judges fit dynamically.
   - No separate AGENT_INSTRUCTIONS.md file for resume-build rules. One
     was created, then deliberately eliminated once it became clear only
     Stage 2 ever used that content — folded directly into Stage 2's own
     prompt instead of a shared file no one else needed.
   - Whitelist over blacklist for file safety: Stage 2 is told "the only
     file you ever edit is resume.yaml" rather than listing every file
     it must NOT touch (which would need updating every time a new file
     is added to the Resume/ project).
   - Search criteria (titles, location, freshness window, exclusions)
     live in Profile.md, not hardcoded in Stage 1's prompt, so they can
     change without ever touching the scheduled task itself.
   - Resume one-page constraint is enforced by actually checking the
     output PDF's page count after every build and shortening content
     (never shrinking font/margins/spacing) if it's over — stop after
     two failed attempts rather than degrade the layout.
   - Resume/ is git-tracked; Stage 2 runs `git checkout -- resume.yaml`
     before each new job so one job's edits can't bleed into the next.
   - Telegram notifications use the Telegram Bot API's sendMessage
     endpoint via a plain URL navigation (no formal integration needed).
     Originally on both Stage 1 and Stage 2; Stage 1's was later removed
     at the user's request. Stage 2 still sends one.
   - Tab-closing bug: user observed Claude in Chrome closing tab groups
     after Stage 3 finished, which would defeat the whole point of
     leaving applications open for review. Research confirmed this is
     NOT a platform default (tab groups are documented to persist/
     accumulate by default) — root cause was an instruction gap: the
     original prompt said not to close a tab while moving between jobs,
     but never explicitly said the same applies at the very end of the
     whole task. Fixed with an explicit end-of-task no-cleanup rule.
   - Approval mode for all three tasks: Auto (Manual is too naggy for
     Stage 3's many fields; Skip would remove the review gate that's the
     entire point).
   - Known open issue, not yet resolved: Default_Resume_Files/ has no
     resume for "Backend Engineer," and has 4 ambiguous ML-adjacent
     files (ML, ML_V2, ML:AI, TinyML) with no automatic disambiguation.
     Stage 1 treats ambiguous ones as needing a custom build. User may
     want to clean these up.

6. NOT YET DONE / OPEN ITEMS:
   - Profile.md's "Fit Preferences" section is still blank — only
     Search Parameters has been filled in. User needs to write this.
   - The pipeline has not been rolled out or tested yet — see the
     Rollout order checklist at the end of this export.

═══════════════════════════════════════════════════════════
CURRENT FINAL STATE — COMPLETE, VERBATIM SETUP
═══════════════════════════════════════════════════════════

--- 1. FOLDER STRUCTURE (existing, nothing needs to move) ---

Job_Finder_And_Applier/                   ← Cowork project root folder
├── Applications/
│   └── <Company>_<JobTitle>/             ← created by Stage 1 per match
│       ├── job.md
│       └── <resume>.pdf                  ← added by Stage 1 or Stage 2
├── Default_Resume_Files/                 ← pre-made resumes
│   ├── Resume-Amin-AI.pdf
│   ├── Resume-Amin-CV.pdf
│   ├── Resume-Amin-DS.pdf
│   ├── Resume-Amin-ML.pdf
│   ├── Resume-Amin-ML_V2.pdf
│   ├── Resume-Amin-ML:AI.pdf
│   └── Resume-Amin-TinyML.pdf
├── Needs-Attention/                      ← Stage 2/3 move blocked jobs here
├── Pending-Review/                       ← Stage 3 moves finished jobs here
├── Profile.md                            ← search parameters + fit preferences
└── Resume/                               ← Poetry-managed Python + LaTeX project
    ├── build/
    ├── compile-resume.sh
    ├── fonts/
    ├── maaz-resume.cls
    ├── poetry.lock
    ├── pyproject.toml
    ├── resume.yaml                       ← the only file ever edited
    ├── src/
    │   └── build_resume.py
    └── templates/
        └── resume.tex.j2

--- 2. Profile.md (starter content — Fit Preferences still needs filling in) ---

# Profile

## Search Parameters
*(Stage 1 reads this section every run — edit freely)*

- Job titles / keywords: AI Engineer, Machine Learning Engineer, Backend
  Engineer, Computer Vision Engineer, Data Scientist (or close equivalents)
- Location: Milan, Italy
- Posted within: last 24 hours
- Exclude: (optional)

## Fit Preferences
*(used for judging whether a specific JD is genuinely a good match)*

- Seniority / experience level:
- What I'm looking for:
- Dealbreakers:
- Nice-to-haves:
- Anything else worth weighing when judging fit:

--- 3. COWORK PROJECT SETUP ---

Description field:

Job Application Assistant — a 3-stage pipeline automating my daily job
search on LinkedIn (roles: AI Engineer, Machine Learning Engineer,
Backend Engineer, Computer Vision Engineer, Data Scientist; location:
Milan, Italy). Stage 1 (scheduled) finds new listings, scores them, and
uses an existing resume when one's a strong fit. Stage 2 (scheduled)
custom-builds a one-page resume for anything that needs it. Stage 3
(manual, triggered by me) fills out application forms up to the final
submit step, which I always do myself.

Instructions field (operative — shapes every session in this project):

This project automates my job search pipeline, run across three tasks:
Job Discovery & Scoring, Resume Preparation, and Application Assistant.

Folder reference: Applications/ — one subfolder per job in progress.
Default_Resume_Files/ — my pre-made resumes; copy from here, never edit.
Resume/ — the LaTeX build project for custom resumes; the only file
ever edited is resume.yaml. Pending-Review/ — finished applications
waiting on me. Needs-Attention/ — anything blocked. Profile.md — my
search parameters and fit preferences; read this every run rather than
relying on any hardcoded criteria.

Standing rule, above anything else in this project: never click Submit,
Send Application, Send, Confirm, or any equivalent final-action button
on a job application, in any task, under any circumstances. I always do
that myself after reviewing.

Keep outputs concise — this project runs largely unattended.

--- 4. SCHEDULED TASK 1 — Job Discovery & Scoring ---

Name: Job Discovery & Scoring
Description: Searches LinkedIn daily for new job postings matching my
  criteria in Profile.md, scores each for fit, and either attaches an
  existing resume or flags it for a custom build.
Folder: Job_Finder_And_Applier/
Approval mode: Auto
Schedule: Daily, weekdays, 8:00

Prompt:

Before searching, read Profile.md in full — its Search Parameters
section tells you which job titles, location, freshness window, and any
exclusions to use for this run. Also read every file in
Default_Resume_Files/, and Profile.md's Fit Preferences section —
together these are your full picture of my background, skills, and
preferences for judging fit.

Use the Claude in Chrome connector to open LinkedIn and search using
exactly the titles, location, and freshness window from Profile.md's
Search Parameters. Apply any exclusions listed there too.

For each listing: open it and read the full job description. Judge
whether it's a genuinely strong match against what you read in
Default_Resume_Files/ and Profile.md's Fit Preferences. Skip anything
marginal.

For each strong match, create a folder at Applications/<Company>_
<JobTitle>/ containing a job.md file with: job title, company, full job
description text, link to the posting, date found, application type
(LinkedIn Easy Apply vs external site), and your match reasoning.

Then decide on a resume for this job:
- If one of the files in Default_Resume_Files/ is already a strong,
  direct fit for this specific JD, copy it into the job's folder as-is.
  Note in job.md which file you used and why.
- If none of them fit well, don't copy anything — just note in job.md
  that this job needs a custom-built resume.

Only ever copy from Default_Resume_Files/, never edit or move the
originals. Do not create a duplicate folder for a job that's already
been captured on a previous run.

If Profile.md's Search Parameters section is missing or empty, stop and
report that rather than guessing at criteria.

--- 5. SCHEDULED TASK 2 — Resume Preparation ---

Name: Resume Preparation
Description: Builds a custom one-page resume for any job that didn't
  get a strong match from my existing resumes, using the LaTeX project
  in Resume/.
Folder: Job_Finder_And_Applier/
Approval mode: Auto
Schedule: Daily, weekdays, 8:40 (after Stage 1)

Prompt:

Look inside Applications/ for job folders that don't yet contain a
resume PDF — that's the signal a job needs a custom-built resume.
(Jobs that already had a good fit in Default_Resume_Files/ were handled
during discovery and already have one.)

For each one, read its job.md for the job description, then customize
the resume:
- The only file you ever edit is Resume/resume.yaml. Do not modify,
  move, or delete any other file in the Resume/ project.
- Reset it first: from Resume/, run `git checkout -- resume.yaml`, so a
  previous job's edits can't bleed into this one.
- Edit only the content fields (summary, skills emphasis, bullet
  selection/order) to match the JD.
- Build with `./compile-resume.sh <CompanyName_JobTitle>` from the
  Resume/ folder. Never compile manually or invent a different command.
- The resume must fit on one page, same as the current resume's length.
  After every build, check the actual page count of the output PDF —
  don't assume it fits. If it's over, shorten content and rebuild —
  never shrink font size, margins, or spacing to force a fit. If two
  attempts don't get it to one page, stop trying.

Copy the resulting PDF into that job's Applications/<Company>_
<JobTitle>/ folder. If it couldn't be built successfully, add a note to
job.md explaining what went wrong, move the folder into
Needs-Attention/, and move on to the next job.

When finished, send me a Telegram message with how many resumes were
built and how many need attention, by navigating to:
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/sendMessage?chat_id=
<YOUR_CHAT_ID>&text=<your summary, URL-encoded>

(Fill in your actual bot token and chat ID before scheduling this.)

--- 6. SCHEDULED TASK 3 — Application Assistant ---

Name: Application Assistant
Description: Fills out job applications for matched roles up to the
  final submit step, then leaves each one open for my review — I always
  submit myself.
Folder: Job_Finder_And_Applier/
Approval mode: Auto (not Manual — too many prompts; not Skip — removes
  the review gate)
Schedule: None — manual trigger only, run when at the desk and able to
  stay with it

Prompt:

Look inside Applications/ for job folders that already contain a resume
PDF — these are ready to apply to.

For each one, in turn: open a new browser tab via the Claude in Chrome
connector, navigate to the job posting, and begin the application
(LinkedIn Easy Apply or the external site, whichever applies), using the
resume PDF from that folder and my Profile.md for any screening
questions. Fill in every step of the application as completely and
accurately as you can.

Stop at the final step, immediately before the submit/send action —
never click Submit, Send Application, or any equivalent button. Leave
the tab open exactly as it is; don't close it or navigate away.

Once a job's application is filled and parked at that final step, move
its folder into Pending-Review/.

If you hit something you can't get past — a CAPTCHA, a required account
sign-up, a broken form — stop working on that one, add a note to its
job.md explaining what blocked you, and move its folder into
Needs-Attention/ instead. Then continue to the next ready job.

Keep going through all ready jobs in this run, one at a time, until
none are left.

This rule applies for the whole task, including once every ready job is
done and you're finished: do not close any tab you opened, do not call
tabs_close_mcp or any other tab-closing action, and do not perform any
tidy-up or cleanup step at the end of this run. Every tab must stay
open exactly as you left it — I will review and close them myself.

--- 7. ROLLOUT ORDER (not yet executed) ---

1. Fill in Profile.md's Fit Preferences section (Search Parameters is
   already filled in).
2. Create Stage 1, Run Now (not scheduled yet). Check the job.md files
   it produces for quality before trusting it.
3. Turn on Stage 1's schedule.
4. Create Stage 2, Run Now against Stage 1's output. Check the PDFs.
5. Turn on Stage 2's schedule.
6. Create Stage 3, test manually with 1-2 jobs before a full batch.