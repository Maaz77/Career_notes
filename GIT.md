# Git Learning Session — Full Summary

## Overview
This was a deep, progressive learning session covering Git concepts from basic 
to advanced, grounded in real team workflows. No specific project — purely 
conceptual and practical Git knowledge building.

---

## Topics Covered (In Order)

### 1. Conflicts During Rebase
- Yes, conflicts can happen during rebase
- Rebase replays commits one by one — each can conflict individually
- Options: fix and continue, skip, or abort
- Rebase can cause more conflicts than merge (once per commit vs once total)

### 2. Updating a Feature Branch With a Fix From Main
- Fix already exists on main, feature branch was created before the fix
- Solution: `git rebase main` or `git merge main` on the feature branch
- Rebase = cleaner linear history
- Merge = creates merge commit M but safer for shared branches
- If already pushed: `git push --force-with-lease`

### 3. Why Git Doesn't Just Pick the Latest Change by Time
- Git uses 3-way comparison: BASE (common ancestor), OURS, THEIRS
- Git asks "what was the original and what did each side change it to?"
- Picking by timestamp would silently discard important changes
- Git defers to the human when intent is ambiguous

### 4. Cherry-Pick
- Takes one specific commit from anywhere and replays it on current branch
- Creates a copy (X') — same changes, new commit hash
- Use cases: hotfix to main, committed on wrong branch, need part of a branch
- Can also cause conflicts for the same reasons as rebase/merge
- Overusing leads to duplicate commits across branches

### 5. Cherry-Pick Between Two Unmerged PRs
- Duplicate commits appear when both PRs eventually merge
- Not a breaking problem — final code is correct
- History becomes messy (same change appears twice)
- Better alternatives:
  - Branch feature2 off feature1
  - Extract shared code into a third branch
  - Wait for feature1 to merge, then rebase feature2

### 6. Rebasing feature2 on Top of feature1
- Possible using `git rebase --onto`
- Simple `git rebase feature1` is risky if branches have different parents
- Correct command:
```bash
  git rebase --onto feature1 main feature2
```
- `--onto feature1` = place here
- `main` = exclude everything up to main
- `feature2` = branch to replay
- Only feature2's own commits (X, Y, Z) are replayed cleanly

### 7. GitHub PR CLI Instructions (Why Merge Main Into Feature First)
- GitHub's CLI guide: checkout main → pull → checkout feature → merge main → push
- Reason: safety-first approach
- Resolve conflicts on YOUR branch, not on main
- Protects main from broken merges
- After push, GitHub PR page updates automatically — merge button becomes green

### 8. Merge Commit M
- M is created locally when you run `git merge main`
- M has TWO parents: your last commit (Z) and main's last commit (FIX)
- M does not contain new code — it's a pointer recording where branches joined
- M is created BEFORE the PR is merged on GitHub
- Clicking merge PR on GitHub just moves M onto main — no new commit created

### 9. What Happens During a Git Merge (3-Way Merge)
- Git compares BASE (common ancestor), OURS (current), THEIRS (incoming)
- Three outcomes per line:
  - Only OURS changed it → takes OURS automatically
  - Only THEIRS changed it → takes THEIRS automatically
  - Both changed it differently → CONFLICT, stops and asks you
- After resolving: `git add` → `git commit` → merge commit M created

### 10. Working Tree
- The actual files and folders on your disk
- Everything except the `.git/` folder
- Three layers of Git:
  - Working tree → staging area (git add) → repository (git commit)
- Clean = matches last commit
- Dirty = has uncommitted changes
- Git rewrites working tree files when you switch branches

### 11. What Happens When You Click Merge PR on GitHub
- GitHub runs merge on their server
- Creates merge commit on remote main
- Updates main branch pointer
- Closes the PR
- Your LOCAL machine is NOT updated — need `git pull` after
- Three merge options: Merge commit, Squash and merge, Rebase and merge

### 12. Merging Main Into Feature Before Clicking Merge PR
- Ideal and recommended flow
- GitHub recognizes it as fast-forward — no new merge commit needed
- All conflicts resolved locally before touching main
- PR merge becomes clean, tested, conflict-free

### 13. "Accept Both Changes" in VSCode Conflict Resolution
- Keeps BOTH versions, one after the other (OURS first, THEIRS second)
- Safe for additive changes (two separate log lines, etc.)
- Dangerous for contradictory changes (two return statements — second is dead code)
- When in doubt: manually edit to the correct final version

### 14. Same File, Different Lines = No Conflict
- Git only conflicts when the EXACT SAME LINE is changed differently by both sides
- Different lines in same file → auto merged silently
- Rule: Git stops only when both sides touched the same line differently

### 15. GitHub Web Conflict Editor Annotations
- `<<<<<<< branch-name` = START of YOUR version
- `=======` = DIVIDER separating the two versions
- `>>>>>>> branch-name` = END of THEIR version
- Multiple conflicts in one file = each marked independently
- All markers must be removed before GitHub allows saving
- "Mark as Resolved" = checkbox per file only (does NOT commit or merge)
- "Commit merge" = creates commit on feature branch
- "Merge Pull Request" = actually merges into base branch

### 16. VSCode Extensions for Conflict Resolution
- Built-in VSCode conflict editor: most widely used, no extension needed
- Inline actions: Accept Current, Accept Incoming, Accept Both, Compare
- 3-Way Merge Editor (VSCode 1.69+): three panels (Current, Incoming, Result)
  - Enable via `git.mergeEditor → true`
- GitLens: most popular Git extension overall, enhances conflict context
- Git Graph: visualizes branch structure

### 17. Trying Changes Before Committing Resolution
- Use VSCode 3-way merge editor Result panel (fully editable)
- Ctrl+Z / Cmd+Z undoes all the way back to original conflict markers
- `git merge --abort` → cancels entire merge, returns to pre-merge state
- Create a temp test branch to experiment safely:
```bash
  git checkout -b feature-branch-test
  git merge main
  # test → if wrong: delete branch, original untouched
```

### 18. Seeing Commits on a Branch
```bash
git log --oneline                          # commits on current branch
git log --oneline --all --graph            # visual tree of all branches
git merge-base main feature-branch         # exact branch-off commit
git log main..feature-branch --oneline     # only feature's own commits
git log main..feature-branch --stat        # with file change details
```
- Git does NOT explicitly store "branched from X" — it infers via common ancestor
- If original parent branch deleted, commit is findable but not branch name

### 19. Seeing Other Developers' Commits When Changing PR Base Branch
- PR commits tab shows: "all commits feature has that base branch doesn't"
- With base = develop: feature already has develop's commits → only yours show
- With base = release: release diverged earlier → more commits appear
- The earlier the divergence point, the more extra commits appear
- Dangerous: changing base to release can include unintended develop commits
- Fix options: cherry-pick only your commits, create fresh branch off release,
  or keep PR on develop and let normal flow handle it

### 20. Can Only Merge Your Commits Into Release (Not Others')
- PR itself cannot selectively merge commits — all or nothing
- Options:
  - Cherry-pick your commits directly to release
  - Create fresh branch off release, cherry-pick your commits, open new PR
  - Keep PR on develop, let normal release flow handle it

### 21. Reverting a Merged PR
- GitHub has built-in Revert button at bottom of merged PR page
- Creates new branch `revert-PR#-feature-branch` automatically
- Creates new PR that undoes all changes
- Revert does NOT delete original commit — creates new reverse commit
- Command line: `git revert -m 1 <merge-commit-hash>`
- `-m 1` required for merge commits (reverts to first parent)
- If you want to re-merge later: revert the revert
```bash
  git revert <revert-commit-hash>
```

### 22. Resolving Conflicts Without Merging on GitHub Web
- Not possible — GitHub's web conflict editor is tied to the merge action
- Correct approach: resolve locally in VSCode
```bash
  git checkout main && git pull
  git checkout feature-branch
  git merge main
  # resolve in VSCode
  git add . && git commit
  git push origin feature-branch
```
- After push: PR page updates automatically, conflicts gone, merge button green

### 23. Squash Merge Main Into Feature (Problem)
- Wrong tool for updating feature branch with main
- Squash breaks the parent link — creates flat commit with no pointer back to main
- PR may show unexpected conflicts, other devs' commits, or duplicate changes
- Fix:
```bash
  git reset --hard HEAD~1         # remove squash commit
  git merge main                  # or rebase main
  git push --force-with-lease origin feature-branch
```

### 24. git reset --hard vs remote
- `git reset --hard` = local only, remote untouched
- `git push --force-with-lease` = syncs removal to remote
- Both steps required to fully remove a commit from everywhere

### 25. --ff-only and --no-ff Merge Flags
- Fast forward: when base branch hasn't moved, Git just moves pointer forward
- `--ff-only`: only merge if fast forward possible, otherwise refuse
- `--no-ff`: always create merge commit even if fast forward possible
- Teams use `--no-ff` to preserve branch history visibility
- Teams use `--ff-only` for clean linear history (usually after rebase)

### 26. HEAD
- A pointer that always points to the commit you are currently on
- Moves with you as you switch branches or make commits
- "Compare with HEAD" in GitLens = compare selected branch with your current position
- Detached HEAD = when you checkout a specific commit instead of a branch

### 27. Undoing a Resolved Conflict in VSCode
- Ctrl+Z / Cmd+Z: works most of the time, goes back to conflict markers
- `git merge --abort`: cancels entire merge, returns to pre-merge state
- Already committed wrong resolution: `git reset --hard HEAD~1` then merge again

### 28. Updating Both feature1 and feature2 (Bottom-Up Order)
- Always update in order: main → feature1 → feature2
- Rebase approach:
```bash
  git checkout main && git pull
  git checkout feature1 && git rebase main
  git push --force-with-lease origin feature1
  git checkout feature2 && git rebase feature1
  git push --force-with-lease origin feature2
```
- Merge approach (no force push needed):
```bash
  git checkout main && git pull
  git checkout feature1 && git merge main && git push origin feature1
  git checkout feature2 && git merge feature1 && git push origin feature2
```
- After initial `rebase --onto`, simple `git rebase feature1` is sufficient
  (no need for --onto anymore since histories are aligned)

### 29. Why Rebasing feature1 Doesn't Include feature2
- Branches are independent pointers — moving one never moves another
- After rebasing feature1, feature2 is orphaned (still points to old commits)
- Must explicitly rebase feature2 on top of new feature1 afterward

### 30. Why Conflicts Happen During Rebase
- Same 3-way comparison as merge (BASE, OURS, THEIRS) per commit
- Rebase replays the diff of each commit — not a copy-paste
- If both the replayed commit and new base touched same lines → conflict
- More painful than merge: one conflict check per commit being replayed

### 31. Merging Main Into Feature — Is It Always Required?
- Not required if main moved forward with zero overlap to feature files
- Git will say "Already up to date" if nothing to bring in
- Still recommended for: team PR policies, indirect dependency safety, 
  keeping branch close to main

### 32. Checking If Branch Is Outdated
- GitHub PR page: "This branch is out of date with the base branch" ⚠️
- Command line: `git log feature-branch..main --oneline`
  - Empty output = up to date
  - Shows commits = behind main
- `git fetch origin` + `git status` = quick local check

### 33. Teammate Pushed to Same Feature Branch
- Problem: local and remote diverged
- Fix:
```bash
  git fetch origin
  git rebase origin/feature-branch
  git push origin feature-branch   # no force push needed
```
- DO NOT force push — would overwrite teammate's commit

### 34. Rolling Back and Deleting a Pushed Commit
```bash
git checkout feature-branch
git reset --hard HEAD~1                          # remove locally
git push --force-with-lease origin feature-branch # remove from remote
```
- `--soft` = keep changes staged
- `--hard` = discard changes completely

### 35. Renaming a Branch
- Local rename: `git branch -m old-name new-name` (remote untouched)
- To rename on remote too:
```bash
  git branch -m old-name new-name
  git push origin new-name
  git push origin --delete old-name
  git branch --set-upstream-to=origin/new-name new-name
```
- No commit needed to push after renaming — existing commits push fine
- Warning: open PRs pointing to old name will break

### 36. Interactive Rebase
- `git rebase -i main` opens editor with all commits to be replayed
- Commands per commit:
  - `pick` / `p` = keep as is
  - `reword` / `r` = edit commit message
  - `edit` / `e` = pause and edit the commit
  - `squash` / `s` = merge into previous commit, combine messages
  - `fixup` / `f` = merge into previous, discard message
  - `drop` / `d` = delete commit entirely
- Common use: clean up messy commits before opening a PR
- Rewrites history → force push needed after

### 37. Previewing What Will Be Replayed Before Rebase
```bash
git log main..feature2 --oneline          # commits that WILL be replayed
git log feature2..feature1 --oneline      # what feature2 is missing from feature1
git log --oneline --graph --all           # full visual of all branches
```

---

## Key Principles Established
- Git thinks in CHANGES not TIME — never picks "latest by timestamp"
- Working tree is local files on disk — separate from Git's stored history
- Branches are independent pointers — moving one never moves another
- Remote and local are always separate — nothing syncs without an explicit push/pull
- force push rewrites remote history — always use `--force-with-lease` not `--force`
- Never force push to main, develop, or release
- Always update branches bottom-up: main → feature1 → feature2
- Resolve conflicts locally in VSCode, not on GitHub web editor
- Squash merge is for feature → main direction, not for updating feature with main
- rebase --onto = surgical precision when branches have different parents

---

## Current State
Session was purely educational — no open tasks or unresolved questions at end.
All topics were fully explained and confirmed understood before moving on.