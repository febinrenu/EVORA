# evora kit — start here (5 minutes)

1. Clone the team repo (M1 creates it and shares the URL).
2. Run the installer from this kit, pointing at your clone:
   `bash setup-local.sh /path/to/evora`
   (Windows: run it in Git Bash or WSL.)
3. In the clone: `git config user.name "Your Name"` and `git config user.email "you@example.com"`.
4. Create your own Groq API key in your own Groq account (console.groq.com → API Keys). Send it to the ingestion-box owner privately; never commit it.
5. Open PLAN.md: read §0, §1, §2, §7, your §8.x and §12.
6. Start your coding session in the repo root, run `/status`, then paste your kickoff prompt from PLAN.md §12.3.

What the installer does: copies PLAN.md, CLAUDE.md and .claude/settings.local.json into the clone; adds local ignore rules in .git/info/exclude so they can never be committed; installs two local git hooks (commit-msg, pre-commit); creates PROGRESS.md and .gitattributes if they are missing (M1 commits those two once).
