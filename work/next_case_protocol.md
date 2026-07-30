# Portfolio Maintenance: Next Case Protocol

## How to Add the Next Case Study
To keep the portfolio fresh without rebuilding from scratch, here are the exact steps to add a new case:
1. **Draft the Case (The 3-Beat Shape)**: Use the Claude Project (which retains the identity kit, voice, and styling rules) to draft the new case study in Markdown. Follow the 3-beat structure: 
   - **The Problem:** What was the business or technical challenge?
   - **What I Did:** The ML/Data approach taken.
   - **What Came of It:** The impact, results, or deployed artifact.
2. **Convert & Format**: Ask Claude to generate the HTML for the new case study, matching the layout of `capstone.html`. Save it as `[new-case-name].html` in the `docs/` folder.
3. **Update the Index**: Add a new `<div class="case-card">` block to `docs/index.html` with the title, a one-sentence summary, and a link to the new HTML file.
4. **Deploy**: Run `git add .`, `git commit -m "add new case study"`, and `git push`. GitHub Pages will automatically update the live site.

## Next Piece of Work to Add
**Name of the next case:** "Semantic Intent Classification Agent" 
*(Based on the FL-06 Agent Design Doc for the FlyRank Semantic Data Assistant).*

## Evidence of Reminder Set
**Concrete Reminder:** 
I have set a recurring Google Calendar event for **August 15th at 10:00 AM** titled: *"Portfolio Update: Draft Semantic Intent Agent case study in Claude Project."*
I have also pinned the FlyRank Claude Project in my workspace so the context (stack, voice, identity kit) is preserved for a quick, cheap update.
