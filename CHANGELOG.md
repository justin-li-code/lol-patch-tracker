# Changelog

Every version below came from one round of requests. Newest first.

## Up next
- Publish the site on Streamlit Community Cloud.
- Knockout bracket odds (chance to win Worlds) once the Swiss stage ends.
- Refresh the Worlds data after each Swiss round and knockout stage.
- Ideas, not started: model item and rune changes (they often move the meta more than champion changes); player ratings across the whole domestic season, not just international events.

## v1.4 · Oct 27: Upcoming matches and previews
- New **Upcoming** tab in Worlds 2026: the next matches (Swiss Round 4) with the favorite and their chance to win the series.
- **Match previews** (click a match, or pick any two teams):
  - Win chances for one game and for the series.
  - Each side's likely bans and likely game-1 picks by role.
  - Players to watch, and head-to-head results this year.
- **Swiss standings** with each team's chance to reach the knockouts, from playing out the rest of the Swiss stage 20,000 times.
- **Track record:** how often our pre-game favorite has actually won at Worlds (29 of 51 so far).
- The schedule comes from Leaguepedia when it responds; the latest known schedule is kept as a backup.

## v1.3 · Oct 27: Plain language, more champions, richer player stats
- **Written for everyone, not just data people.**
  - "Presence" is now **pick/ban rate** everywhere, explained in the sidebar, with changes shown as "points".
  - Statistics jargon is gone. Forecasts are "off by X points", and likely ranges replace confidence intervals.
  - Every page and explanation is rewritten in everyday terms.
- **The Overview's prediction card explains itself:** what was predicted, what it was compared with, where it helped, and where it missed (e.g. pros kept picking the just-nerfed Nocturne at Worlds).
- **More champions:**
  - The Worlds chart shows 25 champions, with a full list of every champion picked or banned there.
  - The forecast chart shows 30, the region map 40 and the Worlds region table 30.
  - The Overview lists the top 5, and other lists are longer.
- **Richer player stats, counted in the rating:**
  - Damage, gold and kill participation, each as a share of the team and of the whole game.
  - Damage and gold against the lane opponent, and damage per gold.
  - Game pages show them in the main player table plus a "More stats" panel.
- Champion names use their official spelling (K'Sante, Kha'Zix, Cho'Gath).

## v1.2 · Oct 27: Champion records in game pages
- In each game's player table, "Games on it" is now **# of Games**, next to a new **Record on Champ** column: the player's wins–losses on that champion this season before the game.

## v1.1 · Oct 27: Series, Worlds Players and champion-aware expectations
- **Series instead of loose games:** Worlds 2026 → *Series & Games* lists each series with its score and format (e.g. "2–1 · Bo3"). Open one, then pick Game 1, 2 or 3.
- **New Worlds Players page:** Best, Lowest Rated, Above Expectations, Below Expectations, and a Player Lookup with each player's most-played champions this season.
- **Expected rating for every player in every game:** from their earlier games (form), how they've done on that champion compared with their usual level, their team's chance of winning, and, at international events, how strong their region has been. Game pages show it next to the actual rating, along with how many games the player had on that champion.
- Ratings now cover every 2026 pro game, domestic and international, on one common scale (2,397 games). That's what makes champion history and "season vs Worlds" comparisons possible.
- **What champion history adds:** the expectation tracks actual Worlds ratings noticeably better with it (correlation 0.24 vs 0.20 without). Single games stay noisy, though: a player's rating swings about 1.1 points from game to game. Expectations are most useful over several games, as in the Above and Below Expectations boards.
- Series show readable stages ("Play-In Round 1", "Swiss Round 3") and their format (Bo1, Bo3, Bo5).
- Game pages show both teams' players straight away, no click needed, with a **#** rank (1–10) for every player in the game. Team stats, draft and gold/objectives sit in tabs below.
- Ranking numbers on every Worlds Players board, and each game's rank in the Player Lookup.
- Menu and tab names are now in Title Case.

## v1.0 · Oct 27: Worlds games, player ratings and expected results
- **Games:** click any Worlds game for the full picture: result and pre-game odds, team stats (kills, gold, towers, dragons, barons, heralds, voidgrubs, plates, first blood/tower), bans and picks in draft order, every player's stats, and the gold-lead curve with objectives.
- **Player ratings out of 5:** each stat is compared with players in the same role at this year's international events, weighted for what the role does (vision matters for supports, lane leads for laners). The result counts relative to expectation, so an upset win is worth more than an expected one. 3.0 is typical; 4.5+ is the top 10%.
- **Players:** a Worlds leaderboard by average rating, with each player's games and what drives their rating.
- **Expected results:** a pre-game win chance for every international game, from team strength (all 2026 games, with cross-region games linking the leagues), using only games played before that day.
  - Tested on First Stand, MSI, EWC and the Demacia Cup before Worlds was scored. Adding the teams' drafts and champion comfort didn't improve the predictions there, so the expectation uses team strength; each game's draft edge is still shown.
- Fixed: stray `**` asterisks on the Overview and Champions pages.
- Refreshed Worlds to 51 games (Play-In and Swiss rounds 1–3).

## v0.9 · Oct 27: Changelog
- Added this changelog, also shown on the site's **Changelog** page.

## v0.8 · Oct 25: Easier to read
- Replaced nine tabs with sidebar pages, so only one page shows at a time.
- New **Overview** page: four headline takeaways, each linking to its page.
- Every page now opens with a one-line summary and one main chart. Methods, caveats and long tables fold into expandable panels and sub-tabs.
- Tables show the top 10 rows, with "Show all" for the rest.
- Plainer labels, fewer numbers on screen, and a color theme that matches the charts.

## v0.7 · Oct 24: Ready for the web
- Added the missing packages so the app installs on a hosting service.
- Added a data-source credit and Riot's fan-project disclaimer to every page.
- Put the project on GitHub (`justin-li-code/lol-patch-tracker`) with the processed data, so the site runs without scraping.

## v0.6 · Oct 24: Champion groups and draft answers
- **Rise & fall:** pick any group of champions (e.g. Nocturne + Qiyana) to see, by name, who lost ground as they rose, and whether those champions were actually pushed out or just declining at the same time.
- **Draft answers:** when a team locks a champion, what the opponent answers with in the same role. Draft order is recovered from how gol.gg groups each team's picks.
  - Patterns are compared with how often the answer is picked on the same patches, so two champions that were just popular together don't count.
  - Each pattern is tagged with who leads it: everyone, Worlds, one region, one team or one player.
- Downloaded every game page: 2,384 games, 366 players, and exact roles for every pick.
- Removed the champion-class analysis in favor of named champions.

## v0.5 · Oct 24: Ripple effects
- Added pro roles, inferred from team compositions. Checked against the exact roles later fetched in v0.6, the inference was right 98.8% of the time.
- **Who took the slot:** after a big buff or nerf, which untouched champion in the same role moved the other way (e.g. Varus nerfs → Ezreal, twice).
- Tested whether rivals' nerfs lift a champion on average: no reliable effect. Ripples are case-by-case.
- Finding: the biggest class swing of 2026 was in support, enchanters vs tank supports.
- The idea that Nocturne rose because tanks fell wasn't supported by the data.

## v0.4.2 · Oct 24: Clearer patches, cleaner charts
- Removed portraits from chart axes, where they covered champion names.
- Made it explicit that Worlds is on **patch 26.20**. The leagues stopped at 26.17, so Worlds is the only pro data on 26.18–26.20.
- New Worlds section: what changed in 26.18, 26.19 and 26.20, and how those champions are doing at Worlds.
- The champion timeline now shows international events (e.g. "26.20 Worlds") as their own points.
- Fixed: a change marker for 26.17 was drawn at the far right of the timeline; events on the same patch now appear in date order.

## v0.4.1 · Oct 24: Chart spacing
- Fixed-size rows for every champion chart, and legends moved above the plot area.
- Forecast chart limited to the 30 most relevant champions.

## v0.4 · Oct 24: Portraits, logos and CBLOL
- Champion portraits beside names in tables and pickers.
- Team logos, plus a **Teams at Worlds** table: record, region, go-to picks.
- Added **CBLOL** as a fifth league (275 games).

## v0.3 · Oct 24: More events, regions and teams
- Added First Stand, MSI and Esports World Cup as tests. The patch model beat "nothing changes" at all three.
- Tested an "international adjustment" (top picks hold at events) on those earlier events; it made forecasts worse, so it was dropped.
- **Regions & teams:** what each region and team prioritizes, signature picks, and what opponents ban against them.
- **Does priority win?** Drafting the most-contested champions, comfort picks and home-region favorites showed no reliable edge once team strength is accounted for.
  - Two analysis bugs found and fixed along the way: same-patch data leaking results, and a winner-first ordering in the source data.

## v0.2 · Oct 24: Worlds as the centerpiece
- Added Worlds 2026 (Play-In + Swiss) and the Demacia Cup as out-of-sample tests, with forecasts saved before scoring.
- Live Worlds estimate for the rest of the tournament, updated as games are played.
- Fixed the buff/nerf classifier: Base Mana changes and some of Riot's phrasings were misread.

## v0.1 · Oct 23: First version
- Scraped 2026 pro games from gol.gg (LCK, LPL, LCS, LEC) and Riot's patch notes 26.1–26.21.
- Auto-classified each champion change as a buff, nerf or adjustment.
- Measured how presence (picks + bans per game) moves after each change: on average a buff adds, and a nerf takes away, roughly 3–4 points.
- A model to predict the next patch, tested patch by patch on earlier data only. A smaller model replaced the first one after that lost to the "nothing changes" baseline.
- First dashboard: buffs vs nerfs, champion timeline, patch explorer and predictions.
