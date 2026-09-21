# Climate source coverage — the 16 added climate decisions

*Drafted 2026-09-21, Phase 1. Curated by Vidhi Bhutani on 2026-09-21: all nine judgements recorded as true for each of the 16.*

The five pilot climate candidates drew on POLIANNA, whose climate material could not
support sixteen further decisions without superficial variants. Three consolidated EU
acts were therefore added to the source registry from EUR-Lex (official PDF renditions,
pinned by SHA-256 in `data/sources/downloads.yaml`):

- `eurlex_ets_directive` — Directive 2003/87/EC (EU ETS), consolidated 2024-03-01
- `eurlex_effort_sharing` — Regulation (EU) 2018/842 (Effort Sharing), consolidated 2023-05-16
- `eurlex_lulucf` — Regulation (EU) 2018/841 (LULUCF), consolidated 2023-05-11

Each is recorded as **citable** in `data/sources/registry.yaml`, seed-only, under CC BY 4.0
with attribution: Vidhi Bhutani reviewed the EUR-Lex legal notice and Commission Decision
2011/833/EU and confirmed their use on 2026-09-21. The consolidated text remains a
documentation tool with no legal effect; that caveat is kept in each registry entry.

Every article below was checked in the pinned text: the heading is present
(`scripts/check_full_topic_bank.py` re-checks this mechanically), and the article was read
to confirm it contains the mechanism the decision turns on. The source supplies only that
mechanism: each two-option decision, and its trade-off, is constructed for this study and
is not claimed to be an option the act itself offers. **Sixteen distinct climate
decisions are supported.** A seventeenth candidate (afforestation, LULUCF) was dropped
because its trade-off duplicated climate_21. climate_18 was replaced on 2026-09-21: the
earlier corrective-action decision misstated Effort Sharing art. 8, under which the
government prepares its own corrective plan and the Commission may only give an opinion.

## Why these are climate decisions, not energy decisions

Each turns on how greenhouse-gas emissions or removals are priced, counted, allocated or
traded — the instruments of climate policy — rather than on how energy is produced,
sold, networked or used. The per-row reason is given below.

## Matrix

### climate_06 — free allocation to trade-exposed industry

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, arts. 10a and 10b (free allocation, carbon leakage) |
| Decision | A regional emissions-trading scheme must decide whether to keep giving free emission allowances to heavy industry that competes with overseas producers or to make those firms buy every allowance. |
| Option 1 | Keep giving free allowances to trade-exposed heavy industry. |
| Option 2 | Require trade-exposed heavy industry to buy every allowance it uses. |
| Trade-off | industrial competitiveness and leakage protection vs the polluter-pays price signal |
| Distinct from the pilot and the other proposals | Only decision about how allowances reach industry. climate_02 prices aviation; climate_01 regulates vehicles; no energy decision covers industrial allocation. |
| Why climate, not energy | Concerns the allocation rules of a greenhouse-gas trading scheme, not electricity or energy markets. |

### climate_07 — use of auction revenues

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 10 (auctioning of allowances; use of revenues) |
| Decision | A regional government must decide whether to spend the money raised by selling emission allowances on rebates for households or on projects that cut emissions. |
| Option 1 | Return the revenue to households as direct rebates. |
| Option 2 | Invest the revenue in projects that cut emissions. |
| Trade-off | distributional relief and public acceptance vs additional emission reductions |
| Distinct from the pilot and the other proposals | Unlike climate_04 (budget for measurement vs cuts), this concerns compensating households vs investing. Distinct from climate_12 (who receives a fund) by asking what revenue is spent on. |
| Why climate, not energy | Revenue from a carbon-pricing instrument; the decision is about climate-policy fiscal design, not energy supply. |

### climate_08 — price-intervention mechanism

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 29a (measures on excessive price fluctuations) |
| Decision | A regional emissions-trading scheme must decide whether to release extra allowances automatically when prices spike or to leave the price to the market. |
| Option 1 | Release extra allowances automatically when prices spike. |
| Option 2 | Leave the allowance price to the market without intervention. |
| Trade-off | price predictability for investors vs the strength of the scarcity signal |
| Distinct from the pilot and the other proposals | The only decision on intervening in an allowance price. Not a stringency choice: the cap is unchanged; only its short-run release is at issue. |
| Why climate, not energy | Governs the market functioning of a greenhouse-gas trading scheme. |

### climate_09 — point of obligation for fuel emissions

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 3(ae) and Chapter IVa (regulated entity for fuels) |
| Decision | A regional authority extending carbon pricing to heating and road fuels must decide whether to place the obligation on fuel suppliers or on large distributors and commercial users, with household fuel charged through ordinary bills. |
| Option 1 | Place the obligation on fuel suppliers when fuel enters the market. |
| Option 2 | Place the obligation on large distributors and commercial users, charging households through their ordinary fuel bills. |
| Trade-off | broad administrative coverage vs responsibility and visibility closer to fuel use |
| Distinct from the pilot and the other proposals | Concerns WHERE an obligation sits, not how strict it is. No pilot or proposed decision addresses the point of obligation. |
| Why climate, not energy | Design of a carbon-pricing instrument for fuel emissions; not energy-market regulation. |

### climate_10 — geographic scope of shipping emissions

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 3ga (scope for maritime transport) |
| Decision | A regional emissions-trading scheme must decide whether to cover the emissions of whole voyages to and from outside ports or only voyages within the region. |
| Option 1 | Cover whole voyages to and from outside ports. |
| Option 2 | Cover only voyages within the region. |
| Trade-off | environmental coverage vs evasion through rerouting and loss of port business |
| Distinct from the pilot and the other proposals | The only shipping decision, and a scope question rather than timing. climate_02 concerns aviation timing pending an international agreement. |
| Why climate, not energy | Extends a greenhouse-gas trading scheme to a transport sector's emissions. |

### climate_11 — coverage of small emitters

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 27a (optional exclusion of small emitters) |
| Decision | A regional emissions-trading scheme must decide whether to let small emitters leave the scheme under simpler national measures or to keep every covered installation inside it. |
| Option 1 | Let small emitters leave the scheme under simpler national measures. |
| Option 2 | Keep every covered installation inside the scheme. |
| Trade-off | proportionate administrative burden vs uniform coverage and equal treatment |
| Distinct from the pilot and the other proposals | Concerns which actors are covered. Unlike climate_04 (a government's spend on measurement), this is about operators' compliance duties. |
| Why climate, not energy | Coverage of installations under a greenhouse-gas trading scheme. |

### climate_12 — distribution rule of a climate investment fund

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 10d (Modernisation Fund) |
| Decision | A regional climate fund must decide whether to share its money mainly among lower-income areas or to direct it wherever emissions can be cut most cheaply. |
| Option 1 | Share the fund mainly among lower-income areas. |
| Option 2 | Direct the fund wherever emissions can be cut most cheaply. |
| Trade-off | fairness across areas vs cost-effectiveness of reductions |
| Distinct from the pilot and the other proposals | Concerns who RECEIVES a fund. climate_07 concerns what revenue is spent ON. climate_05 concerns binding targets per sector, not money. |
| Why climate, not energy | Allocation principle of a climate-investment fund; the object is emission reduction, not energy supply. |

### climate_13 — innovation funding portfolio

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 10a(8) (Innovation Fund) |
| Decision | A regional innovation fund for low-carbon technology must decide whether to back a few large first-of-a-kind projects or many smaller projects close to market. |
| Option 1 | Back a few large first-of-a-kind projects. |
| Option 2 | Back many smaller projects that are close to market. |
| Trade-off | transformative potential vs near-term reliability and risk spreading |
| Distinct from the pilot and the other proposals | The only decision about R&D portfolio risk. Not a spending-destination or fairness question. |
| Why climate, not energy | Funding low-carbon technology to cut industrial emissions. |

### climate_14 — linking two carbon markets

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 25 (links with other trading systems) |
| Decision | A regional emissions-trading scheme must decide whether to link with a trading scheme outside the region or to remain a separate market. |
| Option 1 | Link the scheme with a trading scheme outside the region. |
| Option 2 | Keep the scheme as a separate market. |
| Trade-off | cost-effectiveness of a larger market vs sovereignty over ambition |
| Distinct from the pilot and the other proposals | Concerns merging two markets. climate_02 is about aviation timing pending agreement; climate_15 is about importing offset credits. |
| Why climate, not energy | International linkage of greenhouse-gas trading schemes. |

### climate_15 — international offset credits

| | |
|---|---|
| Source | Directive 2003/87/EC (EU ETS), consolidated 2024-03-01 |
| Locator | CELEX 02003L0087-20240301, art. 11a (use of international credits) |
| Decision | A regional emissions-trading scheme must decide whether to let firms meet part of their obligations with credits from emission-cutting projects abroad or to require all cuts at home. |
| Option 1 | Let firms use credits from projects abroad for part of their obligations. |
| Option 2 | Require firms to make all their cuts at home. |
| Trade-off | cost-effectiveness and support abroad vs integrity and domestic transformation |
| Distinct from the pilot and the other proposals | Importing offset credits, not merging markets (climate_14) and not a domestic flexibility (climate_16). |
| Why climate, not energy | Offsetting within a greenhouse-gas compliance scheme. |

### climate_16 — borrowing against future emission allocations

| | |
|---|---|
| Source | Regulation (EU) 2018/842 (Effort Sharing), consolidated 2023-05-16 |
| Locator | CELEX 02018R0842-20230516, art. 5(1) and 5(2) (borrowing from the following year's allocation) |
| Decision | A regional government must decide whether to allow itself to borrow against future years' emission limits after a difficult year or to require each year's limit to be met. |
| Option 1 | Allow borrowing against future years' emission limits. |
| Option 2 | Require each year's emission limit to be met in that year. |
| Trade-off | intertemporal flexibility vs reliability of year-by-year progress |
| Distinct from the pilot and the other proposals | An across-YEARS flexibility. climate_05 is flexibility across SECTORS within a target; this is when, not where, cuts happen. climate_18 concerns disposing of a surplus already held (art. 5(3), (5)). |
| Why climate, not energy | Compliance flexibility under binding national greenhouse-gas limits. |

### climate_17 — counting land removals against emission limits

| | |
|---|---|
| Source | Regulation (EU) 2018/842 (Effort Sharing), consolidated 2023-05-16 |
| Locator | CELEX 02018R0842-20230516, art. 7 (use of net removals from land use) |
| Decision | A regional government must decide whether to count carbon absorbed by its forests and soils towards its limit on emissions from transport, buildings and farming. |
| Option 1 | Count carbon absorbed by forests and soils towards the emission limit. |
| Option 2 | Keep that emission limit separate from carbon absorbed by land. |
| Trade-off | rewarding land-based removals vs permanence and direct emission cuts |
| Distinct from the pilot and the other proposals | About fungibility between land and emission accounts. The land-sector decisions (climate_19-21) concern how land itself is managed or counted. |
| Why climate, not energy | Interaction of greenhouse-gas accounting between emission and removal sectors. |

### climate_18 — banking or transfer of surplus emission allocations

| | |
|---|---|
| Source | Regulation (EU) 2018/842 (Effort Sharing), consolidated 2023-05-16 |
| Locator | CELEX 02018R0842-20230516, art. 5(3), 5(5) and 5(6) (banking, transfer and use of transfer revenues) |
| Decision | A regional government whose emissions stayed below its annual limit must decide whether to sell the unused part of its emission allocation to another government or bank it for its own later years. |
| Option 1 | Sell the unused allocation to another government. |
| Option 2 | Bank the unused allocation for the region's own later years. |
| Trade-off | immediate transfer revenue and system flexibility vs a retained future compliance buffer |
| Distinct from the pilot and the other proposals | Concerns what a government does with a surplus it already holds. climate_16 is borrowing from the following year after a shortfall (art. 5(1)-(2)); climate_14 merges two trading markets; climate_15 imports project credits; climate_17 counts land removals. Neither option changes the combined limit across governments, so neither is framed as cutting more in total. |
| Why climate, not energy | Banking and transfer of national greenhouse-gas allocations; no energy market is involved. |

### climate_19 — accounting for carbon in wood products

| | |
|---|---|
| Source | Regulation (EU) 2018/841 (LULUCF), consolidated 2023-05-11 |
| Locator | CELEX 02018R0841-20230511, art. 9 (accounting for harvested wood products) |
| Decision | A regional forestry authority must decide whether to count carbon stored in long-lived wood products as storage or to count all harvested wood as an immediate emission. |
| Option 1 | Count carbon stored in long-lived wood products as storage. |
| Option 2 | Count all harvested wood as an immediate emission. |
| Trade-off | incentive for lasting wood use vs accounting caution and simplicity |
| Distinct from the pilot and the other proposals | Accounting for products after harvest. climate_21 concerns how much to harvest in the first place. |
| Why climate, not energy | Greenhouse-gas accounting in the land-use and forestry sector. |

### climate_20 — excluding natural-disturbance emissions

| | |
|---|---|
| Source | Regulation (EU) 2018/841 (LULUCF), consolidated 2023-05-11 |
| Locator | CELEX 02018R0841-20230511, art. 10 (accounting for natural disturbances) |
| Decision | A regional forestry authority must decide whether to leave emissions from wildfires and storms out of its forest carbon account or to count every emission in full. |
| Option 1 | Leave emissions from wildfires and storms out of the forest account. |
| Option 2 | Count every forest emission in full. |
| Trade-off | fairness to managers facing uncontrollable events vs completeness of the account |
| Distinct from the pilot and the other proposals | The only decision on uncontrollable disturbances. Not a harvesting or product-accounting choice. |
| Why climate, not energy | Greenhouse-gas accounting for forest emissions. |

### climate_21 — harvest level vs forest carbon sink

| | |
|---|---|
| Source | Regulation (EU) 2018/841 (LULUCF), consolidated 2023-05-11 |
| Locator | CELEX 02018R0841-20230511, art. 8 (accounting for managed forest land) |
| Decision | A regional forestry authority must decide whether to keep timber harvests at their current level or to reduce harvesting so that forests absorb more carbon. |
| Option 1 | Keep timber harvests at their current level. |
| Option 2 | Reduce harvesting so that forests absorb more carbon. |
| Trade-off | forestry livelihoods and timber supply vs carbon storage |
| Distinct from the pilot and the other proposals | Concerns how much to harvest. climate_19 concerns accounting for wood after harvest; climate_20 concerns uncontrollable losses. |
| Why climate, not energy | Land-use and forestry management for carbon removal. |

## Overlap review notes

Recorded as review notes only: they are the researcher's working judgement that each pair
is substantively distinct, **not** a curation approval. Every curation field in the bank
remains empty, and the validator's lexical warnings are left in place.

- climate_10 vs pilot climate_02: shipping scope vs aviation timing.
- climate_10 vs climate_14: geographic coverage vs linkage between markets (lexical
  similarity 0.40, still reported by the validator).
- climate_14 vs climate_15: merging two capped markets vs importing project credits.
- climate_07 vs climate_12: household compensation vs geographic allocation of
  investment funds.
- climate_16 vs climate_18: both Effort Sharing art. 5; borrowing after a shortfall
  (art. 5(1)-(2)) vs disposing of a surplus by transfer or banking (art. 5(3), (5), (6)).
- climate_16, climate_17 and pilot climate_05 are all flexibilities; they differ in what is
  flexible: years, land against emission accounts, and sectors respectively.

## Changes in the pre-curation pass

- climate_09 reworked so both points of obligation are operationally plausible: an
  upstream obligation on fuel suppliers, or a downstream obligation on large distributors
  and commercial users with household fuel charged through ordinary bills. Same carbon
  price on both sides; the trade-off is broad administrative coverage against
  responsibility and visibility closer to fuel use.
- Variant contexts that argued for one option were rewritten to state only the setting;
  the audit is in review/full_topics_v2/context_tilt_audit.md.
