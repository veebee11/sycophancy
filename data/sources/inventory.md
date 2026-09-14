# Source inventory

An original, category-level description of the downloaded reference material,
written to find topic ideas. It contains no source wording beyond identifiers
(EU document numbers, table and column names). The files themselves are listed,
with checksums, in `downloads.yaml`.

**What was read.** POLIANNA: only the JSON files — each article's
`Policy_Info.json`, `Curated_Annotations.json` and `Tokens.json`, plus
`01_policy_info/Coding_Scheme.json`. Its pickle files were never extracted; its
plain-text files and CSV tables were not opened. JRC: the two pinned CSV tables,
profiled by column; free-text description fields were used only by the overlap
screen, not read for topic ideas.

---

## POLIANNA (`polianna`) — climate and energy

412 articles from 18 EU climate-mitigation and energy laws, each annotated with
policy-design labels. Laws are identified below by their EU document (CELEX)
number, with their general subject in our words.

| CELEX | Articles | General subject |
|---|---|---|
| 32004L0008 | 18 | promoting combined heat-and-power generation for efficiency and supply security |
| 32006L0032 | 20 | improving how efficiently end users consume energy, through targets and market measures |
| 32006L0066 | 30 | restricting hazardous content in batteries and organising their collection and recycling |
| 32008R1099 | 12 | a shared framework for producing comparable energy statistics |
| 32009L0028 | 29 | promoting renewable energy through national shares of consumption |
| 32009R0079 | 16 | approval requirements for hydrogen-powered vehicles and their components |
| 32009R0397 | 2 | amendment allowing regional funds to support efficiency and renewables in existing housing |
| 32012L0027 | 30 | a common framework of efficiency measures, including supplier obligation schemes (art. 7) |
| 32013R0174 | 2 | amendment to an efficiency-labelling programme for office equipment |
| 32013R0525 | 29 | monitoring and reporting greenhouse-gas emissions, including annual inventories (art. 7) |
| 32014L0094 | 13 | deploying infrastructure for alternative transport fuels, including electric charging (art. 4) |
| 32014R0256 | 13 | notifying data on energy-infrastructure investment projects |
| 32014R0421 | 2 | amendment to emissions trading, deferring aviation coverage pending a wider international measure |
| 32018L0844 | 5 | amendments to building energy-performance rules |
| 32018L2001 | 39 | renewable energy with an overall binding target; permitting (art. 16); energy communities (art. 22) |
| 32018R1999 | 59 | governance of national energy and climate plans, objectives (art. 4) and reporting |
| 32019L0944 | 74 | electricity-market rules; dynamic price contracts (art. 11); storage ownership by network operators (arts. 36, 54) |
| 32019R0631 | 19 | emission-performance standards for new cars and vans |

**Policy-design categories** (from the coding scheme): objectives (qualitative
intentions and quantified targets); actors (addressees and authorities);
resources (revenues, spending); timing (compliance, duration, monitoring);
compliance (monitoring, sanctioning); reversibility; technology and energy
specificity; and ten instrument types — framework policy, regulation,
subsidies and incentives, tax incentives, tradable permits, public investment,
research and development, voluntary agreements, education and outreach, and
unspecified.

**Most frequent annotations:** addressees, monitoring provisions, low-carbon
technology and energy references, and regulatory instruments. Quantified targets
are comparatively rare.

**Useful for topics:** who bears obligations (suppliers, network operators,
building owners, vehicle makers); regulate versus incentivise; speed of
deployment versus local participation; measurement versus direct action;
binding versus flexible targets; market competition versus central provision.

**Gap:** mitigation and energy only — no climate-adaptation material.

---

## JRC snapshot (`genai4pa_jrc_2025`) — technology governance

Two tables about generative AI in European public administrations.

### Annex I — 33 guidelines, policies and rules

- **Document type:** guideline 23, rule or regulation 5, policy 3, procedure 2.
- **Issuing level:** national 19, local 7, cross-country 4, regional 3; mostly central government.
- **Audience:** public organisations 14, internal staff 12, public and private organisations 7.
- **Principles flagged** (a document can flag several): human agency and oversight 29, transparency 29, accountability 24, privacy and data governance 23, fairness and non-discrimination 19, intellectual property 19, societal and environmental well-being 9, technical step-by-step guidance 9, technical robustness 7, public–private collaboration 3.

### Annex II — 61 use cases

- **Process type:** public services and engagement 27, internal management 20, analysis and monitoring 14.
- **Application type:** mainly service personalisation, internal support, policy innovation and engagement management.
- **Interaction:** government-to-government 24, government-to-citizen 24, government-to-business 13.
- **Status:** pilot 23, implemented 17, in development 17, planned 4.
- **Government function:** mostly general public services, then public order and safety, economic affairs and others.
- **Intended benefits flagged:** improved services, new channels, efficiency, and — less often — transparency, participation and public control.

**Useful for topics:** each Annex I principle is a governance question in its
own right — how much human oversight, how much disclosure, what personal data
may be used, who is accountable, and how to procure. Annex II shows where those
questions arise: citizen-facing services, new channels and internal automation.

**Gap:** European public sector only; describes practice and guidance, not
outcomes.
