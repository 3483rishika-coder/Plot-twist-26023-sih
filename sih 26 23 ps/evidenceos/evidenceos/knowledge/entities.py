"""Entity registry + resolution (PRD §11.4.1).

Seeded with the Indian coal sector (CIL and subsidiaries, SCCL, NLCIL, states,
sectors, lignite companies) — every alias links to a canonical entity.  Unknown
names found in tables (e.g. mine names) are auto-registered with provenance so
they can be reviewed/merged later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process

from ..db import Entity, EntityAlias
from ..docai.normalize import normalize_text

SEED = [
    # (canonical, type, parent, aliases, meta)
    ("India", "country", None, ["All India", "Grand Total", "Total", "India Total", "Overall", "Country", "India’s", "India's"], {"aggregate": True}),
    ("Coal India Limited", "company", "India", ["CIL", "Coal India", "Coal India Ltd", "Coal India Ltd.", "CIL Total", "Coal India Limited (CIL)"], {"aggregate": True, "hq": "Kolkata"}),
    ("Eastern Coalfields Limited", "subsidiary", "Coal India Limited", ["ECL", "Eastern Coalfields", "Eastern Coalfields Ltd"], {"state": "West Bengal", "hq": "Sanctoria"}),
    ("Bharat Coking Coal Limited", "subsidiary", "Coal India Limited", ["BCCL", "Bharat Coking Coal", "Bharat Coking Coal Ltd"], {"state": "Jharkhand", "hq": "Dhanbad"}),
    ("Central Coalfields Limited", "subsidiary", "Coal India Limited", ["CCL", "Central Coalfields", "Central Coalfields Ltd"], {"state": "Jharkhand", "hq": "Ranchi"}),
    ("Northern Coalfields Limited", "subsidiary", "Coal India Limited", ["NCL", "Northern Coalfields", "Northern Coalfields Ltd"], {"state": "Madhya Pradesh", "hq": "Singrauli"}),
    ("Western Coalfields Limited", "subsidiary", "Coal India Limited", ["WCL", "Western Coalfields", "Western Coalfields Ltd"], {"state": "Maharashtra", "hq": "Nagpur"}),
    ("South Eastern Coalfields Limited", "subsidiary", "Coal India Limited", ["SECL", "South Eastern Coalfields", "South Eastern Coalfields Ltd"], {"state": "Chhattisgarh", "hq": "Bilaspur"}),
    ("Mahanadi Coalfields Limited", "subsidiary", "Coal India Limited", ["MCL", "Mahanadi Coalfields", "Mahanadi Coalfields Ltd"], {"state": "Odisha", "hq": "Sambalpur"}),
    ("North Eastern Coalfields", "subsidiary", "Coal India Limited", ["NEC", "N.E.C.", "North Eastern Coalfields (NEC)"], {"state": "Assam", "hq": "Margherita"}),
    ("Central Mine Planning and Design Institute", "subsidiary", "Coal India Limited", ["CMPDI", "CMPDIL", "CMPDI Ltd", "Central Mine Planning & Design Institute"], {"hq": "Ranchi"}),
    ("Singareni Collieries Company Limited", "company", "India", ["SCCL", "Singareni", "Singareni Collieries", "The Singareni Collieries Company Limited"], {"state": "Telangana"}),
    ("NLC India Limited", "company", "India", ["NLCIL", "NLC", "NLC India", "Neyveli Lignite Corporation", "NLC India Ltd"], {"state": "Tamil Nadu"}),
    ("Captive and Others", "company", "India", ["Captives/Others", "Captive/Others", "Captive & Others", "Captives & Others", "Captive and others", "Others (Captive)", "Captive & Commercial", "Captive/Commercial", "Captive & Others (incl. commercial)"], {"residual": True}),
    ("Captive Mines", "company", "India", ["Captive", "Captives", "Captive Blocks", "Captive Coal Mines", "Captive Mining"], {"residual": True}),
    ("Tata Steel Limited", "company", None, ["Tata Steel", "TISCO"], {}),
    # sectors (dispatch consumers)
    ("Power Utilities", "sector", None, ["Power", "Power (Utilities)", "Power (Utility)", "Power ( Utilities)", "Power Utility", "Power-Utility", "Power (Util.)", "Dispatch to Power", "Power Sector", "Power Utilities (incl. IPP)"], {}),
    ("Commercial Coal Mines", "company", "India", ["Commercial", "Commercial Mines", "Commercial Miners", "Commercial Coal Blocks", "Commercial Mining"], {"residual": True}),
    ("Other Producers", "company", "India", ["Other", "Others (producers)", "Other Companies", "Other Producers"], {"residual": True}),
    ("Captive Power Plants", "sector", None, ["CPP", "Captive Power", "Captive Power Plant", "Power (CPP)", "Power (Captive)", "Captive Power (CPP)", "Power-Captive", "CPP (Captive Power)"], {}),
    ("Steel", "sector", None, ["Steel Plants", "Steel/Coke", "Steel Sector"], {}),
    ("Cement", "sector", None, ["Cement Plants", "Cement Sector"], {}),
    ("Sponge Iron", "sector", None, ["Sponge Iron Plants", "Sponge Iron/CDI", "Sponge Iron / CDI", "Sponge Iron & CDI", "Sponge"], {}),
    ("Fertilizer", "sector", None, ["Fertilizers", "Fertiliser"], {}),
    ("Others (sector)", "sector", None, ["Others", "Other Sectors", "Misc", "Miscellaneous"], {}),
    ("Non-Regulated Sector", "sector", None, ["NRS", "Dispatch to NRS", "Non Regulated Sector", "Non-regulated"], {"aggregate": True}),
    # lignite producers
    ("Gujarat Mineral Development Corporation", "company", None, ["GMDCL", "GMDC"], {"state": "Gujarat"}),
    ("Gujarat Industries Power Company", "company", None, ["GIPCL"], {"state": "Gujarat"}),
    ("Rajasthan State Mines and Minerals", "company", None, ["RSMML", "RSMM"], {"state": "Rajasthan"}),
    ("GHCL Limited", "company", None, ["GHCL"], {"state": "Gujarat"}),
    ("VS Lignite Power", "company", None, ["VSLPPL", "VS Lignite"], {"state": "Rajasthan"}),
    ("Barmer Lignite Mining Company", "company", None, ["BLMCL"], {"state": "Rajasthan"}),
    ("Gujarat Power Corporation", "company", None, ["GPCL"], {"state": "Gujarat"}),
    # coal type / category rows
    ("Coking Coal", "coal_type", None, ["Coking", "Coking Coal (Total)"], {}),
    ("Non-Coking Coal", "coal_type", None, ["Non-coking", "Non Coking", "Non-Coking", "Non coking coal"], {}),
    ("Lignite", "coal_type", None, ["Lignite (Total)"], {}),
    ("Gondwana Coalfields", "coalfield_group", None, ["Gondwana", "Gondwana Coal", "Gondwana coalfields"], {}),
    ("Tertiary Coalfields", "coalfield_group", None, ["Tertiary", "Tertiary Coal", "Tertiary coalfields"], {}),
]

STATES = ["Jharkhand", "Odisha", "Chhattisgarh", "West Bengal", "Madhya Pradesh", "Telangana", "Maharashtra",
          "Andhra Pradesh", "Bihar", "Uttar Pradesh", "Meghalaya", "Assam", "Nagaland", "Arunachal Pradesh", "Sikkim",
          "Tamil Nadu", "Rajasthan", "Gujarat", "Puducherry", "Jammu & Kashmir", "Kerala", "Karnataka", "Punjab", "Haryana"]
STATE_ALIASES = {"Odisha": ["Orissa"], "Tamil Nadu": ["Tamilnadu", "TN"], "Puducherry": ["Pondicherry", "UT of Puducherry"],
                 "Jammu & Kashmir": ["Jammu and Kashmir", "J&K", "Jammu & Kashmir"], "West Bengal": ["W.B.", "WB"],
                 "Madhya Pradesh": ["M.P.", "MP"], "Uttar Pradesh": ["U.P.", "UP"], "Chhattisgarh": ["Chattisgarh", "CG"],
                 "Andhra Pradesh": ["A.P.", "AP"], "Arunachal Pradesh": ["Arunachal"]}

MINE_SEED = [
    ("Rajmahal OCP", "Eastern Coalfields Limited", ["Rajmahal OC", "Rajmahal Project", "Rajmahal Open Cast Project", "Rajmahal Opencast", "Rajmahal"]),
    ("Sonepur Bazari OCP", "Eastern Coalfields Limited", ["Sonepur Bazari OC", "Sonepur Bazari", "Sonpur Bazari"]),
    ("Gevra OCP", "South Eastern Coalfields Limited", ["Gevra OC", "Gevra", "Gevra Project", "Gevra Open Cast"]),
    ("Kusmunda OCP", "South Eastern Coalfields Limited", ["Kusumunda OC", "Kusmunda", "Kusumunda"]),
    ("Dipka OCP", "South Eastern Coalfields Limited", ["Dipka OC", "Dipka"]),
    ("Jayant OCP", "Northern Coalfields Limited", ["Jayant OC Mine", "Jayant", "Jayant OC"]),
    ("Dudhichua OCP", "Northern Coalfields Limited", ["Dudhichua OC Mine", "Dudhichua", "Dudhichua OC"]),
    ("Nigahi OCP", "Northern Coalfields Limited", ["Nigahi OC Mine", "Nigahi"]),
    ("Amlohri OCP", "Northern Coalfields Limited", ["Amlohri OC Mine", "Amlohri"]),
    ("Khadia OCP", "Northern Coalfields Limited", ["Khadia OC Mine", "Khadia"]),
    ("Bina OCP", "Northern Coalfields Limited", ["Bina OC Mine", "Bina"]),
    ("Block B OCP", "Northern Coalfields Limited", ["Block B OC Mine", "Block-B"]),
    ("Krishnashila OCP", "Northern Coalfields Limited", ["Krishnashila OC Mine", "Krishnashila"]),
    ("Amrapali OCP", "Central Coalfields Limited", ["Amrapali OC", "Amrapali"]),
    ("Magadh OCP", "Central Coalfields Limited", ["Magadh OC", "Magadh"]),
    ("Ashoka OCP", "Central Coalfields Limited", ["Ashoka OC", "Ashoka"]),
    ("Bhubaneswari OCP", "Mahanadi Coalfields Limited", ["Bhubaneswari OC Mine", "Bhubaneswari"]),
    ("Kulda OCP", "Mahanadi Coalfields Limited", ["Kulda OC Mine", "Kulda"]),
    ("Lingaraj OCP", "Mahanadi Coalfields Limited", ["Lingaraj OC Mine", "Lingaraj"]),
    ("Lajkura-Belpahar-Lakhanpur Integrated Project", "Mahanadi Coalfields Limited", ["LBL Integrated Project", "LBL"]),
    ("Ananta OCP", "Mahanadi Coalfields Limited", ["Ananta OC Mine", "Ananta"]),
    ("Talabira II & III OCP", "NLC India Limited", ["Talabira", "Talabira Mine", "Talabira II & III"]),
]

SUFFIX_RE = re.compile(r"\b(ocp|oc|ocm|ug|opencast|open cast|project|mine|mines|colliery|integrated project|oc mine|oc plant)\b", re.I)


def norm_key(s: str) -> str:
    return normalize_text(s.replace("&", " and "))


def mine_key(s: str) -> str:
    return " ".join(SUFFIX_RE.sub(" ", norm_key(s)).split())


def seed_entities(session) -> None:
    if session.query(Entity).count() > 0:
        return
    by_name: dict[str, Entity] = {}
    for name, etype, parent, aliases, meta in SEED:
        e = Entity(canonical_name=name, entity_type=etype, meta=meta, is_aggregate=bool(meta.get("aggregate")))
        session.add(e); session.flush()
        by_name[name] = e
        for a in {name, *aliases}:
            session.add(EntityAlias(entity_id=e.entity_id, alias=a, alias_norm=norm_key(a), source="seed", confidence=1.0))
    for name, etype, parent, aliases, meta in SEED:
        if parent and parent in by_name:
            by_name[name].parent_entity_id = by_name[parent].entity_id
    for st in STATES:
        e = Entity(canonical_name=st, entity_type="state", meta={})
        session.add(e); session.flush()
        for a in {st, *STATE_ALIASES.get(st, [])}:
            session.add(EntityAlias(entity_id=e.entity_id, alias=a, alias_norm=norm_key(a), source="seed", confidence=1.0))
    for name, parent, aliases in MINE_SEED:
        e = Entity(canonical_name=name, entity_type="mine", parent_entity_id=by_name[parent].entity_id if parent in by_name else None,
                   meta={"subsidiary": parent})
        session.add(e); session.flush()
        for a in {name, *aliases}:
            session.add(EntityAlias(entity_id=e.entity_id, alias=a, alias_norm=norm_key(a), source="seed", confidence=1.0))
            session.add(EntityAlias(entity_id=e.entity_id, alias=a, alias_norm=mine_key(a), source="seed-stem", confidence=0.95))
    session.flush()


JUNK_LABEL_RE = re.compile(r"^(coal|lignite|overburden|ob|in|of|the|and|total|others?|misc|mt|nos?|road|rail|mgr|belt|ropeway|conveyor|"
                           r"production|despatch|dispatch|offtake|average|growth|target|actual|year|month|quarter|state|states|company|companies|"
                           r"mines?|grade|sector|sectors|type|unit|units|all india|india total|sub[- ]?total|grand total|"
                           r"power|steel|cement|item|product|particulars|unit|units|value|quantity|qty|remarks?|figures?|source|note|"
                           r"raw coal|washed coal|coking|non[- ]?coking|opencast|underground|oc|ug|mixed|male|female|yes|no|nil|na|n\.a\.)$", re.I)


def _creatable(raw: str) -> bool:
    """Only labels that look like proper names may become new entities (no units, stray tokens, generic words)."""
    t = (raw or "").strip()
    alpha = re.sub(r"[^A-Za-z]", "", t)
    if len(alpha) < 3 or len(t) > 60:
        return False
    if t.count("(") != t.count(")") or t[0] in "-–—*/(),." or t.endswith((")", "-", "/")) and t.count("(") == 0:
        return False
    if JUNK_LABEL_RE.match(t.strip(" .:-")):
        return False
    if re.fullmatch(r"[\d.,%\s]+[A-Za-z)]*", t):
        return False
    return True


@dataclass
class Resolution:
    entity: Optional[Entity]
    confidence: float
    method: str
    candidates: list[tuple[str, float]]


class Resolver:
    """In-memory alias index over the entity table; refreshed when new entities are created."""

    def __init__(self, session):
        self.session = session
        self.reload()

    def reload(self):
        self.entities = {e.entity_id: e for e in self.session.query(Entity).all()}
        self.alias_map: dict[str, str] = {}
        self.alias_list: list[tuple[str, str]] = []
        for a in self.session.query(EntityAlias).all():
            if a.alias_norm:
                self.alias_map.setdefault(a.alias_norm, a.entity_id)
                self.alias_list.append((a.alias_norm, a.entity_id))
        self.alias_keys = [k for k, _ in self.alias_list]

    def aliases_of(self, entity) -> list[str]:
        return [a for a, eid in self.alias_list if eid == entity.entity_id]

    def resolve(self, raw: str, expected_type: Optional[str] = None, parent_hint: Optional[Entity] = None,
                create: bool = True, source: str = "auto") -> Resolution:
        raw = (raw or "").strip()
        if not raw:
            return Resolution(None, 0.0, "empty", [])
        key = norm_key(raw)
        cands: list[tuple[str, float]] = []
        # exact
        eid = self.alias_map.get(key)
        if eid is None and expected_type in (None, "mine"):
            eid = self.alias_map.get(mine_key(raw))
        if eid is not None:
            e = self.entities[eid]
            if expected_type is None or e.entity_type == expected_type or expected_type in ("company", "subsidiary") and e.entity_type in ("company", "subsidiary", "country"):
                return Resolution(e, 1.0, "exact", [(e.canonical_name, 100.0)])
            # exact alias of a different type (e.g. 'ECL' in a mines column) – still the right entity
            return Resolution(e, 0.9, "exact-type-mismatch", [(e.canonical_name, 100.0)])
        # fuzzy
        pool_ids = [eid for _, eid in self.alias_list]
        if self.alias_keys:
            matches = process.extract(key, self.alias_keys, scorer=fuzz.WRatio, limit=8, score_cutoff=75)
            for m_key, score, idx in matches:
                e = self.entities[pool_ids[idx]]
                if expected_type and e.entity_type != expected_type and not (expected_type in ("company", "subsidiary") and e.entity_type in ("company", "subsidiary")):
                    continue
                if parent_hint is not None and e.entity_type == "mine" and e.parent_entity_id not in (None, parent_hint.entity_id):
                    score -= 15
                cands.append((e.canonical_name, float(score)))
        cands.sort(key=lambda c: -c[1])
        if cands and cands[0][1] >= 92:
            e = next(x for x in self.entities.values() if x.canonical_name == cands[0][0])
            return Resolution(e, round(cands[0][1] / 100, 3), "fuzzy", cands[:5])
        if not create or not _creatable(raw):
            return Resolution(None, round(cands[0][1] / 100, 3) if cands else 0.0, "unresolved", cands[:5])
        # create new entity with provenance
        etype = expected_type or "organization"
        e = Entity(canonical_name=_titleize(raw), entity_type=etype,
                   parent_entity_id=parent_hint.entity_id if parent_hint is not None else None,
                   meta={"auto_created": True, "source": source, **({"subsidiary": parent_hint.canonical_name} if parent_hint is not None else {})})
        self.session.add(e); self.session.flush()
        self.session.add(EntityAlias(entity_id=e.entity_id, alias=raw, alias_norm=key, source=source, confidence=0.8))
        if etype == "mine":
            self.session.add(EntityAlias(entity_id=e.entity_id, alias=raw, alias_norm=mine_key(raw), source=source + "-stem", confidence=0.75))
        self.session.flush()
        self.entities[e.entity_id] = e
        self.alias_map[key] = e.entity_id
        self.alias_list.append((key, e.entity_id)); self.alias_keys.append(key)
        if etype == "mine":
            mk = mine_key(raw)
            self.alias_map.setdefault(mk, e.entity_id); self.alias_list.append((mk, e.entity_id)); self.alias_keys.append(mk)
        return Resolution(e, 0.8 if not cands else round(max(0.6, cands[0][1] / 100), 3), "created", cands[:5])


def _titleize(s: str) -> str:
    s = " ".join(s.split())
    if s.isupper() and len(s) > 5:
        return " ".join(w if w in ("OC", "OCP", "OCM", "UG", "LBL", "II", "III", "IV") else w.title() for w in s.split())
    return s
