"""
NPI action mapping for the outbreak policy benchmark.

Translates LLM policy decisions into Covasim simulation state changes.
"""

import covasim as cv
import numpy as np

DEFAULT_POLICY = {
    "schools": "open",
    "workplaces": "open",
    "masks": False,
    "mass_testing": False,
    "contact_tracing": False,
    "gathering_limits": "none",
    "stay_at_home": False,
}

AGE_TARGETED_FIELDS = {"shielding_elderly"}

DEFAULT_POLICY_AGE_TARGETED = {
    **DEFAULT_POLICY,
    "shielding_elderly": False,
}

SCHOOL_OPTIONS = {"full", "partial", "open"}
WORKPLACE_OPTIONS = {"full", "partial", "open"}
GATHERING_OPTIONS = {"ban_large", "ban_all", "none"}

_CLOSURE_FRACTIONS = {"full": 0.0, "partial": 0.5, "open": 1.0}
_GATHERING_BETA = {"ban_large": 0.5, "ban_all": 0.2, "none": 1.0}
_MASK_BETA = 0.7
_TESTING_SYMP_PROB = 0.3
_TESTING_BASELINE_PROB = 0.05
_TRACE_PROBS = {"h": 1.0, "s": 0.8, "w": 0.5, "c": 0.05}
_TRACE_TIME = {"h": 1, "s": 2, "w": 2, "c": 5}
_STAY_AT_HOME_FRACTION = 0.3
_SHIELDING_CONTACT_FACTOR = 0.3  # elderly keep 30% of contacts
_SHIELDING_AGE = 60

FATIGUE_RATE = 0.05  # per week; half-life ~14 weeks (Eikenberry 2020)
FATIGUE_FLOOR = 0.5  # compliance never drops below 50%

_FATIGUE_KEYS = ["schools", "workplaces", "masks", "mass_testing",
                 "contact_tracing", "gathering_limits", "stay_at_home",
                 "shielding_elderly"]


def _is_active(key, value):
    """Whether an NPI setting differs from its default (no-intervention) state."""
    return value != DEFAULT_POLICY_AGE_TARGETED[key]


def _compliance(weeks_active):
    """Compliance multiplier given consecutive weeks of use."""
    return FATIGUE_FLOOR + (1.0 - FATIGUE_FLOOR) * np.exp(-FATIGUE_RATE * weeks_active)


def validate_policy(policy, age_targeting=False):
    """Validate a policy dict. Returns normalized copy with defaults filled in."""
    base = DEFAULT_POLICY_AGE_TARGETED if age_targeting else DEFAULT_POLICY
    out = dict(base)
    out.update(policy)
    assert out["schools"] in SCHOOL_OPTIONS
    assert out["workplaces"] in WORKPLACE_OPTIONS
    assert out["gathering_limits"] in GATHERING_OPTIONS
    assert isinstance(out["masks"], bool)
    assert isinstance(out["mass_testing"], bool)
    assert isinstance(out["contact_tracing"], bool)
    assert isinstance(out["stay_at_home"], bool)
    if age_targeting:
        assert isinstance(out["shielding_elderly"], bool)
    else:
        out.pop("shielding_elderly", None)
    return out


class NPIManager:
    """Manages NPI application on a Covasim sim across decision cycles."""

    def __init__(self, sim, age_targeting=False):
        """Initialize from a sim that has been initialized (sim.initialize() called)."""
        self._age_targeting = age_targeting
        self._baseline_betas = dict(sim["beta_layer"])
        self._baseline_edges = {}
        self._stored_edges = {}
        for lkey in sim.people.layer_keys():
            layer = sim.people.contacts[lkey]
            self._baseline_edges[lkey] = len(layer)
            self._stored_edges[lkey] = {
                "p1": np.array([], dtype=layer["p1"].dtype),
                "p2": np.array([], dtype=layer["p2"].dtype),
                "beta": np.array([], dtype=layer["beta"].dtype),
            }

        self._test_prob = None
        self._contact_tracing = None
        base = DEFAULT_POLICY_AGE_TARGETED if age_targeting else DEFAULT_POLICY
        self._current_policy = dict(base)
        self._active_weeks = {k: 0 for k in _FATIGUE_KEYS}

        # For elderly shielding: store per-edge baseline betas and elderly masks
        self._shielding_active = False
        if age_targeting:
            elderly = sim.people.age >= _SHIELDING_AGE
            self._elderly_edge_masks = {}
            for lkey in sim.people.layer_keys():
                layer = sim.people.contacts[lkey]
                p1_old = elderly[layer["p1"]]
                p2_old = elderly[layer["p2"]]
                self._elderly_edge_masks[lkey] = p1_old | p2_old

    def apply(self, sim, policy):
        """Apply an NPI policy to the sim. Call before advancing to the next week."""
        policy = validate_policy(policy, age_targeting=self._age_targeting)
        self._update_fatigue(policy)
        self._apply_contacts(sim, policy)
        self._apply_betas(sim, policy)
        self._apply_testing(sim, policy)
        self._apply_tracing(sim, policy)
        if self._age_targeting:
            self._apply_shielding(sim, policy)
        self._current_policy = policy

    def _update_fatigue(self, policy):
        for key in _FATIGUE_KEYS:
            if key not in policy:
                continue
            if _is_active(key, policy[key]):
                self._active_weeks[key] += 1
            else:
                self._active_weeks[key] = 0

    def compliance(self, key):
        """Current compliance multiplier for an NPI (1.0 = full, FATIGUE_FLOOR = minimum)."""
        return _compliance(self._active_weeks[key])

    @property
    def current_policy(self):
        return dict(self._current_policy)

    def _desired_contact_fractions(self, policy):
        fracs = {}
        fracs["h"] = 1.0

        c_s = self.compliance("schools")
        target_s = _CLOSURE_FRACTIONS[policy["schools"]]
        fracs["s"] = 1.0 - c_s * (1.0 - target_s)

        c_w = self.compliance("workplaces")
        target_w = _CLOSURE_FRACTIONS[policy["workplaces"]]
        w_closure = 1.0 - c_w * (1.0 - target_w)

        c_sah = self.compliance("stay_at_home")
        if policy["stay_at_home"]:
            w_stay = 1.0 - c_sah * (1.0 - _STAY_AT_HOME_FRACTION)
        else:
            w_stay = 1.0
        fracs["w"] = min(w_closure, w_stay)

        fracs["c"] = 1.0 - c_sah * (1.0 - _STAY_AT_HOME_FRACTION) if policy["stay_at_home"] else 1.0
        return fracs

    def _apply_contacts(self, sim, policy):
        fracs = self._desired_contact_fractions(policy)
        for lkey in sim.people.layer_keys():
            desired = fracs.get(lkey, 1.0)
            self._set_layer_fraction(sim, lkey, desired)

    def _set_layer_fraction(self, sim, lkey, desired_frac):
        layer = sim.people.contacts[lkey]
        stored = self._stored_edges[lkey]
        n_sim = len(layer)
        n_stored = len(stored["p1"])
        n_total = n_sim + n_stored

        if n_total == 0:
            return

        current_frac = n_sim / n_total
        diff = current_frac - desired_frac

        if abs(diff) < 0.01:
            return

        if diff > 0:
            # Move edges from sim to storage
            n_to_move = int(n_total * diff)
            n_to_move = min(n_to_move, n_sim)
            if n_to_move == 0:
                return
            inds = np.random.choice(n_sim, size=n_to_move, replace=False)
            for key in ("p1", "p2", "beta"):
                moved = layer[key][inds]
                stored[key] = np.concatenate([stored[key], moved])
            keep = np.ones(n_sim, dtype=bool)
            keep[inds] = False
            for key in ("p1", "p2", "beta"):
                layer[key] = layer[key][keep]
        else:
            # Move edges from storage back to sim
            n_to_move = int(n_total * (-diff))
            n_to_move = min(n_to_move, n_stored)
            if n_to_move == 0:
                return
            inds = np.random.choice(n_stored, size=n_to_move, replace=False)
            for key in ("p1", "p2", "beta"):
                moved = stored[key][inds]
                layer[key] = np.concatenate([layer[key], moved])
            keep = np.ones(n_stored, dtype=bool)
            keep[inds] = False
            for key in ("p1", "p2", "beta"):
                stored[key] = stored[key][keep]

    def _apply_betas(self, sim, policy):
        c_mask = self.compliance("masks")
        c_gath = self.compliance("gathering_limits")
        for lkey in sim.people.layer_keys():
            mult = 1.0
            if policy["masks"] and lkey in ("s", "w", "c"):
                mult *= 1.0 - c_mask * (1.0 - _MASK_BETA)
            if lkey == "c":
                g_beta = _GATHERING_BETA[policy["gathering_limits"]]
                mult *= 1.0 - c_gath * (1.0 - g_beta)
            sim["beta_layer"][lkey] = self._baseline_betas[lkey] * mult

    def _apply_testing(self, sim, policy):
        if policy["mass_testing"]:
            c = self.compliance("mass_testing")
            if self._test_prob is None:
                self._test_prob = cv.test_prob(
                    symp_prob=_TESTING_SYMP_PROB * c,
                    asymp_prob=0.01 * c,
                    start_day=0,
                    test_delay=1,
                )
                self._test_prob.initialize(sim)
                sim["interventions"].append(self._test_prob)
            else:
                self._test_prob.symp_prob = _TESTING_SYMP_PROB * c
                self._test_prob.asymp_prob = 0.01 * c
        else:
            if self._test_prob is not None:
                if self._test_prob in sim["interventions"]:
                    sim["interventions"].remove(self._test_prob)
                self._test_prob = None

    def _apply_tracing(self, sim, policy):
        if policy["contact_tracing"]:
            c = self.compliance("contact_tracing")
            if self._contact_tracing is None:
                self._contact_tracing = cv.contact_tracing(
                    trace_probs={k: v * c for k, v in _TRACE_PROBS.items()},
                    trace_time=_TRACE_TIME,
                    start_day=0,
                )
                self._contact_tracing.initialize(sim)
                sim["interventions"].append(self._contact_tracing)
            else:
                for k, v in _TRACE_PROBS.items():
                    self._contact_tracing.trace_probs[k] = v * c
        else:
            if self._contact_tracing is not None:
                if self._contact_tracing in sim["interventions"]:
                    sim["interventions"].remove(self._contact_tracing)
                self._contact_tracing = None

    def _apply_shielding(self, sim, policy):
        """Reduce contacts for people aged 60+ across all layers."""
        if policy.get("shielding_elderly", False):
            c = self.compliance("shielding_elderly")
            target = 1.0 - c * (1.0 - _SHIELDING_CONTACT_FACTOR)
            self._shielding_active = True
        elif self._shielding_active:
            target = 1.0
            self._shielding_active = False
        else:
            return

        elderly = sim.people.age >= _SHIELDING_AGE
        for lkey in sim.people.layer_keys():
            layer = sim.people.contacts[lkey]
            stored = self._stored_edges[lkey]

            # Rebuild elderly mask for current edges (edges may have been
            # added/removed by other NPIs since init)
            n_sim = len(layer["p1"])
            n_stored = len(stored["p1"])
            if n_sim == 0 and n_stored == 0:
                continue

            sim_elderly = np.zeros(n_sim, dtype=bool)
            if n_sim > 0:
                sim_elderly = elderly[layer["p1"]] | elderly[layer["p2"]]
            stored_elderly = np.zeros(n_stored, dtype=bool)
            if n_stored > 0:
                stored_elderly = elderly[stored["p1"]] | elderly[stored["p2"]]

            n_elderly_total = int(np.sum(sim_elderly)) + int(np.sum(stored_elderly))
            if n_elderly_total == 0:
                continue

            n_elderly_sim = int(np.sum(sim_elderly))
            current_frac = n_elderly_sim / n_elderly_total
            diff = current_frac - target

            if abs(diff) < 0.01:
                continue

            if diff > 0:
                n_to_move = min(int(n_elderly_total * diff), n_elderly_sim)
                if n_to_move == 0:
                    continue
                elderly_inds = np.where(sim_elderly)[0]
                chosen = np.random.choice(elderly_inds, size=n_to_move, replace=False)
                for key in ("p1", "p2", "beta"):
                    stored[key] = np.concatenate([stored[key], layer[key][chosen]])
                keep = np.ones(n_sim, dtype=bool)
                keep[chosen] = False
                for key in ("p1", "p2", "beta"):
                    layer[key] = layer[key][keep]
            else:
                n_to_move = min(int(n_elderly_total * (-diff)), int(np.sum(stored_elderly)))
                if n_to_move == 0:
                    continue
                elderly_stored_inds = np.where(stored_elderly)[0]
                chosen = np.random.choice(elderly_stored_inds, size=n_to_move, replace=False)
                for key in ("p1", "p2", "beta"):
                    layer[key] = np.concatenate([layer[key], stored[key][chosen]])
                keep = np.ones(n_stored, dtype=bool)
                keep[chosen] = False
                for key in ("p1", "p2", "beta"):
                    stored[key] = stored[key][keep]
