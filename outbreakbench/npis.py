"""
NPI action mapping for the outbreak policy benchmark.

Translates LLM policy decisions into Covasim simulation state changes.
"""

import numpy as np
import covasim as cv


DEFAULT_POLICY = {
    "schools": "open",
    "workplaces": "open",
    "masks": False,
    "mass_testing": False,
    "contact_tracing": False,
    "gathering_limits": "none",
    "stay_at_home": False,
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


def validate_policy(policy):
    """Validate a policy dict. Returns normalized copy with defaults filled in."""
    out = dict(DEFAULT_POLICY)
    out.update(policy)
    assert out["schools"] in SCHOOL_OPTIONS
    assert out["workplaces"] in WORKPLACE_OPTIONS
    assert out["gathering_limits"] in GATHERING_OPTIONS
    assert isinstance(out["masks"], bool)
    assert isinstance(out["mass_testing"], bool)
    assert isinstance(out["contact_tracing"], bool)
    assert isinstance(out["stay_at_home"], bool)
    return out


class NPIManager:
    """Manages NPI application on a Covasim sim across decision cycles."""

    def __init__(self, sim):
        """Initialize from a sim that has been initialized (sim.initialize() called)."""
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
        self._current_policy = dict(DEFAULT_POLICY)

    def apply(self, sim, policy):
        """Apply an NPI policy to the sim. Call before advancing to the next week."""
        policy = validate_policy(policy)
        self._apply_contacts(sim, policy)
        self._apply_betas(sim, policy)
        self._apply_testing(sim, policy)
        self._apply_tracing(sim, policy)
        self._current_policy = policy

    @property
    def current_policy(self):
        return dict(self._current_policy)

    def _desired_contact_fractions(self, policy):
        fracs = {}
        fracs["h"] = 1.0
        fracs["s"] = _CLOSURE_FRACTIONS[policy["schools"]]
        w_closure = _CLOSURE_FRACTIONS[policy["workplaces"]]
        w_stay = _STAY_AT_HOME_FRACTION if policy["stay_at_home"] else 1.0
        fracs["w"] = min(w_closure, w_stay)
        c_stay = _STAY_AT_HOME_FRACTION if policy["stay_at_home"] else 1.0
        fracs["c"] = c_stay
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
        for lkey in sim.people.layer_keys():
            mult = 1.0
            if policy["masks"] and lkey in ("s", "w", "c"):
                mult *= _MASK_BETA
            if lkey == "c":
                mult *= _GATHERING_BETA[policy["gathering_limits"]]
            sim["beta_layer"][lkey] = self._baseline_betas[lkey] * mult

    def _apply_testing(self, sim, policy):
        if policy["mass_testing"]:
            if self._test_prob is None:
                self._test_prob = cv.test_prob(
                    symp_prob=_TESTING_SYMP_PROB,
                    asymp_prob=0.01,
                    start_day=0,
                    test_delay=1,
                )
                self._test_prob.initialize(sim)
                sim["interventions"].append(self._test_prob)
        else:
            if self._test_prob is not None:
                if self._test_prob in sim["interventions"]:
                    sim["interventions"].remove(self._test_prob)
                self._test_prob = None

    def _apply_tracing(self, sim, policy):
        if policy["contact_tracing"]:
            if self._contact_tracing is None:
                self._contact_tracing = cv.contact_tracing(
                    trace_probs=_TRACE_PROBS,
                    trace_time=_TRACE_TIME,
                    start_day=0,
                )
                self._contact_tracing.initialize(sim)
                sim["interventions"].append(self._contact_tracing)
        else:
            if self._contact_tracing is not None:
                if self._contact_tracing in sim["interventions"]:
                    sim["interventions"].remove(self._contact_tracing)
                self._contact_tracing = None
