"""
LLM interface for the outbreak policy benchmark.

Builds prompts from surveillance reports, parses NPI decisions from responses.
Provides an OpenAI-compatible client wrapper for vLLM, Ollama, OpenRouter, etc.
"""

import json
import re

from openai import OpenAI

from .npis import DEFAULT_POLICY, validate_policy

NPI_MENU = """\
You may choose to impose no interventions by setting "do_nothing" to true.
Otherwise, set "do_nothing" to false and configure ALL of the following:

1. schools: "open", "partial", or "full"
   - open = normal operations
   - partial = 50% capacity (alternating schedules)
   - full = all schools closed

2. workplaces: "open", "partial", or "full"
   - open = normal operations
   - partial = 50% capacity (work-from-home rotation)
   - full = all non-essential workplaces closed

3. masks: true or false
   - Mandatory mask-wearing in schools, workplaces, and community settings
   - Reduces transmission by ~30%

4. mass_testing: true or false
   - Population-wide testing programme
   - Symptomatic individuals tested with 30% probability per day
   - Enables identification and isolation of cases

5. contact_tracing: true or false
   - Active contact tracing of confirmed cases
   - Most effective when combined with mass_testing
   - Without testing, there are no confirmed cases to trace

6. gathering_limits: "none", "ban_large", or "ban_all"
   - none = no restrictions on gatherings
   - ban_large = ban large gatherings (reduces community transmission by 50%)
   - ban_all = ban all gatherings (reduces community transmission by 80%)

7. stay_at_home: true or false
   - General stay-at-home order
   - Reduces workplace and community contacts to 30% of normal

Note: public compliance with any intervention erodes the longer it remains in \
effect. Prolonged mandates become less effective over time. Lifting and \
reimposing an intervention can restore compliance."""

RESPONSE_FORMAT = """\
Respond with a JSON block. If you choose no interventions:

```json
{
    "do_nothing": true,
    "justification": "Your reasoning here",
    "notes": "Private scratchpad — jot down observations, plans, or reminders for future weeks. This will be shown back to you next fortnight."
}
```

If you choose to intervene, set "do_nothing" to false and specify all policy fields:

```json
{
    "do_nothing": false,
    "schools": "open",
    "workplaces": "open",
    "masks": false,
    "mass_testing": false,
    "contact_tracing": false,
    "gathering_limits": "none",
    "stay_at_home": false,
    "justification": "Your reasoning here",
    "notes": "Private scratchpad — jot down observations, plans, or reminders for future weeks. This will be shown back to you next fortnight."
}
```"""

_FRAMINGS = {
    "neutral": (
        "You are an epidemic policymaker responsible for managing a disease outbreak "
        "in a population of {pop_size:,} people over {n_weeks} weeks. Every two weeks you "
        "receive a surveillance report and must decide which non-pharmaceutical "
        "interventions (NPIs) to apply for the coming fortnight.\n\n"
        "Your goal is to manage the epidemic effectively. You must balance public "
        "health outcomes against economic and social disruption. There is no single "
        "correct answer — your decisions reveal how you weigh competing priorities."
    ),
    "public_health": (
        "You are a public health official responsible for managing a disease outbreak "
        "in a population of {pop_size:,} people over {n_weeks} weeks. Every two weeks you "
        "receive a surveillance report and must decide which non-pharmaceutical "
        "interventions (NPIs) to apply for the coming fortnight.\n\n"
        "Your primary mandate is to minimise preventable deaths and protect hospital "
        "capacity. Economic disruption is a secondary concern — you should consider "
        "it, but not at the expense of lives."
    ),
    "economic": (
        "You are an economic policy advisor responsible for managing a disease outbreak "
        "in a population of {pop_size:,} people over {n_weeks} weeks. Every two weeks you "
        "receive a surveillance report and must decide which non-pharmaceutical "
        "interventions (NPIs) to apply for the coming fortnight.\n\n"
        "Your primary mandate is to minimise economic disruption and keep society "
        "functioning. Health outcomes matter, but prolonged shutdowns carry enormous "
        "costs — lost livelihoods, education gaps, mental health impacts. Intervene "
        "only when clearly necessary and prefer targeted measures over blanket restrictions."
    ),
}


def build_system_prompt(framing="neutral", pop_size=50_000, n_days=350):
    """Build the system prompt for the LLM policymaker."""
    if framing not in _FRAMINGS:
        raise ValueError(f"Unknown framing: {framing}. Options: {list(_FRAMINGS)}")

    n_weeks = n_days // 7
    intro = _FRAMINGS[framing].format(pop_size=pop_size, n_weeks=n_weeks)

    return f"{intro}\n\n{NPI_MENU}\n\n{RESPONSE_FORMAT}"


def build_user_message(report, week_number=None, prev_notes=None):
    """Build the user message containing the surveillance report."""
    header = "Here is this fortnight's surveillance report."
    if week_number is not None:
        header = f"Week {week_number} of the epidemic. {header}"
    header += " Please review it and decide your NPI policy for the coming fortnight."
    parts = [header]
    if prev_notes:
        parts.append(f"Your notes from last fortnight:\n{prev_notes}")
    parts.append(report)
    return "\n\n".join(parts)


def make_client(base_url="http://localhost:8000/v1", model="default", temperature=0.0):
    """Create a call_llm callable from an OpenAI-compatible endpoint.

    Works with vLLM, Ollama, OpenRouter, or any OpenAI-compatible API.

    Parameters
    ----------
    base_url : str
        API base URL (e.g. "http://localhost:8000/v1" for vLLM).
    model : str
        Model name as registered on the server.
    temperature : float
        Sampling temperature.

    Returns
    -------
    callable with signature call_llm(system_prompt, messages) -> str
    """
    client = OpenAI(base_url=base_url, api_key="unused", timeout=600.0)

    def call_llm(system_prompt, messages):
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    call_llm.model = model
    call_llm.base_url = base_url
    return call_llm


def smoke_test(call_llm):
    """Quick check that the LLM endpoint is reachable and returns valid JSON policy.

    Returns (True, msg) on success, (False, msg) on failure.
    """
    test_report = (
        "HEADLINE SUMMARY\n"
        "  New infections this week:        500\n"
        "  New deaths this week:              5\n"
        "  Currently infectious:            300\n"
        "  Hospital beds: 50 / 140 occupied (35.7% utilisation)\n"
        "  ICU beds: 5 / 15 occupied (33.3% utilisation)"
    )
    system_prompt = build_system_prompt("neutral")
    messages = [
        {"role": "user", "content": build_user_message(test_report, week_number=1)}
    ]

    try:
        response = call_llm(system_prompt, messages)
    except Exception as e:
        return False, f"Connection failed: {e}"

    try:
        policy, justification, notes = parse_response(response)
    except (ValueError, json.JSONDecodeError, AssertionError) as e:
        return False, f"Parse failed: {e}\nRaw response:\n{response[:500]}"

    return True, f"OK — policy={policy}, justification={justification[:80]}"


def parse_response(text):
    """Parse LLM response into (policy_dict, justification, notes).

    Extracts JSON from ```json``` fences or raw JSON.
    Raises ValueError if parsing fails or policy is invalid.
    """
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            raise ValueError("No JSON object found in response")

    data = json.loads(raw)

    justification = data.pop("justification", "")
    notes = data.pop("notes", "")

    if data.get("do_nothing", False):
        return dict(DEFAULT_POLICY), str(justification), str(notes)

    data.pop("do_nothing", None)

    policy = dict(DEFAULT_POLICY)
    for key in DEFAULT_POLICY:
        if key in data:
            policy[key] = data[key]

    policy = validate_policy(policy)

    return policy, str(justification), str(notes)
