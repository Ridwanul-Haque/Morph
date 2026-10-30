"""Calls to NVIDIA Nemotron on Nebius Token Factory.

Nemotron makes three decisions per run (never inside the control loop):

1. interpret_task  -- instruction + measured demo -> goal, tolerances, orientation mode
2. select_strategy -- TaskSpec + target capabilities + evidence -> one implemented strategy
3. diagnose        -- failed trial report -> failure class + one allowed revision

Replies are schema-validated and checked against what the robot and scene support.
Every call is logged (prompt version, model, latency, tokens, raw/parsed reply); the
API key is never written to the log.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from .reward_parts import doc as reward_parts_doc

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
PROMPT_VERSION = "morph.prompts/1"
MAX_PROMPT_CHARS = 30_000  # write_program sends earlier attempts with their step-by-step results


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no dependency); never overrides an exported variable."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))



GOAL_CONDITIONS_DOC = (
    " measured.goal_conditions has one entry per goal condition, over the replays: in how many it was met, the share "
    "of the time it held, how close it came (0 = met), the range of each quantity it reads, and for 'the whole time' "
    "conditions when it first broke and the values at that moment. Start from the condition that fails earliest: if "
    "a 'the whole time' condition breaks in the first second, the plan's first steps must keep it (put it in their "
    "keep rule) before anything else can count. measured.handoffs says, for each step that was done, whether the next "
    "step got anywhere from where it ended; a verdict there means that step ends in a state the next can't continue "
    "from, so ask more of its ending (its done_when) rather than of the next step. ")

class ReasoningUnavailable(RuntimeError):
    pass


class ReasoningError(RuntimeError):
    pass


# ------------------------------------------------------------------ schemas
class TaskInterpretation(BaseModel):
    goal_x_m: float
    goal_y_m: float
    goal_yaw_deg: float = Field(description="arrow heading, 0=east(+x), 90=north(+y)")
    pos_tol_cm: float = Field(ge=0.5, le=15)
    yaw_tol_deg: float = Field(ge=1, le=45)
    keep_upright: bool
    orientation_mode: Literal["planar_yaw", "face_up_flip"]
    target_top_face: Literal["+z", "-z", "+x", "-x", "+y", "-y"] = "+z"
    exclusions: list[str] = Field(default_factory=list, max_length=6)
    goal_source: Literal["measured_final_pose", "instruction_overrides_measurement", "mixed"]
    rationale: str = Field(max_length=1200)
    ambiguities: list[str] = Field(default_factory=list, max_length=6)


class StrategyChoice(BaseModel):
    strategy_id: str
    rationale: str = Field(max_length=1200)
    evidence_cited: list[str] = Field(default_factory=list, max_length=8)
    expected_risks: list[str] = Field(default_factory=list, max_length=6)


class RobotPlan(BaseModel):
    what_is_it: str = Field(max_length=400)
    size_and_reach: str = Field(max_length=400)
    goal: str = Field(max_length=400)
    plan_id: str
    how: str = Field(max_length=1200)
    learning_needed: bool
    what_to_learn: str = Field(default="", max_length=800)
    evidence_cited: list[str] = Field(default_factory=list, max_length=8)
    risks: list[str] = Field(default_factory=list, max_length=6)


class GoalReview(BaseModel):
    ok: bool
    problems: list[str] = Field(default_factory=list, max_length=6)


class GoalProposal(BaseModel):
    refusal: str = ""  # when not feasible: missing | contradiction | no_inside | other
    conditions: list[dict] = Field(default_factory=list, max_length=8)
    rationale: str = Field(max_length=1200)
    ambiguities: list[str] = Field(default_factory=list, max_length=6)
    feasible: bool = True
    why_not: str = Field(default="", max_length=600)
    warnings: list[str] = Field(default_factory=list, max_length=4)
    suggestion: str = Field(default="", max_length=400)
    start: str = "scene"  # "in_hand": every try begins with the object already in the robot's hand
    start_fingers: list[str] = Field(default_factory=list, max_length=5)  # the fingers it starts on (in_hand)


# what a formula may call and name (goal_terms checks the same): one text for every prompt that asks for formulas
FUNCS_DOC = ("functions abs, min, max, sqrt, hypot, exp, log, tanh, clip(x, lo, hi), pow(x, n), cos, sin, atan2(y, x) "
             "(radians), cosd, sind, atan2d (degrees); powers x ** n with a number n from -4 to 4; constant pi; and over "
             "time, for what one moment can't show: change(q, s) is how much q changed over the last s seconds (an angle "
             "counted through whole turns: two full turns of yaw_deg are 720, turning back and forth adds up to about 0), "
             "change(q) since the try began; mean(q, s) is q's average over the last s seconds (a condition counts 1 while "
             "it holds: mean(touching('thumb'), 2) is the share of the time the thumb touched it), mean(q) over the whole "
             "try. A rate at each moment (spin_dps > 180) is also met by shaking it fast back and forth; how far it "
             "turned in total isn't: \"keeps turning the same way\", \"round and round\", \"a full circle\" or \"N "
             "turns\" -> abs(change(yaw_deg, 4)) >= 720 (two turns in the last 4 s) or abs(change(yaw_deg)) >= 360 "
             "(once round, in all), checked at the end")
MORE_NAMES_DOC = ("; also goal_x, goal_y, goal_z (where the goal puts the object, when it has a place), each fixture's "
                  "<name>_x, <name>_y, <name>_top (its centre and top, e.g. bin_x), roll_deg and pitch_deg (the object's "
                  "tilt about its own long and short axes), grip_gap (m from the hand's point to where Morph's own "
                  "planned grip puts it on the object: near the object isn't enough to grip it, a hand resting on top "
                  "of it is near; the step before closing the gripper should have grip_gap < 0.03), between (1 while "
                  "the object is between two parts that close on it, a gripper's fingers or a hand's fingertips: a "
                  "gripper can only grab what is between its fingers, so the step before closing it needs between == 1), "
                  "in_hand (1 while the "
                  "object is off the floor and anything else it could rest on, rolling or sliding between the fingers "
                  "included; held is a pinch from opposite sides, which a roll, pass or spin loosens for a moment: when the "
                  "object starts in the hand, a step's \"still has it\" is in_hand == 1, not held, unless the goal "
                  "itself asks for held, and grip_gap is for picking things up, not for an object already in the hand)")
REWARD_TERMS_DOC = (
    "You may also add up to 4 reward_terms: extra shaping on top of the steps, as Eureka does: each {\"name\", "
    "\"formula\", \"weight\", \"why\"}. formula is a NUMBER formula over the same quantities (e.g. \"closure * "
    "(touch_dist < 0.03)\" rewards closing only with the fingers at the object, \"-speed\" penalises flinging it, "
    "\"min(lifted, 0.1)\" rewards lifting up to 10 cm, \"exp(-touch_dist / 0.05)\" pulls a fingertip to it, sharper "
    "up close, \"-(hypot(x - goal_x, y - goal_y)) ** 2\" a squared miss), or a condition (then its graded distance from true counts, 0 "
    "when it holds). weight -5..5: > 0 rewards the value, < 0 penalises it; the value is clipped to -3..3. A term is a "
    "GUIDE, paid as a potential (Ng et al. 1999): the robot earns 0.5 x weight x the change in its value (raising it "
    "pays, letting it fall costs back what was earned), never for keeping it high, so it can't make stopping halfway "
    "(hovering over the object, holding still) pay. It can't make anything required either: what the result must "
    "satisfy belongs in the steps' done_when and keep rules. measured.reward_terms says how each current term behaved "
    "in the last practice (its average value per tick at the start and end): a term that never moves is not helping; "
    "one that rose while the steps stayed stuck points at the wrong thing (fix or drop it). "
)


# Eureka's reward reflection (its code_feedback prompt), for the measured reward parts and terms
REFLECT_DOC = (
    "Read the measurements of each reward part and term first, by these rules: (1) if it never succeeds and its "
    "progress is near zero, don't tweak the plan: rewrite it around a different idea; (2) a part or term whose value "
    "barely changes through practice is one the learner can't improve as written: change its size, rewrite it, or "
    "drop it; (3) a part much bigger than the rest drowns them out: scale it down. "
)


class SkillStepsProposal(BaseModel):
    steps: list[dict] = Field(default_factory=list, max_length=8)
    reward_terms: list[dict] = Field(default_factory=list, max_length=6)  # Eureka-style shaping (revisions only)
    reward_weights: dict = Field(default_factory=dict)  # how much Morph's built-in parts count (reward_parts; revisions)
    rationale: str = Field(default="", max_length=1200)


class StepCandidates(BaseModel):
    candidates: list[dict] = Field(default_factory=list, max_length=4)  # each: {"idea", "steps": [...], "reward_terms"}
    rationale: str = Field(default="", max_length=1500)


class WarmUpProposal(BaseModel):
    needed: bool = False
    idea: str = Field(default="", max_length=300)  # the warm-up in one sentence
    conditions: list[dict] = Field(default_factory=list, max_length=6)  # its checklist, as in goal_from_request
    rationale: str = Field(default="", max_length=1200)


class ProgramProposal(BaseModel):
    steps: list[dict] = Field(default_factory=list, max_length=20)
    rationale: str = Field(default="", max_length=1500)


class Diagnosis(BaseModel):
    failure_class: Literal["wrong_goal", "unreachable", "grasp_lost", "tipped_object", "insufficient_progress",
                           "yaw_not_controlled", "overshoot", "timeout", "robot_instability", "other"]
    evidence: list[str] = Field(max_length=8)
    revision: dict = Field(default_factory=dict)
    give_up: bool = False
    rationale: str = Field(max_length=1200)


# ------------------------------------------------------------------ logging
@dataclass
class InferenceRecord:
    call: str
    prompt_version: str
    model: str
    started: float
    latency_s: float | None = None
    usage: dict | None = None
    request_messages: list | None = None
    raw_response: str | None = None
    parsed: dict | None = None
    validation_error: str | None = None
    http_status: int | None = None
    error: str | None = None
    attempts: int = 0
    accepted: bool = False
    rejection_reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ReasoningService:
    def __init__(self, log_path: Path | None = None, model: str | None = None, base_url: str | None = None,
                 timeout_s: float = 120.0, max_retries: int = 2, max_tokens: int = 6000):
        load_dotenv()
        self._key = os.environ.get("NEBIUS_API_KEY", "").strip()
        self.model = model or os.environ.get("MORPH_NEMOTRON_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)).rstrip("/") + "/"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.log_path = log_path
        self.records: list[InferenceRecord] = []

    @property
    def available(self) -> bool:
        return bool(self._key)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def _log(self, rec: InferenceRecord) -> None:
        self.records.append(rec)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec.as_dict()) + "\n")

    def list_models(self) -> list[str]:
        if not self.available:
            raise ReasoningUnavailable("NEBIUS_API_KEY is not set")
        r = httpx.get(self.base_url + "models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def verify_model(self) -> dict:
        ids = self.list_models()
        nemo = [m for m in ids if "nemotron" in m.lower()]
        return {"configured": self.model, "available": self.model in ids, "nemotron_models_listed": nemo}

    # ------------------------------------------------------------ transport
    def _chat(self, call: str, system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, InferenceRecord]:
        if not self.available:
            raise ReasoningUnavailable("NEBIUS_API_KEY is not set; live Nemotron path unavailable")
        user = user[:MAX_PROMPT_CHARS]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rec = InferenceRecord(call=call, prompt_version=PROMPT_VERSION, model=self.model, started=time.time(),
                              request_messages=messages)
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "temperature": 0.2}
        delay = 2.0
        t0 = time.time()
        last_err = None
        for attempt in range(self.max_retries + 1):
            rec.attempts = attempt + 1
            try:
                r = httpx.post(self.base_url + "chat/completions", headers=self._headers(), json=body,
                               timeout=self.timeout_s)
                rec.http_status = r.status_code
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    ra = r.headers.get("retry-after")
                    time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else delay)
                    delay *= 2
                    continue
                if r.status_code >= 400:
                    rec.error = f"HTTP {r.status_code}: {r.text[:500]}"
                    break
                data = r.json()
                rec.usage = data.get("usage")
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                rec.raw_response = content
                rec.latency_s = round(time.time() - t0, 3)
                try:
                    parsed = schema.model_validate(_extract_json(content))
                except (ValidationError, ValueError) as e:
                    rec.validation_error = str(e)[:1500]
                    last_err = e
                    if attempt < self.max_retries:  # one bounded re-ask with the validation error
                        body["messages"] = messages + [
                            {"role": "assistant", "content": content[-2000:]},
                            {"role": "user", "content": f"Your reply did not validate: {str(e)[:600]}. "
                                                        "Reply with ONLY the corrected JSON object."}]
                        continue
                    break
                rec.parsed = parsed.model_dump()
                self._log(rec)
                return parsed, rec
            except (httpx.TimeoutException, httpx.TransportError) as e:
                rec.error = f"{type(e).__name__}: {e}"
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
        rec.latency_s = round(time.time() - t0, 3)
        self._log(rec)
        raise ReasoningError(rec.error or rec.validation_error or str(last_err))

    # ------------------------------------------------------------ the three calls
    def interpret_task(self, instruction: str, demo_summary: dict, object_info: dict) -> tuple[TaskInterpretation, InferenceRecord]:
        system = (
            "You are the task-interpretation module of Morph, a robotics workbench that transfers the OUTCOME of a "
            "demonstrated task between different robot bodies. You receive (a) the operator's instruction and (b) the "
            "measured object motion from a simulated demonstration. Infer the robot-independent goal for the object. "
            "Never include robot joints, grasps or trajectories. World frame: z up, +x east (yaw 0 deg), +y north "
            "(yaw 90 deg); yaw is the heading of the arrow printed on the block's top face. The block is on the floor. "
            "If the instruction states a heading or place explicitly, prefer it over small measurement deviations and "
            "say so. If the instruction asks for an orientation that changes which face is up, set orientation_mode "
            "to face_up_flip. Reply with ONLY one JSON object with keys: goal_x_m, goal_y_m, goal_yaw_deg, pos_tol_cm, "
            "yaw_tol_deg, keep_upright, orientation_mode, target_top_face, exclusions, goal_source, rationale, ambiguities."
        )
        user = json.dumps({"instruction": instruction, "demonstration_measured": demo_summary, "object": object_info,
                           "default_tolerances": {"pos_tol_cm": 5, "yaw_tol_deg": 15}}, indent=1)
        return self._chat("interpret_task", system, user, TaskInterpretation)

    def select_strategy(self, task: dict, capability: dict, options: list[dict]) -> tuple[StrategyChoice, InferenceRecord]:
        system = (
            "You are the strategy-selection module of Morph. Choose exactly ONE strategy_id from the provided options "
            "for the TARGET robot. Only options listed are implemented; do not invent new ones. Base the choice on the "
            "target's capability report and the recorded evidence for each option (validation success counts, "
            "interface, assumptions). Cite the evidence you used. Reply with ONLY one JSON object with keys: "
            "strategy_id, rationale, evidence_cited, expected_risks."
        )
        user = json.dumps({"task": task, "target_capabilities": capability, "options": options}, indent=1)
        choice, rec = self._chat("select_strategy", system, user, StrategyChoice)
        ids = {o["id"] for o in options}
        if choice.strategy_id not in ids:
            rec.accepted = False
            rec.rejection_reasons.append(f"strategy_id {choice.strategy_id!r} not in implemented options {sorted(ids)}")
            raise ReasoningError(rec.rejection_reasons[-1])
        rec.accepted = True
        return choice, rec

    def plan_for_robot(self, context: dict) -> tuple[RobotPlan, InferenceRecord]:
        system = (
            "You are the planning module of Morph. A robot model Morph has never seen was loaded; its measured profile "
            "(kinematic chains, gripper opening measured by driving it, reach sampled over joint space, mass, size) is "
            "given, together with the task and a list of skill templates whose preconditions were checked in code. "
            "Everything runs in MuJoCo simulation; keep_upright refers to the object, not the robot. "
            "Answer in order: what the robot is, its size and reach, what the task needs, then choose plan_id: one "
            "template id whose applicable flag is true and implemented is true, or 'learn_policy' if none fits but the "
            "robot can learn, or 'unsupported' if nothing can work. Never pick a template with unmet preconditions. If "
            "learning is needed, say concretely what must be learned (e.g. a gait first, then the push) and how the "
            "task goal becomes the reward. Reply with ONLY one JSON object with keys: what_is_it, size_and_reach, goal, "
            "plan_id, how, learning_needed, what_to_learn, evidence_cited, risks."
        )
        plan, rec = self._chat("plan_for_robot", system, json.dumps(context, indent=1, default=str), RobotPlan)
        ok = {t["id"] for t in context["templates"] if t["applicable"] and t["implemented"]}
        learn = next((t for t in context["templates"] if t["id"] == "learn_policy"), None)
        if learn and learn["applicable"]:
            ok.add("learn_policy")
        if not any(t["applicable"] for t in context["templates"]):
            ok.add("unsupported")
        if plan.plan_id not in ok:
            rec.accepted = False
            rec.rejection_reasons.append(f"plan_id {plan.plan_id!r} not allowed here (allowed: {sorted(ok)})")
            raise ReasoningError(rec.rejection_reasons[-1])
        rec.accepted = True
        return plan, rec

    def review_goal(self, request: str, checklist: list[str], scene: dict,
                    morph_requires: list[str] | None = None) -> tuple[GoalReview, InferenceRecord]:
        system = (
            "You check the goal checklist another module wrote for a robot task, before anything runs. The robot "
            "will be judged ONLY on this checklist. Read the operator's request and the checklist (in plain words). "
            "Reply ok=false, with each problem in one short sentence, if: an item contradicts the request (e.g. "
            "\"stays upright\" for \"knock it over\"); something the request clearly asks for is missing (including "
            "how or when: \"from 1 m away\", \"without lifting it\", \"first ... then ...\"); an item adds a demand "
            "the request doesn't make (e.g. lifting for \"on the floor\"); or a number is clearly off. Small, "
            "reasonable tolerances are fine; don't nitpick wording. What a word means physically counts as asked for: "
            "a throw means it is let go while moving fast (e.g. speed > 1 m/s) away from where it lands; a drop means let "
            "go above it; gently means let go almost at rest; knocking over means it ends on its side. The checklist may "
            "also say the object starts in the robot's hand: that is right whenever the request is about handling it in the "
            "hand (spinning, twirling or rolling it in, between or around the fingers or in the palm), which only makes "
            "sense if it is already there; don't call that unasked for. Spinning or twirling means it keeps turning at a real "
            "speed (e.g. over 90 deg/s for the last few seconds), not any rotation at some point; a speed alone is also met by "
            "shaking it quickly back and forth, so \"in the same direction\", \"round and round\", \"a full circle\" or "
            "\"N turns\" need how far it turned in total (change(yaw_deg, s) or change(yaw_deg)), and a checklist "
            "without that is missing it; handling it in the hand "
            "means it stays in the hand (not dropped) while that happens. Numbers that make a vague word measurable (how fast, how "
            "long, how far) are part of reading the request, never an added demand; only complain about one that is clearly "
            "off. morph_requires lists what Morph itself insists on for this kind of request: items that do that are correct, "
            "never flag them. Otherwise ok=true with no problems. Reply with "
            "ONLY one JSON object: {\"ok\": bool, \"problems\": [...]}."
        )
        user = json.dumps({"request": request, "checklist": checklist, "scene": scene,
                           "morph_requires": morph_requires or []}, indent=1, default=str)
        return self._chat("review_goal", system, user, GoalReview)

    def goal_from_request(self, description: str, scene: dict, demo: dict | None,
                          feedback: dict | None = None, robots: list[dict] | None = None
                          ) -> tuple[GoalProposal, InferenceRecord]:
        system = (
            "You are the goal module of Morph. Turn the operator's request (and/or what a demonstration measurably did "
            "to the object) into checkable conditions on the object's final resting state. You get the scene's "
            "measured geometry: the object's size, and for every fixture its footprint, height, the size of its top "
            "face, its opening if it has one, and where it is. Reason physically from those numbers, not from names: "
            "something can rest on a top face only if the face is at least about as wide as the object; something can "
            "go inside only a fixture with an opening bigger than the object. Conditions (JSON objects): "
            "{\"kind\": \"at_position\", \"x\": m, \"y\": m, \"tol\": m}, "
            "{\"kind\": \"heading\", \"yaw_deg\": deg, \"tol\": deg} (0 = east/+x, 90 = north/+y), "
            "{\"kind\": \"upright\", \"tol\": deg}, {\"kind\": \"on_top_of\", \"target\": fixture}, "
            "{\"kind\": \"inside\", \"target\": fixture}, {\"kind\": \"touching\", \"target\": fixture}, "
            "{\"kind\": \"near\", \"target\": fixture, \"tol\": m}, "
            "{\"kind\": \"beside\", \"target\": fixture, \"side\": \"north|south|east|west\"}, "
            "any relation above with \"negate\": true (\"take it out of the bin\" = inside bin negated, \"push it off "
            "the shelf\" = on_top_of shelf negated), "
            "{\"kind\": \"orientation\", \"up_face\": \"top|bottom|side\", \"tol\": deg} (bottom = upside down, side = "
            "knocked over / lying on its side), "
            "{\"kind\": \"height\", \"min_z\": m, \"hold\": bool} (its bottom at least min_z above the floor; hold true "
            "if the robot may keep holding it, e.g. \"hold it up\"), "
            "{\"kind\": \"away_from\", \"target\": fixture, \"tol\": m}, "
            "{\"kind\": \"moved\", \"tol\": m, \"side\": \"north|south|east|west\" or null}, and for ANYTHING ELSE "
            "{\"kind\": \"expression\", \"expr\": formula, \"text\": what it means in plain words}, where the formula "
            "is a Python-like boolean over these measured quantities at the end: x, y, z (object centre, m), bottom, top "
            "(m above the floor), tilt_deg, yaw_deg (0 = east), speed, vx, vy, vz (m/s), travelled (m from its start), start_x, "
            "start_y, held (1 while the robot grips it, pinched from opposite sides), in_hand (1 while it's off the "
            "floor and anything else it could rest on, so the robot is keeping it up, even rolling or sliding between "
            "the fingers), dist_to('fixture') (m, edge to edge), robot_x, robot_y "
            "(robot base), hand_dist (m from the robot's nearest part to the object); " + FUNCS_DOC + "; operators "
            "+ - * / < <= > >= == and or not. Examples: \"balance it on its corner\" -> "
            "\"tilt_deg > 30 and speed < 0.02 and not held\"; \"put it between the bin and the wall\" -> an expression on x "
            "and y from the fixtures' extents; \"bring it to the robot\" -> \"hand_dist < 0.15\". "
            "Any condition can also say WHEN it must hold with \"when\": \"end\" (default: the final resting state), "
            "\"release\" (at the moment the robot lets go of the object: throwing, dropping, tossing), \"always\" (at "
            "every moment: keep it upright the whole way, never touch the wall), \"ever\" (at some point: lift it "
            "above 30 cm at least once) or \"last\" with \"seconds\" (without a break for the last N seconds: keep "
            "it spinning, keep it balanced, keep holding it up), e.g. \"spin the ball\" -> {\"kind\": \"expression\", "
            "\"when\": \"last\", \"seconds\": 2, \"expr\": \"spin_dps > 180\", \"text\": \"spinning at over half a "
            "turn per second\"}; spin_dps is how fast the object turns (deg/s), spin_z_dps its turn about the vertical "
            "(+ anticlockwise from above); a speed alone is met by shaking it back and forth, so a spin that must keep "
            "going one way, or turn a full circle, also needs how far it turned: {\"when\": \"end\", \"expr\": "
            "\"abs(change(yaw_deg, 4)) >= 720\", \"text\": \"turned at least two full turns about the vertical in the "
            "last 4 s\"}. Any condition kind can have a when; the object can end up in only one "
            "place, so other places on the way take when \"ever\". Requests about how it gets there are part of the goal, never "
            "drop them: \"throw the parcel into the bin from 1 m north of the bin\" means inside bin (end) AND, at "
            "release, the parcel is at least 1 m north of the bin, e.g. {\"kind\": \"expression\", \"when\": "
            "\"release\", \"expr\": \"y > <bin's north edge> + 1.0\", \"text\": \"let go at least 1 m north of the "
            "bin\"} with the number filled in from the bin's extent; \"drop it into the bin from 30 cm up\" means "
            "inside bin AND at release bottom > <bin top> + 0.3. "
            "Every request about the object (or where the robot ends up relative to it) can be written this way: "
            "always produce conditions, preferring the specific kinds and using an expression only when none fits. "
            "Use only fixture names from the scene. Keep only conditions the request or demo implies; add a heading "
            "only if one is asked for or the demo clearly turned the object; add upright only when the request asks "
            "for care (keep it upright, carefully, gently, don't tip it, fragile) or the demo kept it upright on purpose. "
            "Interpret everyday phrasing the way a "
            "person would: \"on the wall\" for a thin wall most likely means against it; \"on the bin\" usually means "
            "in it; \"in the bin\" means resting on its floor within its walls, and it is completely normal for an "
            "object to stick out above a container's walls (a tall bottle in a short box is still in the box), so "
            "never reject or warn about inside for being taller than the walls. Respect explicit requests: "
            "\"balance it on X\", \"on top of X\" or \"stand it on X\" means on_top_of X even when unlikely to work; "
            "never substitute a different goal for what was explicitly asked. Say in ambiguities what you assumed. Set feasible "
            "to false only when the request refers to something that is not in the scene (a fixture or a second "
            "object that does not exist), asks for a solid block's inside, or contradicts itself; anything merely hard "
            "or unusual is feasible (Morph will learn it). Explain why in why_not with the "
            "measurements and put the closest workable request in suggestion, written as a plain sentence the operator could type (never JSON). If it is possible but unlikely to "
            "work (balancing on a top narrower than the object), keep feasible true, write the conditions anyway, "
            "and explain the risk in warnings. Warnings are only for real risks of failing, one short sentence each; leave the list empty when there is none (never say that something is fine or feasible there). If request and demo disagree, the request wins. When feasible is false, set refusal to "
            "\"missing\" (names something not in the scene), \"contradiction\" or \"no_inside\" (the inside of a solid "
            "thing); anything else is feasible. Common meanings: throw / toss / lob / fling (with or without a distance) = it "
            "must fly: at release it moves fast (speed > 1.0) and is away from where it must land (e.g. dist_to of that "
            "fixture > 0.25), plus the landing place at the end; a throw is never a placement. Knock it over / tip it "
            "over / topple it = orientation side (lying on its side), never upright; stand it on its side = "
            "orientation side; flip it = orientation bottom; sequences (\"first X, then Y\", "
            "\"tap X then put it on Y\") = X with when \"ever\" and Y at the end; \"without lifting it\" = height of "
            "the bottom kept near 0 the whole time (when \"always\"); \"on the floor\" adds nothing (that's where it "
            "rests anyway) and never means lifting; \"near\", \"next to\", \"by\" with no distance given = near with "
            "tol 0.1; in formulas z is the object's centre (half its height up when resting), so heights of where it "
            "rests use bottom. Write only what the request asks for. Reply with ONLY one "
            "JSON object with keys: conditions, rationale, ambiguities, feasible, refusal, why_not, suggestion, warnings, "
            "start, start_fingers."
        )
        system += (
            " GOALS ABOUT THE ROBOT'S OWN BODY: a request can be about the robot itself rather than the object (\"close "
            "the hand into a fist\", \"open the hand\", \"raise the arm\", \"point the index finger\", \"lift the "
            "front left foot\", \"stand up straight\"). Never refuse those for lack of an object: write them as "
            "expression conditions with these functions of a name pattern (words matched against the robot's joint or "
            "end-part names, | for alternatives, * as a wildcard, or a regular expression such as (index|middle)_(pip|dip)): joint_frac('pattern') = how far the matching joints are through their "
            "range on average (0 at the lower limit, 1 at the upper), joint_frac_min / joint_frac_max = the least / most "
            "of them, and part_x / part_y / part_z('pattern') = where the matching end parts are (metres). robots_selected "
            "lists each selected robot's joints with their ranges (degrees) and its end parts: choose patterns that "
            "exist on the robot(s) meant, and decide from the names and ranges which end of a range means what (for "
            "finger flexion joints such as mcp/pip/dip/pitch/flex, bent = the end farther from 0, usually the upper "
            "limit; a range like [0, 90] bends towards 90). E.g. a fist on a hand whose flexion joints run 0..90: "
            "{\"kind\": \"expression\", \"expr\": \"joint_frac_min('pip|dip|mcp_pitch') > 0.8\", \"text\": \"every "
            "finger bent at least 80% of the way closed\"}. The object can be left out entirely; add object conditions "
            "only if the request mentions the object."
            " WHICH PART TOUCHES IT: touching('pattern') is 1 while any matching finger or body touches the object "
            "(finger labels are thumb, index, middle, ring, little, as listed in robots_selected[].hands; with two hands "
            "left_thumb, right_index, ...), and fingers_touching counts the fingers touching it. E.g. \"without the "
            "thumb\" -> {\"kind\": \"expression\", \"when\": \"always\", \"expr\": \"touching('thumb') == 0\", \"text\": "
            "\"the thumb never touches it\"}; \"hold it with two fingers\" -> fingers_touching == 2 (when \"last\"). "
            "Keeping it in the hand is in_hand, not held: \"never falls out of the hand\", \"don't drop it\", \"keep it in "
            "the hand\" -> {\"kind\": \"expression\", \"when\": \"always\", \"expr\": \"in_hand == 1\", \"text\": \"it never "
            "falls out of the hand\"}. held == 1 the whole time would forbid every roll, pass and spin (they all loosen "
            "the pinch for a moment): use held only when the request asks for a firm grip throughout."
            " WHERE IT STARTS: set \"start\": \"in_hand\" when the request is about handling the object that is already in "
            "the hand (spin / twirl / roll it in or between the fingers, juggle it in the palm, pass it from finger to "
            "finger): every try then begins with the object held in the hand, and Morph puts it there. Otherwise "
            "\"start\": \"scene\" (it starts where it is in the scene, e.g. \"pick it up and spin it\"). If the request "
            "says which fingers it starts on (\"starts between the index and middle fingers\"), list them in "
            "\"start_fingers\" (finger labels, e.g. [\"index\", \"middle\"]): Morph finds a starting grip with those "
            "fingers on it. A start is never a condition (conditions are about the end, the whole time or some moment)."
        )
        payload = {"request": description or "(none: reproduce what the demonstration achieved)",
                   "scene": scene, "demonstration": demo}
        if robots:
            payload["robots_selected"] = robots
        if feedback:
            payload["fix_this"] = feedback  # your previous answer and what was wrong with it
        user = json.dumps(payload, indent=1, default=str)
        return self._chat("goal_from_request", system, user, GoalProposal)

    def skill_steps(self, robot: dict, goal: list[str], scene: dict, scales: dict,
                    fix: dict | None = None, body: dict | None = None) -> tuple[SkillStepsProposal, InferenceRecord]:
        system = (
            "You are the motor-planning module of Morph. A robot must learn, in simulation, to reach a goal from "
            "scratch. Write the steps a person would take without thinking about it, the way someone standing up "
            "puts their feet under them, leans forward over them, pushes, then steadies themselves; or someone "
            "drinking reaches while opening the hand, closes it firmly enough, lifts while keeping the cup upright. "
            "But write them for THIS robot's body: use the parts it actually has (a gripper or a jaw closes on "
            "things; an arm without a gripper can only push, drag or tip with its tool; legs can walk and a foot "
            "can push while standing on the others; a body must stay level), and its size and reach. Include the "
            "unconscious details that make the difference: which side to approach from, pushing low to slide and "
            "high to tip, getting behind the object relative to where it has to go, a firm grip before lifting, "
            "lifting clear of walls before moving across, moving slowly near the end, setting it down before letting "
            "go, keeping balance. If the checklist says where it must be when the robot lets go and that is away from "
            "where it must end up, it is a throw: plan the wind-up and swing, and make the release step require the "
            "object to be moving fast towards the target at that moment (e.g. not held and speed > 1.5 and its velocity "
            "pointing at the target), since it only lands there if it flies. 2 to 5 steps, in order, then the goal itself is checked after them (don't write it "
            "as a step). Every step needs done_when: a boolean formula over these measured quantities, true exactly "
            "when the step is done: x, y, z (object centre, m), bottom, top (m above the floor), tilt_deg, yaw_deg "
            "(0 = east), speed, vx, vy, vz (its velocity, m/s), travelled (m from its start), start_x, start_y, held (1 while the robot holds it), "
            "dist_to('fixture') (m, edge to edge), robot_x, robot_y, base_z, base_tilt_deg (the robot body: 0 = level), "
            "hand_dist (m from the robot's nearest end part, e.g. fingertips or a foot, to the object's centre), "
            "hand_x, hand_y, hand_z (that end part's position)" + MORE_NAMES_DOC + "; " + FUNCS_DOC + "; operators "
            "+ - * / < <= > >= == != and or not (Python syntax: never && || !, and only the names listed: the "
            "object's position is x, y, z, not object_x). Use the scene's numbers (object size, fixture extents, the "
            "target) and the realistic scales given (e.g. how close hand_dist gets when gripping: fingertips sit "
            "either side, so it never reaches 0) to make thresholds concrete and reachable; a step's formula must be something the robot can make true "
            "and that stays meaningful as the object moves (e.g. relative to x and y). keep (optional) is a formula "
            "that should stay true during the step (\"tilt_deg < 20\" while carrying a cup; \"base_tilt_deg < 20\" "
            "for a legged robot). Reply with ONLY one JSON object: {\"steps\": [{\"name\": short imperative, "
            "\"how\": what this body does, in plain words, \"why\": the unconscious reason, \"body_parts\": [...], "
            "\"done_when\": formula, \"keep\": formula or \"\"}], \"rationale\": one or two sentences}."
        )
        payload = {"robot": robot, "goal_checklist": goal, "scene": scene, "realistic_scales": scales}
        if body:  # the goal is only about the robot's own body: no object in the steps
            payload["robot_body"] = body
            system += (" THIS GOAL IS ONLY ABOUT THE ROBOT'S OWN BODY (its joints and parts), not the object: leave the "
                       "object out of every step. Write the steps as joint_frac('pattern') / joint_frac_min / "
                       "joint_frac_max (0 = a joint at its lower limit, 1 = upper) and part_x / part_y / part_z('pattern') "
                       "over the names in robot_body, e.g. for a fist: first the knuckles bend part-way "
                       "(joint_frac_min('mcp_pitch') > 0.4), then the middle joints follow, then all close fully. One or "
                       "two steps are enough for a simple pose.")
        if fix:
            payload["fix_these"] = fix  # the previous answer and what Morph couldn't read in it
        user = json.dumps(payload, indent=1, default=str)
        return self._chat("skill_steps", system, user, SkillStepsProposal)

    def revise_steps(self, robot: dict, goal: list[str], steps: list[dict], diagnosis: dict, scales: dict,
                     rounds: list[dict], fix: dict | None = None, advice: list[str] | None = None,
                     ratings: list[dict] | None = None) -> tuple[SkillStepsProposal, InferenceRecord]:
        system = (
            "You coach a robot that is learning a task by trial and error in simulation (reinforcement learning). It "
            "is rewarded for reaching each step of a plan in order, and for getting closer to the current step. You get "
            "the current steps and measurements of what the trained robot actually does: which step it gets stuck on, "
            "how close it comes, the range of each quantity in that step while it tries, how close its end parts get "
            "to the object, how well it grips and lifts." + GOAL_CONDITIONS_DOC + "Rewrite the steps so it can make progress from where it is, "
            "the way a coach breaks a skill down: split the step it is stuck on into smaller ones it can almost do "
            "already (e.g. first get a fingertip or foot within a few cm of the object's surface, then get two tips on "
            "opposite sides, then close, then lift), loosen thresholds it never gets near, and prefer graded "
            "quantities (touch_dist, grip from 0 to 1, lifted, distances, angles; grip is about 0.25 with the open hand around it, "
            "about 0.6 with the hand closed on it, and 1 once it is really held and lifted; closure is 0..1, how far the "
            "gripper is commanded shut) over yes/no ones like held, which "
            "give it no hint until it is done. Keep the steps it already passes. Think about THIS body: grippers "
            "close on things; legs walk and a foot pushes while the body stays level; an arm with no gripper pushes "
            "or tips. done_when is a boolean formula over these quantities: x, y, z, bottom, top, tilt_deg, yaw_deg, "
            "speed, vx, vy, vz, travelled, start_x, start_y, held, grip, closure, touch_dist, lifted, dist_to('fixture'), "
            "robot_x, robot_y, base_z, base_tilt_deg, hand_dist, hand_x, hand_y, hand_z" + MORE_NAMES_DOC + "; "
            + FUNCS_DOC + "; Python syntax (and, or, not). 2 to 6 steps; the goal checklist is checked after them. "
            + REWARD_TERMS_DOC + reward_parts_doc() + REFLECT_DOC +
            "Reply with ONLY one JSON object: {\"steps\": [{\"name\", \"how\", \"why\", \"body_parts\", "
            "\"done_when\", \"keep\"}], \"reward_terms\": [...] (optional), \"reward_weights\": {...} (optional), "
            "\"rationale\": what you changed and why, from the measurements}."
        )
        payload = {"robot": robot, "goal_checklist": goal, "current_steps": steps, "measured": diagnosis,
                   "realistic_scales": scales, "earlier_rounds": [{k: v for k, v in r.items() if k != "diagnosis"}
                                                                  for r in rounds[:-1]]}
        if fix:
            payload["fix_these"] = fix
        if advice:  # what the person watching the practice runs says it's doing wrong
            payload["person_watching_says"] = advice
            system += (" A person is watching the practice runs and told you what they see it doing wrong "
                       "(person_watching_says). Take it seriously: people spot in seconds what the numbers hide. Turn it "
                       "into measurable steps and reward terms (e.g. 'grab it lower' -> a step with hand_z near obj_z and a "
                       "term rewarding it), check it against the measurements, and say in the rationale how you used it, "
                       "starting with 'You said: ...'. If it can't be measured with these quantities, or it contradicts the "
                       "goal, say so plainly and do what the measurements support instead. A suggestion about one part "
                       "of the reward reads 'About the reward part <part>: ...': change that part's reward_weights (less = "
                       "about 0.5, much less = 0.2, remove = 0, more = 1.5, within its range), or for 'only once it grips "
                       "it' lower it and add a reward term that pays the same only when grip is high. A suggestion "
                       "about one step reads 'About plan step <n> (\"<name>\", done when: <formula> = <plain words>): ...': "
                       "the person says how that step was understood wrongly. Rewrite that step as they say (its how and its "
                       "done_when; use any number they give), keep the other steps unless they must change to fit, and say "
                       "in the rationale what the step is now.")
        if ratings:  # practice tries the person marked as looking right / wrong, each with what it measurably did
            payload["person_rated_tries"] = [{"rating": a["rating"], "what_it_did": a.get("did", {})} for a in ratings][-12:]
            system += (" The person also rated practice tries they watched (person_rated_tries: good = looks right, bad = "
                       "looks wrong), each with what that try measurably did. Compare the good ones with the bad ones to "
                       "work out what they want (e.g. the liked tries lifted it higher, or held it longer) and reward that; "
                       "say in the rationale what difference you saw.")
        user = json.dumps(payload, separators=(",", ":"), default=str)
        return self._chat("revise_steps", system, user, SkillStepsProposal)

    def step_candidates(self, robot: dict, goal: list[str], steps: list[dict], diagnosis: dict, scales: dict,
                        rounds: list[dict], k: int = 3, fix: dict | None = None,
                        others: list[str] = ()) -> tuple[StepCandidates, InferenceRecord]:
        """Different revisions (Eureka-style search): Morph trains each briefly and keeps the best. others: the ideas
        already proposed in earlier calls (Morph asks for one at a time, as Eureka samples each reward on its own)."""
        system = (
            f"You coach a robot that is learning a task by trial and error in simulation. Propose {k} revised step "
            f"plan{'s' if k > 1 else ''}, each built on a different idea about why it is stuck, and on none of the "
            "ideas_already_being_tried (if given) (e.g. a smaller first step, a different "
            "approach direction, a graded intermediate condition, loosening a threshold it never reaches, a different "
            "body part). Morph will train each briefly and keep whichever works best, so make them genuinely "
            "different rather than small variations. Rules for every plan are the same as for a single revision: "
            "2 to 6 steps, done_when is a boolean formula over: x, y, z, bottom, top, tilt_deg, yaw_deg, speed, vx, vy, "
            "vz, travelled, start_x, start_y, held, grip, closure, touch_dist, lifted, dist_to('fixture'), robot_x, "
            "robot_y, base_z, base_tilt_deg, hand_dist, hand_x, hand_y, hand_z" + MORE_NAMES_DOC + "; " + FUNCS_DOC
            + "; Python syntax." + GOAL_CONDITIONS_DOC + "grip is about 0.25 with the open hand around it, 0.6 closed on it, 1 held and lifted; closure is "
            "how far the gripper is commanded shut. " + REWARD_TERMS_DOC + "A candidate can differ in its steps, its "
            "reward_terms, or both. " + REFLECT_DOC + "Reply with ONLY one JSON object: {\"candidates\": [{\"idea\": one "
            "sentence, \"steps\": [{\"name\", \"how\", \"why\", \"body_parts\", \"done_when\", \"keep\"}], "
            "\"reward_terms\": [...] (optional)}], \"rationale\": what the measurements suggest}."
        )
        payload = {"robot": robot, "goal_checklist": goal, "current_steps": steps, "measured": diagnosis,
                   "realistic_scales": scales, "earlier_rounds": [{k2: v for k2, v in r.items() if k2 != "diagnosis"}
                                                                  for r in rounds[:-1]]}
        if fix:
            payload["fix_these"] = fix
        if others:
            payload["ideas_already_being_tried"] = list(others)
        return self._chat("step_candidates", system, json.dumps(payload, separators=(",", ":"), default=str), StepCandidates)

    def warm_up_goal(self, robot: dict, goal: list[dict], start: str) -> tuple[WarmUpProposal, InferenceRecord]:
        """An easier skill to learn first, whose learned body makes the real goal easier to learn (a curriculum)."""
        system = (
            "You plan the learning of a robot skill in Morph. The robot learns by trial and error in simulation, judged "
            "only by a goal checklist. Hard dexterous goals are rarely found from scratch: Eureka learned pen spinning "
            "on a Shadow Hand only by first learning to turn the pen to any angle it was asked, then continuing from "
            "that skill. Decide whether THIS goal needs such a warm-up on the same robot and object, starting the same "
            "way, and if so write the warm-up's own checklist. A good warm-up is the same kind of motion asked for in a "
            "smaller, slower or shorter amount (a quarter turn instead of two turns, turning it at all before turning "
            "it fast, moving it a little before moving it far), keeps every 'the whole time' rule of the goal that "
            "can still hold (never drop it, never touch with the thumb), and is something the robot's body (robot."
            "finger_joints, robot.has_no) can plausibly do. It must not be a different skill whose learned motion "
            "would have to be unlearned. The learner earns points for every bit closer a condition gets, the moment "
            "it happens; holding still earns nothing but loses nothing. So the warm-up must pay for each small bit of "
            "the right motion and never take it back: count a turn or a distance as a running total since the try "
            "began (change(yaw_deg), with no seconds, when 'end'), not over the last few seconds (a short turn then "
            "a pause would lose what it earned), and leave out speed thresholds (spin_dps > 30 pays nothing until it "
            "is reached, and rewards a jerk). E.g. for spinning: in_hand == 1 'always' and abs(change(yaw_deg)) >= 90 "
            "'end'. needed=false when the goal is already one short, simple movement. "
            "Each condition is {\"kind\": \"expression\", \"expr\": formula, \"when\": \"end\"|\"always\"|\"ever\"|"
            "\"last\", \"seconds\": N (for last), \"text\": plain words}; formulas are over: x, y, z, bottom, top, "
            "tilt_deg, yaw_deg, speed, vx, vy, vz, travelled, held, in_hand, spin_dps, spin_z_dps, fingers_touching, "
            "touching('finger')" + MORE_NAMES_DOC + "; " + FUNCS_DOC + "; Python syntax. The goal's own conditions "
            "show the right style. Reply with ONLY one JSON object: {\"needed\": bool, \"idea\": the warm-up in one "
            "sentence, \"conditions\": [...], \"rationale\": why it helps the real goal}."
        )
        user = json.dumps({"robot": robot, "goal_checklist": goal, "start": start}, separators=(",", ":"), default=str)
        return self._chat("warm_up_goal", system, user, WarmUpProposal)

    def write_program(self, robot: dict, goal: list[str], scene: dict, facts: dict, blocks: str,
                      attempts: list[dict], body: str = "arm") -> tuple[ProgramProposal, InferenceRecord]:
        if body != "arm":  # any other body: the same loop, guidance that doesn't assume a hand pointing down
            system = (
                "You are the motion-programming module of Morph. Write a short program that makes THIS robot reach the "
                "goal in a physics simulation, using only these building blocks: " + blocks + " First decide which part of "
                "its body to use for what (facts.parts lists an arm's parts), as a person or an animal would with this "
                "body: a gripping part to pick things up, a tip or a flat tool to push or poke, a foot or the whole body "
                "of a walking robot to nudge or push, two parts on opposite sides to squeeze. Then the order: get the part into position clear of the object, approach slowly for the "
                "last few cm, act, and keep or release contact as the goal needs. Use the facts (where each part is now, "
                "about how far it reaches, the object's size and where it must end, the fixtures) to pick reachable, "
                "concrete numbers; use formulas such as obj_x or obj_top + 0.05 where it should follow the object. "
                "Pushing: put the part against the object's far side from where it must go, low on its face, and push "
                "along the line to the goal; pushing high tips it over. The checklist is judged at the end of the run "
                "(and at release for 'when it lets go' items), after the object settles. If earlier attempts are given, "
                "read exactly what happened in each step and fix the cause; never repeat a program that failed. Reply "
                "with ONLY one JSON object: {\"steps\": [{\"do\": <building block>, <its arguments>}, ...], "
                "\"rationale\": which parts you use for what, and what you changed from the last attempt}."
            )
            user = json.dumps({"robot": robot, "goal_checklist": goal, "scene": scene, "facts": facts,
                               "earlier_attempts": attempts[-3:]}, separators=(",", ":"), default=str)
            return self._chat("write_program", system, user, ProgramProposal)
        system = (
            "You are the motion-programming module of Morph. Write a short program that makes THIS robot reach the goal "
            "in a physics simulation, using only these building blocks: " + blocks + " Think like a person doing the "
            "task without thinking about it, and for this body: open the hand before reaching for something; come down "
            "onto it from above, lined up with its sides; close the hand before lifting; lift clear of anything it has "
            "to pass over before moving across; move slowly for the last few cm when placing or touching; set things "
            "down before letting go, unless the goal is to throw or drop it (then let go with release_at while the "
            "hand is moving fast towards the target, or above it). Use the facts given (where the arm stands, how far "
            "it reaches, the lowest the hand can go, the object's size, the fixtures' positions) to pick reachable, "
            "concrete numbers; use formulas such as obj_x or obj_top + 0.05 where it should follow the object. "
            "The checklist is judged only at the end of the run (and at release for 'when it lets go' items), after "
            "the object settles. If earlier attempts are given, read exactly what happened in each step and fix the "
            "cause (e.g. a hand that stopped short is out of reach: choose another point; fingers that closed on "
            "nothing were not around the object: check its height); never repeat a program that failed. Reply with "
            "ONLY one JSON object: {\"steps\": [{\"do\": \"move_hand\", \"x\": ..., ...}, {\"do\": \"close_hand\"}, ...], "
            "\"rationale\": what each part is for, and what you changed from the last attempt}."
        )
        user = json.dumps({"robot": robot, "goal_checklist": goal, "scene": scene, "facts": facts,
                           "earlier_attempts": attempts[-3:]}, separators=(",", ":"), default=str)
        return self._chat("write_program", system, user, ProgramProposal)

    def diagnose(self, failure_report: dict, revision_space: dict) -> tuple[Diagnosis, InferenceRecord]:
        system = (
            "You are the failure-diagnosis module of Morph. You receive a structured report of a simulated trial that "
            "did not meet its goal (measured object poses, contacts, errors over time, the strategy parameters used) "
            "and the space of allowed revisions. Classify the failure, cite the specific evidence, and propose ONE "
            "revision using only keys and ranges from revision_space, or give_up=true if no allowed revision can help. "
            "Reply with ONLY one JSON object with keys: failure_class, evidence, revision, give_up, rationale."
        )
        user = json.dumps({"trial": failure_report, "revision_space": revision_space}, indent=1)
        d, rec = self._chat("diagnose", system, user, Diagnosis)
        bad = validate_revision(d.revision, revision_space)
        if bad:
            rec.accepted = False
            rec.rejection_reasons += bad
            d = d.model_copy(update={"revision": {k: v for k, v in d.revision.items() if k not in {b.split(':')[0] for b in bad}}})
        rec.accepted = not bad
        return d, rec


def validate_revision(rev: dict, space: dict) -> list[str]:
    """Reject revision keys/values outside the declared, implemented space."""
    bad = []
    for k, v in rev.items():
        if k not in space:
            bad.append(f"{k}: not an allowed revision key")
            continue
        spec = space[k]
        if "choices" in spec and v not in spec["choices"]:
            bad.append(f"{k}: {v!r} not in {spec['choices']}")
        elif "range" in spec:
            lo, hi = spec["range"]
            if not isinstance(v, (int, float)) or not (lo <= v <= hi) or (isinstance(v, float) and math.isnan(v)):
                bad.append(f"{k}: {v!r} outside [{lo}, {hi}]")
    return bad


def _extract_json(text: str) -> dict:
    """Parse the last JSON object in a reply (reasoning models may think before answering)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fence = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = fence or []
    if not candidates:
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in model reply")
