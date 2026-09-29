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

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
PROMPT_VERSION = "morph.prompts/1"
MAX_PROMPT_CHARS = 12_000


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
                 timeout_s: float = 90.0, max_retries: int = 2, max_tokens: int = 2500):
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
