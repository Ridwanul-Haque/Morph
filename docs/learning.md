# How Morph learns a skill

The detail behind the README's [Learning](../README.md#3-it-learns-when-a-program-isnt-enough) section: every method
Morph uses when a robot practises a task, and the failure that made each one necessary. Most of these were found by
watching a real session go wrong and fixing it in a way that works for any robot and any task.

## The methods

* **Corrections on top of the program** (residual learning): the learner's action is the program's action at that
  moment plus a learned correction, so a policy that outputs nothing *is* the program. The correction network starts
  at exactly zero and is first fitted to the corrections the program needed from other starts, on slightly noisy inputs
  so it stays steady near the recorded path. The program's own commands are copied with no corrections first, and the
  corrected copy only if it does better: a program's other runs differ from its first mostly in timing, so "other run
  minus first run" taught jolting corrections that shook Spot's parcel loose (5/10, against 10/10 for the plain copy).
* **Keep copying while practising** (DAPG, Rajeswaran et al. 2018): an extra term in PPO's loss keeps a pull towards
  the program's actions, so practice refines the skill instead of slowly forgetting it.
* **Easy-first practice** (reverse curriculum): early rounds start from moments late in the program (already holding
  it, nearly there) and each round moves the starts back. If it's stuck at the same step twice, most practice starts
  where it got closest on that step.
* **A warm-up skill first** (as Eureka's pen spinning first learned to turn the pen to any angle): for a hard goal (the
  object starts in the hand, or the goal is about a stretch of time or a turn) Nemotron may propose an easier version
  of the same motion ("turn it a quarter turn without dropping it"). Morph learns that in `warmup/` and starts the real
  goal from its learned network; if the warm-up isn't learned, the real goal starts from scratch. In a warm-up, how far
  something turned or moved counts from the start of the try (a little progress is never taken back after a pause),
  and breaking a "the whole time" rule (never drop it, never touch X, never tip over) still ends the try but costs
  less than in the real task, so the robot doesn't learn to do nothing for fear of breaking it.
* **Plans compared from the start:** before the first round, Nemotron's alternative plans and the current one practise
  briefly side by side and the best goes on, instead of comparing only once progress stalls. Learning also carries on
  past the planned rounds (up to 4 more) while each round still does better than the last.
* **Plans that fit the body:** Nemotron is told what each finger's joints do and which parts the robot doesn't have; a
  step that relies on a missing part ("rotate the wrist" on a hand fixed in place), or a rule that can never hold
  ("heading at least 360°"), is sent back. A hand holding the object from the start also feels which fingers touch it.
* **The steps a person would take** (Nemotron, per body: which parts, in what order) are the reward: progress
  towards the current step, and a step only counts after it has held for 0.2 s.
* **Eureka-style revision:** after each round Morph replays the policy, measures where it gets stuck and what it
  actually does, and sends that to Nemotron: per goal condition, how often it was met, how close it came, and for
  "the whole time" ones when it first broke and what things looked like then; which fingers touched the object and
  for how long; and, per step, whether the next step got anywhere from where it ended (a step that ends somewhere the
  next can't continue from is flagged, as in Sequential Dexterity). Nemotron rewrites the steps and can add weighted
  **reward terms** (checked formulas such as `closure * (touch_dist < 0.03)`), and sees how each term behaved in
  training. A term is paid as a potential (Ng, Harada & Russell 1999): for raising its value, never for keeping it
  high, so it can guide the robot but can't make stopping halfway pay (a term for keeping the hand over the parcel once
  taught the Stretch to hover there instead of picking it up). A revision is only kept if it practises better than the
  plan it replaces, from the same point. A suggestion of yours gets twice the practice in that race, is judged on the
  goal alone (not on how many of its own new steps it got through, which favours the plan the robot has practised for
  rounds), and wins a tie.
* **Plans are checked before any practice** (LLM-Modulo, Kambhampati et al. 2024): Morph sends a plan back to
  Nemotron, with the reason, if its last step never gets the object where the goal wants it, or if a step's keep rule
  is already broken where that step starts (the robot's start, or saved states where the step before is done), so
  the step would be penalised from its first moment, or if the steps are out of order: an object that has to be
  carried mustn't be asked to be at the goal before it's picked up, and along Morph's own route to the goal the steps
  must come true in their order. A step that asks for a measured value to be exactly a number
  (`base_tilt_deg == 0`) is sent back too: a Stretch's body sits tilted about 0.2°, so that step could never be done.
* **A place it has to be carried into isn't eased** (missed by 29 cm, "in the bin" was met by nudging a pen along the
  floor, and practice learned to push it). Practice starts with it already held help there instead.
* **Wheeled robots grip too:** a gripper is recognised on a mobile base as well, a telescoping arm driven by one motor
  through a tendon is moved as one joint, and the base is set down where the hand can reach the object from one side.
  Morph then grips it itself and practice also starts holding it. If no grip is possible it says why (a Stretch's
  fingertips get no lower than about 5 cm, too high for a pen lying flat).
* **Robots on legs grip too, and from above:** where to stand is also worked out for reaching down onto the object
  (Spot can't get its hand level at a pen on the floor, but reaches down onto it from about 50 cm back). A gripper's
  fingers are only what hangs from its own mount (Spot's feet shifted as it settled and were taken for its fingers, so
  its "hand" was its body). A gripper opens only a little wider than the object (`{"do": "grip", "how": "open",
  "width": 0.04}`), as a hand does: Spot's jaw swings through 90° to open fully and dug into the floor. Where one jaw
  closes against a fixed one, the hand's point follows the middle of the gap as it opens, and "pointing down" levels
  the two jaws' ends; on long jaws (Spot's reach 12 cm past the point measured between them) the point is between
  their ends. When the arm is asked to reach a far point and point a certain way, it also tries getting there first
  and turning after (from Spot's folded arm, doing both at once stalled 58 cm short), and a straight-line move that
  stalls on the way moves the joints straight to the end pose instead.
  Still open: Spot's jaws close onto a pen on the floor but don't hold it. Their tips are rounded underneath (the
  drawn model's too), and the pen, 1.25 cm across, is pushed under them; Morph's pick test finds no grip that holds
  and stops before practice. The Panda's flat finger pads pick up the same pen.
* **Legs and an arm, as a person uses them:** a robot on legs with an arm walks to a spot where its hand reaches the
  object and where it has to go, stands there and works with its arm. Learning a task practises the arm and gripper
  at that spot, the legs holding it standing (practising every leg and arm motor at once from nothing never got
  Spot's hand nearer the pen than it started, 1.3 m off); runs walk there first on its walking skill. "Learn this
  task" teaches it to walk first if it can't yet, in the same job, so it is one button. Standing still there, it
  gets what a bolted-down arm gets: Nemotron's program to learn from and the up-front check that it can pick the
  object up at all.
* **In the hand isn't gripped:** "never falls out of the hand" is `in_hand` (the object kept up by nothing but the
  robot, rolling or sliding between the fingers included), not `held` (a pinch from opposite sides) the whole time,
  which forbids every roll, pass and spin. A request can also say which fingers it starts on ("starts between the
  index and middle fingers"): Morph's in-hand start then lays it over those fingers and keeps a grip with them on it.
  When it doesn't, the start is where the goal first needs it: "at some point it's between the index and middle
  fingers" starts it on those two, not in the steadiest near-fist, which practice never worked it out of.
* **Ready to grip, not just near it:** a step that sets up a grasp has to have the open hand where Morph's own grip
  puts it (`grip_gap`, the hand's distance from that grip), not just near the object: the Stretch learned to rest its
  hand on top of the parcel, which was "within 6 cm", and closing there gripped nothing. Morph adds this to the plan
  itself if Nemotron leaves it out, and the reach reward aims for that grip. A share of practice runs also starts at
  the moment just before Morph's own grip, the gripper still open, so closing on the object is practised every round.
  Without a grip of Morph's own, the step before closing needs the object between the fingers (`between`: the line
  from one fingertip to the other passes through it, in the middle of the gap and not beside one tip), for any robot
  with a gripper or fingers: the Stretch's fingers came within 2.5 cm of a pen round after round and closed beside
  it. (A looser first version let a shut gripper, one tip on the floor next to the pen and the other above it, pass.)
* **The best round is the one that got furthest:** after real successes, moving targets and time in the goal, rounds
  and plans are ranked by the share of the plan's steps they get through, before how close the object ended up (once
  a skill that never touched the pen was kept over ones that reached the grasp, because nudging the pen while trying
  to grip it had left it a little further from the bin). A round only counts as getting better if something clearly
  moved: a success, a moving target, more time in the goal, a twentieth more of the plan, or ending 5% closer. A
  wobble no longer keeps one plan practising round after round without anyone asking where it's stuck.
* **A part it earns instead of the task is flagged:** if one part of the reward pays more and more while the time in
  the goal falls by over half, Nemotron is told which part it is.
* **A small population when progress stalls** (Population Based Training, DexPBT): Nemotron's different plans, plus the
  current plan exploring more and with calmer updates, each practise briefly; the weaker half is dropped and the rest
  practise again; the best plan, policy and settings are kept. They take turns on the GPU. Exploring more is capped
  at 2x the trainer's own setting, however many times it wins. Nemotron is asked for each plan on its own, told the
  ideas already being tried, and reads its measurements by Eureka's rules (no progress: a new idea, not a tweak; a part
  that never changes can't be learned as written; one much bigger than the rest drowns them out).
* **Moving targets** (as Eureka's pen spinning chased a target 30 degrees ahead): for a goal about how far something
  turns or travels, practice pays a bonus each time it reaches the next target (30 degrees, 2.5 cm), and the target
  moves on, so every bit of the motion pays as it happens and holding still never does. Plans and rounds are ranked
  by the targets reached, so progress shows before any try succeeds. Progress only counts once the goal's "the whole
  time" rules have kept holding after it: what was made in the couple of seconds before one breaks is taken back, so
  progress that comes from breaking a rule (the object slipping away, a box shoved until it tips, a lunge that ends in
  a fall) doesn't pay. Breaking such a rule costs the same late in a try as early, so it's never worth it near the
  end, and practice looks about 2 s ahead instead of 0.7 s, far enough to see it coming.
* **Plain practice first:** no varied friction, weight, motor strength, noise or delays until the skill first
  succeeds; then they are turned on, and a skill that is done gets one more round with them before it's finished.
  A hard goal's learning rate follows how much each update changes the policy (rl_games' adaptive schedule).
* **A trainer that goes silent** for 20 minutes is stopped and its round started again once (a hung GPU call once froze
  one for hours); Stop now ends a silent trainer too.
* **An easier version of the goal first** (automatic curriculum): a goal condition it keeps missing may be missed by
  about as much as it misses it now (spin 180 deg/s may first be 100), practice spreads between that and the real goal,
  and the slack shrinks as it improves and is gone after a real success. Rules (no thumb, keep holding it) are never
  eased, and every judgement is on the real goal.
* **Tries end when they can no longer succeed** (early termination, DeepMimic): once a "the whole time" condition has
  broken (the thumb on the pen, the pen on the floor), the practice try stops there and loses what the rest of it
  could have earned.
* **It senses what the goal is about:** if the goal is about spin, the policy also sees how fast the object turns; if
  it's about which fingers touch, it feels each finger's distance to the object (whole fingers, not only the tips:
  the judge counts a thumb's base on the pen too).
* **You can coach it too:** while it practises, type what you see it doing wrong in the Learning tab ("it grabs the
  top edge and it slips; grab it lower"), or rate the practice try you just watched 👍 / 👎. At the end of the round
  (or right away, keeping what it learned so far) Nemotron reads your words, and compares what the tries you liked
  did with the ones you didn't (preference feedback, Christiano et al. 2017), together with Morph's measurements, and
  changes the steps and reward terms. Your input is advice: the measurements still decide what is kept. When you give
  it, or when the copied program doesn't work, practice is allowed to move further away from the program.
* **A critic that sees more** (asymmetric actor-critic, Pinto et al. 2018): during training, the part that judges its
  moves also sees how well it really grips, how far through the steps it is, and each practice copy's true object
  weight and friction. The exported policy only uses what a robot can sense.
* **Fair judging and no lost progress:** 10 tries per round, and the best policy so far is always kept, both across
  rounds and within one (practice that wrecks what it learned hands the next round its best snapshot, not the wreck; a round that ends
  with fewer successes than the best goes on from the best). Contacts come and go for a tick or two even in a steady grip, so a "held" or "touching" that flickers for under 0.1 s isn't a break.
* **Out of reach:** a bolted-down robot that never gets near the object because it stands too far away is moved
  closer, once (not if you placed it yourself).
* **Facing the right way:** for a goal that asks for a heading ("facing west"), turning the object pays from any angle
  while the robot holds it, and once it's at its place: a Stretch set the parcel in the bin but never turned it while
  facing only counted with the parcel already in place and nearly right. It never pays on the floor, where it paid a
  Stretch for turning the parcel by bumping it, and it stopped reaching for it.
* **Open until lined up:** closing the gripper before the hand is at the grip costs a little (nothing while it's open,
  once it's there, or while it holds the object), so practice lines the fingers up first and closes after: a Stretch
  closed early every try and never got the parcel between its fingers.
* Varied physics per practice copy (friction, object mass, motor strength, a one-step delay, sensor noise).

Arm programs and general programs are recorded on the learner's own model, or translated joint by joint, so the copy
matches. Learning from legged programs isn't supported yet: legged robots learn from scratch or use their program.

## Hands

Morph finds hands in any model (on an arm or on their own) and labels the fingers from their names, or from the
hand's shape when the names don't say (the thumb is the finger set apart from the others). Goals can say which part
touches the object: `touching('thumb') == 0` the whole time, `fingers_touching >= 2`. Tries are judged on the real
contacts; the GPU learner counts a finger as touching when any part of it is within 6 mm of the surface. A robot
without the named part is refused before anything runs.

A request about handling the object in the hand ("spin the pen around the fingers") starts with it there: Morph
turns the hand palm up, drops the object onto the palm or fingers, and tries grips (which fingers close, how far)
until one holds it, already meets the goal's "the whole time" conditions, and keeps doing so untouched for 3 s in both
the accurate and the learner's physics. Holding still isn't enough: early practice twitches the fingers at random, so
each grip is also tried with the object laid a few slightly different ways, every pose is shaken like that through
whole tries, and only the sturdiest poses are kept. That grip is what the policy holds when it does nothing; tries start
from those poses, so practice can't depend on one lucky resting pose, and isn't cut short by drops before anything is
learned. In-hand goals
are learned, not programmed (`examples/pen` with the Shadow Hand).
