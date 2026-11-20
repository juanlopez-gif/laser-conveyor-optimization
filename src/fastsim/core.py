"""Event-driven simulator of the laser + conveyors cell (strong-RL plan, phase 1).

Same model and rules as the fixed-step engine (:mod:`simulation.engine`), but time
jumps from one event to the next instead of advancing by ``dt``. Between two events
every belt moves at constant speed (or stands still) and the laser moves each axis at
constant speed towards its target, so every crossing time is known in closed form:

* an arrival;
* the end of the current processing;
* a part entering the laser zone (x = a: MOVING -> READY);
* an unprocessed part reaching the end of the laser zone (x = b: the belt blocks);
* a part leaving the belt (x = L);
* the last part on a belt reaching the minimum spacing (the next waiting part enters);
* the laser reaching its target point, and the target part reaching it (contact);
* the arrival horizon and ``max_time`` (stop rule).

At each event time the instantaneous logic runs in the same order as one step of the
fixed-step engine: finish processing, arrivals, blocking, admission, statuses,
revocation of a pre-positioning, dispatching, laser contact. The fixed-step engine
rounds every event up to its ``dt`` grid; this simulator does not. The two therefore
agree up to O(dt) per event; ``tests/test_fastsim.py`` quantifies the agreement.

Scope: ``unprocessed_at_zone_end: block`` (the confirmed rule of the real cell), both
conveyor stop policies, pre-positioning ``none`` / ``when_idle`` / ``always``, both
travel-time models. ``pass`` mode is not implemented.

Decision API (used by policies, search and RL):

    sim = FastSim(cfg)
    while sim.run_until_decision():          # False when the episode is over
        cands = sim.candidates()             # part indices
        sim.apply(choice)                    # one of cands
    sim.summary()

or ``sim.run(policy)`` with ``policy(sim, candidates) -> part index``.
``clone()`` returns an independent copy (flat lists, no deepcopy).
"""

from __future__ import annotations

import math

from simulation.arrivals import generate_arrivals
from simulation.config import Config
from simulation.motion import Intercept, find_intercept, path_length, travel_time

EPS = 1e-7          # same tolerance as simulation.engine
SNAP = 1e-9         # position tolerance for "at the target" / thresholds
INF = math.inf

# part status codes (same life cycle as simulation.part.PartStatus)
WAITING, MOVING, READY, PROCESSING, COMPLETED, EXITED = range(6)
STATUS_NAMES = ("WAITING", "MOVING", "READY", "PROCESSING", "COMPLETED", "EXITED")

# laser states
IDLE, HEADING, BUSY = 0, 1, 2      # HEADING = MOVING or WAITING_FOR_PART


class FastSim:
    """One episode of the cell, advanced event by event."""

    # ------------------------------------------------------------------ setup
    def __init__(self, cfg: Config, arrivals=None) -> None:
        s = cfg.simulation
        if s.unprocessed_at_zone_end != "block":
            raise NotImplementedError("fastsim implements unprocessed_at_zone_end: block only")
        g = cfg.geometry
        self.cfg = cfg
        self.a, self.b, self.L = g.active_start, g.active_end, g.total_length
        self.spacing = s.min_part_spacing
        self.horizon, self.max_time, self.drain = s.arrival_horizon, s.max_time, s.drain
        self.stop_all = s.conveyor_stop_policy == "all_conveyors"
        if s.conveyor_stop_policy not in ("processing_conveyor_only", "all_conveyors"):
            raise NotImplementedError(f"stop policy {s.conveyor_stop_policy}")
        self.mode = cfg.laser.prepositioning
        self.n = len(cfg.conveyors)
        self.v = [c.speed for c in cfg.conveyors]
        self.yk = [c.y for c in cfg.conveyors]
        sp = cfg.laser
        self.vx, self.vy, self.motion = sp.speed_x, sp.speed_y, sp.motion_model
        self.ws = sp.workspace
        self.allowance = s.due_date_allowance

        recs = generate_arrivals(cfg) if arrivals is None else arrivals
        # immutable instance data, shared by clones
        self.arr = tuple((r.time, r.conveyor_index, r.processing_time) for r in recs)

        self.t = 0.0
        self.next_arr = 0
        # per part (grows on arrival)
        self.pk: list[int] = []          # conveyor index
        self.pr: list[float] = []        # arrival time r_j
        self.pp: list[float] = []        # processing time p_j
        self.pst: list[int] = []         # status
        self.pE: list[float | None] = []   # entry coordinate E_j (odometer at entry)
        self.pD0: list[float] = []       # odometer at arrival D_k(r_j)
        self.pentry: list[float | None] = []
        self.pready: list[float | None] = []
        self.pS: list[float | None] = []
        self.pC: list[float | None] = []
        self.pB: list[float | None] = []
        self.pX: list[float | None] = []
        self.pDS: list[float | None] = []
        self.pexit: list[float | None] = []
        # per belt
        self.D = [0.0] * self.n                      # odometers
        self.belt = [[] for _ in range(self.n)]      # part indices on the belt, downstream first
        self.buf = [[] for _ in range(self.n)]       # waiting part indices, FIFO
        self.s_proc = [False] * self.n               # stop reasons
        self.s_blk = [False] * self.n
        self.moved = [False] * self.n                # belt moved since the last event
        # laser
        self.lx, self.ly = sp.home_x, sp.home_y
        self.lstate = IDLE
        self.target = -1
        self.tx = self.ty = 0.0
        self.planned_v = 0.0
        self.proc_end = INF
        self.assigned = -1                           # part the laser is committed to
        self.dist = 0.0
        # accumulators
        self.wip_area = 0.0          # integral of (arrived - completed)  = sum F_j at the end
        self.blocked_area = 0.0      # integral of the number of blocked belts = T_B
        self.n_completed = 0
        self.n_block_events = 0
        self.n_decisions = 0
        self.order: list[int] = []   # processing order (part indices)
        self.done = False
        self._pending: list[int] | None = None       # candidates of an open decision
        self._memo: dict[int, Intercept | None] = {}
        self._zero_steps = 0

    def clone(self) -> "FastSim":
        """Independent copy of the current state (instance data shared, read-only)."""
        c = FastSim.__new__(FastSim)
        c.__dict__.update(self.__dict__)
        for name in ("pk", "pr", "pp", "pst", "pE", "pD0", "pentry", "pready", "pS", "pC",
                     "pB", "pX", "pDS", "pexit", "D", "s_proc", "s_blk", "moved", "order"):
            setattr(c, name, list(getattr(self, name)))
        c.belt = [list(q) for q in self.belt]
        c.buf = [list(q) for q in self.buf]
        c._pending = None if self._pending is None else list(self._pending)
        c._memo = dict(self._memo)
        return c

    # ---------------------------------------------------------------- helpers
    def x(self, j: int) -> float:
        """Current position of part j (0 while waiting)."""
        e = self.pE[j]
        return 0.0 if e is None else self.D[self.pk[j]] - e

    def running(self, k: int) -> bool:
        return not (self.s_proc[k] or self.s_blk[k])

    def speed(self, k: int) -> float:
        return self.v[k] if not (self.s_proc[k] or self.s_blk[k]) else 0.0

    def intercept(self, j: int) -> Intercept | None:
        """Meeting point between the laser and part j now (memoized per instant)."""
        m = self._memo
        if j not in m:
            k = self.pk[j]
            if self.s_proc[k] or self.s_blk[k]:
                xj = self.D[k] - self.pE[j] if self.pE[j] is not None else 0.0
                if self.a - 1e-9 <= xj <= self.b + 1e-9:
                    # stopped part inside the zone: the same expression as
                    # motion.travel_time, inlined (most common case at heavy load)
                    tx = abs(xj - self.lx) / self.vx
                    ty = abs(self.yk[k] - self.ly) / self.vy
                    m[j] = Intercept(max(tx, ty) if self.motion == "simultaneous" else tx + ty,
                                     xj, self.yk[k])
                    return m[j]
            m[j] = find_intercept(self.lx, self.ly, self.x(j), self.yk[k], self.speed(k),
                                  self.b, True, self.vx, self.vy, self.motion, x_min=self.a)
        return m[j]

    def travel_time(self, j: int) -> float:
        icp = self.intercept(j)
        return INF if icp is None else icp.time

    def number(self, j: int) -> int:
        """1-based part number (id P001 = 1), as in simulation.part.Part.number."""
        return j + 1

    # ------------------------------------------------------------------ API
    def candidates(self) -> list[int]:
        """Candidates of the open decision (empty if none is open)."""
        return [] if self._pending is None else list(self._pending)

    def run_until_decision(self) -> bool:
        """Advance until the laser must choose among >= 1 candidates (True) or the
        episode ends (False)."""
        if self._pending is not None:
            return True
        while not self.done:
            self._instant_pre()
            if self.lstate == IDLE:
                cands = self._candidates()
                if cands:
                    self._pending = cands
                    return True
            if self._instant_post():       # assignment dropped: decide again now
                continue
            self._advance()
        return False

    def apply(self, j: int) -> None:
        """Assign the laser to candidate part j of the open decision."""
        if self._pending is None or j not in self._pending:
            raise ValueError(f"part {j} is not a candidate of the open decision")
        self._pending = None
        self.n_decisions += 1
        icp = self.intercept(j)
        self.assigned = self.target = j
        self.tx, self.ty = self.ws.clamp(icp.x, icp.y)
        self.planned_v = self.speed(self.pk[j])
        self.lstate = HEADING
        if not self._instant_post():          # if dropped, the next call decides again
            self._advance()

    def run(self, policy) -> dict:
        """Run to the end with ``policy(sim, candidates) -> part index``."""
        while self.run_until_decision():
            self.apply(policy(self, self.candidates()))
        return self.summary()

    # ----------------------------------------------------- instantaneous logic
    def _instant_pre(self) -> None:
        """Steps 1-4 of the fixed-step engine + statuses + revocation, at time t."""
        t = self.t
        self._memo = {}
        # 1 finish processing
        if self.lstate == BUSY and t >= self.proc_end - EPS:
            j = self.target
            self.pst[j] = COMPLETED
            self.pC[j] = t
            self.n_completed += 1
            self.order.append(j)
            for k in range(self.n):
                self.s_proc[k] = False
            self.lstate, self.target, self.assigned, self.proc_end = IDLE, -1, -1, INF
        # 2 arrivals
        arr = self.arr
        while self.next_arr < len(arr) and arr[self.next_arr][0] <= t + EPS:
            r, k, p = arr[self.next_arr]
            self.next_arr += 1
            j = len(self.pk)
            self.pk.append(k); self.pr.append(r); self.pp.append(p); self.pst.append(WAITING)
            self.pE.append(None); self.pD0.append(self.D[k]); self.pentry.append(None)
            self.pready.append(None); self.pS.append(None); self.pC.append(None)
            self.pB.append(None); self.pX.append(None); self.pDS.append(None)
            self.pexit.append(None)
            self.buf[k].append(j)
        # statuses (the fixed-step engine updates them at the end of the previous step)
        # and 3 blocking. Both can only change on a belt that moved since the last
        # event (a block is released at S_j, in _start_processing).
        a, L, bthr = self.a - EPS, self.L - EPS, self.b - 1e-9
        pst, pE, pS, pB = self.pst, self.pE, self.pS, self.pB
        moved = self.moved
        for k in range(self.n):
            if not moved[k]:
                continue
            moved[k] = False
            q, Dk = self.belt[k], self.D[k]
            blocking = -1
            for j in list(q):
                xj = Dk - pE[j]
                if pst[j] == MOVING and xj >= a:
                    pst[j] = READY
                    self.pready[j] = t
                if xj >= L:
                    q.remove(j)
                    pst[j] = EXITED
                    self.pexit[j] = t
                elif blocking < 0 and pS[j] is None and xj >= bthr:
                    blocking = j
            if blocking >= 0:
                self.s_blk[k] = True
                if pB[blocking] is None:
                    pB[blocking] = t
            else:
                self.s_blk[k] = False
        # 4 admission
        for k in range(self.n):
            while self.buf[k] and self.running(k):
                q = self.belt[k]
                if q and self.D[k] - self.pE[q[-1]] < self.spacing - 1e-9:
                    break
                j = self.buf[k].pop(0)
                self.pE[j] = self.D[k]
                self.pentry[j] = t
                self.pst[j] = MOVING
                q.append(j)
                if self.spacing > 0:
                    break
        # revocation of a pre-positioning (when_idle)
        if (self.mode == "when_idle" and self.lstate == HEADING
                and self.pst[self.target] == MOVING and self._ready_exists()):
            self._drop()

    def _ready_exists(self) -> bool:
        for q in self.belt:
            for j in q:
                if self.pst[j] == READY and j != self.assigned:
                    return True
        return False

    def _candidates(self) -> list[int]:
        upcoming = self.mode == "always" or (self.mode == "when_idle" and not self._ready_exists())
        out = []
        for q in self.belt:
            for j in q:
                st = self.pst[j]
                if j != self.assigned and (st == READY or (upcoming and st == MOVING)) \
                        and self.intercept(j) is not None:
                    out.append(j)
        return out          # engine order: conveyor by conveyor, downstream first

    def _drop(self) -> None:
        self.lstate, self.target, self.assigned = IDLE, -1, -1

    def _instant_post(self) -> bool:
        """Step 6: drive the laser (re-plan, contact, start processing).
        Returns True if the assignment was dropped (the laser became idle)."""
        if self.lstate != HEADING:
            return False
        j = self.target
        k = self.pk[j]
        st = self.pst[j]
        reachable = st == READY or (self.mode != "none" and st == MOVING)
        xj = self.x(j)
        if not reachable or xj > self.b + EPS:
            self._drop()
            return True
        vk = self.speed(k)
        at_target = abs(self.lx - self.tx) < SNAP and abs(self.ly - self.ty) < SNAP
        contact = at_target and self.tx - EPS <= xj <= self.tx + EPS
        if not contact and (vk != self.planned_v or xj > self.tx + EPS):
            self._memo.pop(j, None)
            icp = self.intercept(j)
            if icp is None:
                self._drop()
                return True
            self.tx, self.ty = self.ws.clamp(icp.x, icp.y)
            self.planned_v = vk
            at_target = abs(self.lx - self.tx) < SNAP and abs(self.ly - self.ty) < SNAP
            contact = at_target and self.tx - EPS <= xj <= self.tx + EPS
        if contact:
            self._start_processing(j)
        return False

    def _start_processing(self, j: int) -> None:
        t, k = self.t, self.pk[j]
        xj = self.x(j)
        # final alignment (sub-EPS)
        self.dist += path_length(xj - self.lx, self.yk[k] - self.ly)
        self.lx, self.ly = xj, self.yk[k]
        self.lstate = BUSY
        self.proc_end = t + self.pp[j]
        self.pst[j] = PROCESSING
        self.pS[j] = t
        self.pX[j] = xj
        self.pDS[j] = self.D[k]
        if self.pB[j] is not None and t - self.pB[j] > 1e-9:
            self.n_block_events += 1
        # the block ends at S_j only if no other unprocessed part sits at b
        bthr = self.b - 1e-9
        self.s_blk[k] = any(self.pS[i] is None and self.D[k] - self.pE[i] >= bthr
                            for i in self.belt[k])
        if self.stop_all:
            for i in range(self.n):
                self.s_proc[i] = True
        else:
            self.s_proc[k] = True

    # ----------------------------------------------------------------- time
    def _should_stop(self) -> bool:
        t = self.t
        if t >= self.max_time - EPS:
            return True
        if t < self.horizon - EPS or self.next_arr < len(self.arr):
            return False
        if not self.drain:
            return True
        return (self.lstate == IDLE and not any(self.belt) and not any(self.buf))

    def _next_event(self) -> float:
        """Time until the next event (> 0 unless something is due now)."""
        t = self.t
        dt = self.max_time - t
        if t < self.horizon - EPS:
            dt = min(dt, self.horizon - t)
        if self.next_arr < len(self.arr):
            dt = min(dt, self.arr[self.next_arr][0] - t)
        if self.lstate == BUSY:
            dt = min(dt, self.proc_end - t)
        a, b, L, sp = self.a, self.b, self.L, self.spacing
        aE, bE, LE = a - EPS, b - 1e-9, L - EPS
        pE, pst, pS = self.pE, self.pst, self.pS
        for k in range(self.n):
            if self.s_proc[k] or self.s_blk[k]:
                continue
            Dk = self.D[k]
            q = self.belt[k]
            dist = INF                      # shortest distance to the next crossing
            for j in q:
                xj = Dk - pE[j]
                if xj < aE and pst[j] == MOVING:
                    d = a - xj
                elif xj < bE and pS[j] is None:
                    d = b - xj
                elif xj < LE:
                    d = L - xj
                else:
                    continue
                if d < dist:
                    dist = d
            if self.buf[k] and q:
                xl = Dk - pE[q[-1]]
                if xl < sp - 1e-9 and sp - xl < dist:
                    dist = sp - xl
            if dist < INF:
                dt = min(dt, dist / self.v[k])
        if self.lstate == HEADING:
            if abs(self.lx - self.tx) >= SNAP or abs(self.ly - self.ty) >= SNAP:
                dt = min(dt, travel_time(self.lx, self.ly, self.tx, self.ty,
                                         self.vx, self.vy, self.motion))
            k = self.pk[self.target]
            if self.running(k):
                xj = self.x(self.target)
                if xj < self.tx - EPS:
                    dt = min(dt, (self.tx - xj) / self.v[k])
        return max(dt, 0.0)

    def _advance(self) -> None:
        """Stop check, then jump to the next event and integrate the costs."""
        if self._should_stop():
            self.done = True
            return
        dt = self._next_event()
        if dt <= 1e-12:
            self._zero_steps += 1
            if self._zero_steps > 1000:
                raise RuntimeError(f"fastsim stuck at t={self.t}")
            dt = 0.0
        else:
            self._zero_steps = 0
        n_wip = len(self.pk) - self.n_completed
        self.wip_area += n_wip * dt
        self.blocked_area += sum(self.s_blk) * dt
        if dt > 0:
            for k in range(self.n):
                if not (self.s_proc[k] or self.s_blk[k]):
                    self.D[k] += self.v[k] * dt
                    self.moved[k] = True
        if self.lstate == HEADING and dt > 0:
            self._move_laser(dt)
        self.t += dt

    def _move_laser(self, dt: float) -> None:
        x0, y0 = self.lx, self.ly
        dx, dy = self.tx - x0, self.ty - y0
        if self.motion == "simultaneous":
            sx, sy = self.vx * dt, self.vy * dt
            nx = self.tx if abs(dx) <= sx + 1e-12 else x0 + math.copysign(sx, dx)
            ny = self.ty if abs(dy) <= sy + 1e-12 else y0 + math.copysign(sy, dy)
            # distance: diagonal while both axes move, then the remaining axis
            tx_, ty_ = abs(nx - x0) / self.vx, abs(ny - y0) / self.vy
            tc = min(tx_, ty_)
            d1 = path_length(self.vx * tc if tx_ > 0 else 0.0, self.vy * tc if ty_ > 0 else 0.0)
            d2 = abs(nx - x0) - (self.vx * tc if tx_ > 0 else 0.0) \
                + abs(ny - y0) - (self.vy * tc if ty_ > 0 else 0.0)
            self.dist += d1 + max(d2, 0.0)
        else:  # sequential: X first, the remaining time goes to Y
            sx = self.vx * dt
            if abs(dx) <= sx + 1e-12:
                nx = self.tx
                rem = dt - abs(dx) / self.vx
                sy = self.vy * rem
                ny = self.ty if abs(dy) <= sy + 1e-12 else y0 + math.copysign(sy, dy)
            else:
                nx, ny = x0 + math.copysign(sx, dx), y0
            self.dist += abs(nx - x0) + abs(ny - y0)
        self.lx, self.ly = nx, ny

    # ---------------------------------------------------------------- output
    def cost_so_far(self) -> tuple[float, int, float]:
        """(sum of flow so far = WIP integral, N_B, T_B)."""
        return self.wip_area, self.n_block_events, self.blocked_area

    def summary(self) -> dict:
        cfg = self.cfg
        done = [j for j in range(len(self.pk)) if self.pC[j] is not None]
        n = len(done)
        flows = [self.pC[j] - self.pr[j] for j in done]
        waits = [self.pS[j] - self.pr[j] for j in range(len(self.pk)) if self.pS[j] is not None]
        nb = sum(1 for j in range(len(self.pk)) if self.pB[j] is not None and (
            self.pS[j] is None or self.pS[j] - self.pB[j] > 1e-9))
        tb = self.blocked_area
        pen = cfg.blocking_penalty_per_event * nb + cfg.blocking_penalty_per_second * tb
        avg_flow = sum(flows) / n if n else None
        return {
            "total_simulation_time": self.t,
            "parts_generated": len(self.pk),
            "parts_completed": n,
            "avg_flow_time": avg_flow,
            "avg_waiting_time": sum(waits) / len(waits) if waits else None,
            "makespan": max((self.pC[j] for j in done), default=0.0),
            "blocking_events": nb,
            "total_blocked_time": tb,
            "blocking_penalty": pen,
            "penalized_flow_time": (avg_flow + pen / n) if n else INF,
            "total_laser_travel_distance": self.dist,
            "decisions": self.n_decisions,
        }

    def to_parts(self):
        """The episode as :class:`simulation.part.Part` objects (for the checker)."""
        from simulation.part import Part, PartStatus
        out = []
        for j in range(len(self.pk)):
            k = self.pk[j]
            p = Part(id=f"P{j + 1:03d}", conveyor_id=k + 1, arrival_time=self.pr[j],
                     processing_time=self.pp[j], due_date=self.pr[j] + self.allowance,
                     x=self.x(j), y=self.yk[k], status=PartStatus(STATUS_NAMES[self.pst[j]]))
            p.entry_time, p.ready_time = self.pentry[j], self.pready[j]
            p.start_processing_time, p.completion_time = self.pS[j], self.pC[j]
            p.exit_time, p.block_time = self.pexit[j], self.pB[j]
            p.arrival_displacement, p.entry_coordinate = self.pD0[j], self.pE[j]
            p.start_displacement, p.processing_x = self.pDS[j], self.pX[j]
            p.completion_displacement = self.pDS[j] if self.pC[j] is not None else None
            out.append(p)
        return out

    def result(self):
        """Minimal result object accepted by optimization.schedule_check.check_schedule."""
        from types import SimpleNamespace
        return SimpleNamespace(config=self.cfg, parts=self.to_parts(), summary=self.summary())
