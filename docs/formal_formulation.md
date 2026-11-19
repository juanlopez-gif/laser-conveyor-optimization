# Formal mathematical formulation

> PDF version: [docs/formulation/formal_formulation.pdf](formulation/formal_formulation.pdf) (LaTeX source alongside).

This document states the laser–conveyor scheduling problem formally. It covers:

1. an **exact continuous-time model** (the reference definition of the problem);
2. a **time-indexed MILP** that implements it up to a time discretization $\Delta$;
3. the **sequence-only (decoder) view** used by heuristics and metaheuristics;
4. the **online** version actually faced by the cell;
5. structural properties and complexity;
6. the correspondence with the simulator, which is checked automatically
   (`src/optimization/schedule_check.py`, `tests/test_formulation.py`).

The end-of-zone rule is **blocking** (confirmed for the real cell): an unprocessed part
that reaches the end of the laser zone stops its conveyor until the laser processes it.
Since version 0.4.0 every such event is **penalized** in the objective (§1.4), and the
laser may optionally **pre-position** itself for parts that have not reached the laser
zone yet (§1.5). Changes are logged in [CHANGELOG.md](CHANGELOG.md).

---

## 0. Notation

### Sets

| symbol | meaning |
|---|---|
| $K=\lbrace 1,\dots,N\rbrace$ | conveyors |
| $J$, $n=\lvert J\rvert$ | parts (jobs) of the instance |
| $J_k\subseteq J$ | parts of conveyor $k$, indexed in arrival order $j^k_1, j^k_2,\dots$ |
| $k(j)$ | conveyor of part $j$ |
| $\mathrm{pred}(j)$ | part of $J_{k(j)}$ that arrived immediately before $j$ (none for the first) |

### Parameters

| symbol | meaning | default |
|---|---|---|
| $r_j$ | arrival (release) time of part $j$ | from the arrival process |
| $p_j$ | processing time | 2.5 s |
| $d_j$ | due date ($r_j + 20$ s) | |
| $y_k$ | transverse position of conveyor $k$ | 0, 12, 24 cm |
| $v_k$ | conveyor speed | 3 cm/s |
| $a,\ b$ | start / end of the laser zone along the belt | 10, 45 cm |
| $L$ | belt length (exit) | 55 cm |
| $\delta$ | minimum spacing between consecutive parts on a belt | 4 cm |
| $v_x,\ v_y$ | laser axis speeds | 10, 10 cm/s |
| $P_0=(x_0,y_0)$ | initial laser position | workspace centre |
| $\sigma_{kk'}\in\lbrace 0,1\rbrace$ | stop rule: 1 if processing a part of conveyor $k'$ stops conveyor $k$ | $\sigma=I$ (own conveyor only) |

We assume $\sigma_{kk}=1$ for all $k$ (a part never moves while it is being processed).

### Laser travel time and path length

For two points $P=(x,y)$, $Q=(x',y')$ with $\Delta x=\lvert x'-x\rvert$, $\Delta y=\lvert y'-y\rvert$:

$$
\tau(P,Q)=\begin{cases}
\max\!\left(\dfrac{\Delta x}{v_x},\dfrac{\Delta y}{v_y}\right) & \text{simultaneous axes (default)}\\[2ex]
\dfrac{\Delta x}{v_x}+\dfrac{\Delta y}{v_y} & \text{sequential axes}
\end{cases}
$$

Both are metrics (weighted $\ell_\infty$ and $\ell_1$ norms), so $\tau$ satisfies the
triangle inequality. The **path length** travelled (for the distance objective) is

$$
\ell(P,Q)=\begin{cases}
\max\!\left(\Delta y + c_x\,\Delta x,\ \ \Delta x + c_y\,\Delta y\right) & \text{simultaneous}\\
\Delta x+\Delta y & \text{sequential}
\end{cases}
\qquad
c_x=\frac{\sqrt{v_x^2+v_y^2}-v_y}{v_x},\quad
c_y=\frac{\sqrt{v_x^2+v_y^2}-v_x}{v_y}.
$$

(With simultaneous axes the head moves diagonally until one axis arrives and then
straight; for $v_x=v_y$, $c_x=c_y=\sqrt2-1$.) Both expressions are convex in
$\Delta x$, which matters for the MILP in §2.

---

## 1. Exact continuous-time model $(\mathcal P)$

### 1.1 Belt odometer and part positions

The key modelling device is the **belt odometer** of each conveyor,

$$
D_k(t)=v_k\int_0^t u_k(s)\,ds,\qquad u_k(t)\in\lbrace 0,1\rbrace \ \ (1=\text{running}),
$$

a non-decreasing function. Every part on a belt moves rigidly with it, so a part is
fully described by one constant, its **entry coordinate** $E_j$ (the odometer reading
at the moment it is placed at $x=0$):

$$
E_j=\max\!\Big(D_{k(j)}(r_j),\ E_{\mathrm{pred}(j)}+\delta\Big)
\qquad(E_{\mathrm{pred}(j)}:=-\infty \text{ for the first part}),
$$

$$
x_j(t)=\max\!\big(0,\ D_{k(j)}(t)-E_j\big)\qquad (t\ge r_j).
$$

The $\max$ in $E_j$ is the upstream queue: if the previous part has not yet moved
$\delta$ away from the entry, part $j$ waits (state WAITING) until the odometer reaches
$E_{\mathrm{pred}(j)}+\delta$. The entry time is $e_j=\min\lbrace t\ge r_j: D_{k(j)}(t)\ge E_j\rbrace$.

### 1.2 Decision variables

* $\pi$: a permutation of $J$ (processing order of the laser);
* $S_j\ge 0$: processing start time of part $j$, with $C_j=S_j+p_j$;
* $u_k(\cdot)$: belt control (fully determined by the other decisions in the
  non-idling variant, a decision in the relaxed variant).

Derived quantities: processing position $X_j=D_{k(j)}(S_j)-E_j$ and processing point
$P_j=(X_j,\,y_{k(j)})$.

### 1.3 Constraints

**(C1) Reachability window.** A part can only be processed inside the laser zone:

$$
a\ \le\ X_j\ \le\ b\qquad\forall j\in J.
$$

**(C2) One part at a time, with travel (interception).** For consecutive parts of $\pi$:

$$
S_{\pi(1)}\ \ge\ \tau\big(P_0,P_{\pi(1)}\big),\qquad
S_{\pi(i+1)}\ \ge\ C_{\pi(i)}+\tau\big(P_{\pi(i)},P_{\pi(i+1)}\big)\quad i=1,\dots,n-1.
$$

Because $P_{\pi(i+1)}$ depends on $S_{\pi(i+1)}$ through $D_k$, this inequality is
*implicit*: it says that the laser, leaving $P_{\pi(i)}$ at $C_{\pi(i)}$, reaches the
point where the moving part will be at time $S_{\pi(i+1)}$ no later than the part
(it may wait there). This is exactly the interception logic of the simulator. Since
$\tau$ satisfies the triangle inequality, (C2) implies
$S_j\ge C_i+\tau(P_i,P_j)$ for **every** pair with $i$ before $j$, so the processing
intervals are disjoint.

**(C3) Belt stops during processing.**

$$
u_k(t)=0\qquad\forall t\in[S_j,C_j),\ \forall j\in J \text{ with } \sigma_{k\,k(j)}=1 .
$$

In particular the part stays at $X_j$ during its processing.

**(C4) Blocking (no overrun).** An unprocessed part can never be carried beyond $b$:
$D_{k(j)}(t)-E_j\le b$ for all $t\le S_j$. Since $D_k$ is non-decreasing this is
equivalent to $X_j\le b$, i.e. it is already the right-hand side of (C1). The conveyor
therefore *must* stop when its most downstream unprocessed part reaches $b$.

**(C5) Non-idling conveyors** (variant $\mathcal P$; the conveyors "normally run"). A
conveyor stops **only** when (C3) or (C4) forces it:

$$
u_k(t)=0\ \Longrightarrow\
\Big(\exists j:\ \sigma_{k\,k(j)}=1,\ t\in[S_j,C_j)\Big)\ \ \lor\ \
\Big(\exists j\in J_k:\ S_j>t,\ D_k(t)-E_j=b\Big).
$$

Dropping (C5) gives the **relaxed variant** $\mathcal P^{R}$, where the controller may
also stop a conveyor voluntarily (e.g. to keep a part inside the laser zone).

**(C6) Workspace.** $P_j\in W=[a,b]\times[\min_k y_k,\max_k y_k]$ follows from (C1);
straight moves between points of the convex set $W$ stay in $W$.

### 1.4 Objectives (minimized)

| objective | expression |
|---|---|
| makespan | $C_{\max}=\max_j C_j$ |
| total / mean flow time | $\sum_j (C_j-r_j)$ |
| total / mean waiting time | $\sum_j (S_j-r_j)$ |
| maximum lateness | $L_{\max}=\max_j (C_j-d_j)$ |
| laser travel distance | $\sum_{i=1}^{n-1}\ell(P_{\pi(i)},P_{\pi(i+1)})+\ell(P_0,P_{\pi(1)})$ |
| conveyor downtime | $\sum_k\int_0^{T}(1-u_k(t))\,dt=\sum_k\big(T-D_k(T)/v_k\big)$ |
| throughput (finite instance) | $n/C_{\max}$ — maximizing it is equivalent to minimizing $C_{\max}$ |

$T$ is the time at which the last part exits ($x_j=L$).

#### Blocking penalty (version 0.4.0)

A part that reaches the end of the laser zone unprocessed stops its **whole conveyor**
(C4)–(C5): every part behind it stops too and no new part can enter that belt. The
time losses this causes are already contained in $C_j$ and $F_j$, but in the cell a
blocked conveyor has additional costs that the time objectives do not capture
(restart, accumulation upstream, risk for the upstream process). It is therefore
penalized explicitly. For part $j$ on conveyor $k=k(j)$ define the time it would reach
the zone end,

$$
B_j=\min\lbrace t:\ D_k(t)-E_j\ \ge\ b\rbrace ,
$$

which is well defined because, by (C4), the belt stops exactly when $x_j=b$. Part $j$
**blocks** its conveyor iff it gets there before the laser starts on it:

$$
\beta_j=\mathbb 1\big[S_j>B_j\big],\qquad \text{blocked duration } (S_j-B_j)^+ .
$$

On each conveyor at most one part (the most downstream unstarted one) can sit at $b$,
and a block lasts from $B_j$ to $S_j$, so the blocking intervals of a conveyor are
disjoint. The blocking indicators are

$$
N_B=\sum_j \beta_j\qquad\text{(number of blocking events)},\qquad
T_B=\sum_j (S_j-B_j)^+=\sum_k\int_0^T \mathbb 1\big[\text{conveyor }k\text{ blocked at }t\big]\,dt
\quad\text{(conveyor-seconds)} ,
$$

and the **blocking penalty** is

$$
\Pi=\lambda_N\,N_B+\lambda_T\,T_B,\qquad \lambda_N,\lambda_T\ge 0 ,
$$

with $\lambda_N$ in seconds per event and $\lambda_T$ in seconds per blocked
conveyor-second (`objective.blocking_penalty.per_event` / `per_second`; defaults 10 and
1, placeholders until the real cost of a stopped conveyor is known). Two penalized
objectives are defined:

$$
\overline F_\Pi=\frac1n\Big(\sum_j F_j+\Pi\Big)\quad(\texttt{penalized\_flow\_time}),\qquad
C_{\max}+\Pi\quad(\texttt{penalized\_makespan}).
$$

$\overline F_\Pi$ is the total cost (flow time plus penalty, both in seconds) per part,
and it is the default objective of the code since 0.4.0 (it replaced $C_{\max}$, which
is almost fixed by the arrivals at moderate load and does not discriminate policies).
Since $\Pi\ge 0$, every lower bound of $\overline F$ (resp. $C_{\max}$) is also a lower
bound of $\overline F_\Pi$ (resp. $C_{\max}+\Pi$). The composite objective of the code is
the weighted sum of any of these criteria (`objective.weights`), e.g.

$$
\min\ \ \alpha\,C_{\max}+\beta\,\textstyle\sum\ell+\gamma\sum_k\int(1-u_k)+\delta_W\,\overline W+\eta\,\overline F_\Pi .
$$

Multi-objective studies use the Pareto front of these criteria.

### 1.5 Pre-positioning and the restricted model $\mathcal P^{\mathrm{sim}}$

(C2) allows the laser to move towards a part **before** it reaches the laser zone (it
may pre-position at $(a, y_k)$ and wait). Up to version 0.3 the simulator and every
implemented policy only dispatched the laser to parts that are already READY; this is
still the default (`laser.prepositioning: none`). Let
$R_j=\min\lbrace t: x_j(t)\ge a\rbrace$ be the time part $j$ enters the laser zone. The
simulator satisfies, in addition,

$$
\textbf{(C7)}\qquad S_{\pi(i+1)}\ \ge\ \max\big(C_{\pi(i)},\ R_{\pi(i+1)}\big)+\tau\big(P_{\pi(i)},P_{\pi(i+1)}\big)
$$

(and $S_{\pi(1)}\ge R_{\pi(1)}+\tau(P_0,P_{\pi(1)})$). We write
$\mathcal P^{\mathrm{sim}}=\mathcal P+\text{(C7)}$. Since (C7) only removes
schedules, $\mathcal P$ is a **relaxation** of $\mathcal P^{\mathrm{sim}}$: its optimum is
a lower bound for everything the simulator can do, and the gap between the two measures
the value of pre-positioning the laser. Example (counterexample D1 in
`docs/methods/theory.md`): a single part arriving at $t=0$ completes at 5.83 s in
$\mathcal P$ and at 7.19 s in $\mathcal P^{\mathrm{sim}}$. Both models are solved by the
CP-SAT methods `cpsat` ($\mathcal P_\Delta$) and `cpsat_ready`
($\mathcal P^{\mathrm{sim}}_\Delta$), see `docs/methods/exact_cpsat.md`.

**Pre-positioning in the simulator (version 0.4.0).** The option
`laser.prepositioning` selects which model the simulator realises:

| mode | candidates when the laser is free | model realised |
|---|---|---|
| `none` (default) | READY parts ($x_j\ge a$) | $\mathcal P^{\mathrm{sim}}=\mathcal P+$(C7) |
| `always` | READY parts and parts in the entry zone of a running belt | $\mathcal P$ (no (C7)) |
| `when_idle` | entry-zone parts only if no part is READY; the assignment is revoked as soon as a part becomes READY | between both: feasible for $\mathcal P$, not necessarily for (C7) |

For a part in the entry zone the meeting point is searched only for $x\ge a$: the
laser goes to $(\max(a,\hat x),y_k)$ and waits there, i.e. it starts processing at
$S_j=\max\big(R_j,\ t+\tau\big)$ or later along the belt
(`motion.find_intercept(x_min=a)`). With `always` and the `fixed_sequence` policy the
simulator is the decoder of $\mathcal P$ (§3): for the single part of the example above
it gives 5.84 s (5.83 s in $\mathcal P$, up to `dt`). Physically, pre-positioning needs
to know that a part is in the entry zone (an entry sensor); whether the real cell has
it is open. Revocable pre-positioning (`when_idle`) never commits the laser to a future
part while another part is already waiting, which is what makes it useful for myopic
policies (§4, `docs/methods/nearest_head_prepositioning.md`).

### 1.6 Problem statement

$$
(\mathcal P)\qquad \min_{\pi,\,S,\,u}\ f(\pi,S,u)\quad\text{s.t.}\quad \text{(C1)–(C5)},\ \ D_k,\ E_j,\ X_j \text{ defined as in §1.1–1.2.}
$$

---

## 2. Time-indexed MILP $(\mathcal P_\Delta)$

Discretize time in steps of length $\Delta$ (e.g. 0.1 s), $t\in\mathcal T=\lbrace 0,\dots,H-1\rbrace$,
and assume $p_j=\Delta\,\bar p_j$ and $r_j=\Delta\,\bar r_j$ with integer
$\bar p_j,\bar r_j$. $M$ denotes a sufficiently large constant (each big-$M$ can be
tightened individually, e.g. to $L+v_k H\Delta$ for positions).

### Variables

| variable | domain | meaning |
|---|---|---|
| $s_{jt}$ | $\lbrace 0,1\rbrace$ | processing of $j$ starts at step $t$ |
| $u_{kt}$ | $\lbrace 0,1\rbrace$ | conveyor $k$ runs during step $t$ |
| $D_{kt}$ | $\ge 0$ | odometer of conveyor $k$ at the start of step $t$ |
| $E_j,\ X_j$ | $\mathbb R$ | entry coordinate, processing position |
| $\lambda_j$ | $\lbrace 0,1\rbrace$ | which term of the $\max$ defines $E_j$ |
| $w_{ij}$ | $\lbrace 0,1\rbrace$ | $i$ is processed before $j$ ($i\ne j$) |
| $\tau_{ij},\ \tau_{0j}$ | $\ge 0$ | travel times |
| $\beta_{jt}$ | $\lbrace 0,1\rbrace$ | part $j$ is blocking its conveyor at step $t$ (only for (C5)) |

$S_j:=\Delta\sum_t t\,s_{jt}$ and $C_j:=S_j+p_j$ are linear expressions.

### Constraints

$$
\begin{aligned}
&\textstyle\sum_{t\in\mathcal T} s_{jt}=1 && \forall j &&\text{(M1) one start per part}\\
&D_{k0}=0,\qquad D_{k,t+1}=D_{kt}+v_k\Delta\,u_{kt} && \forall k,t &&\text{(M2) odometer}\\
&E_j\ge D_{k(j),\bar r_j},\qquad E_j\ge E_{\mathrm{pred}(j)}+\delta && \forall j &&\text{(M3) entry}\\
&E_j\le D_{k(j),\bar r_j}+M\lambda_j,\qquad E_j\le E_{\mathrm{pred}(j)}+\delta+M(1-\lambda_j) && \forall j\\
&\lvert X_j-(D_{k(j),t}-E_j)\rvert\le M(1-s_{jt}) && \forall j,t &&\text{(M4) position at start}\\
&a\le X_j\le b && \forall j &&\text{(M5) = (C1) + (C4)}\\
&u_{kt}+\textstyle\sum_{j:\ \sigma_{k\,k(j)}=1}\ \sum_{t'=t-\bar p_j+1}^{t} s_{jt'}\le 1 && \forall k,t &&\text{(M6) = (C3)}\\
&w_{ij}+w_{ji}=1 && \forall i<j &&\text{(M7) order}\\
&S_j\ge S_i+p_i+\tau_{ij}-M(1-w_{ij}) && \forall i\ne j &&\text{(M8) disjunctive + travel}\\
&S_j\ge \tau_{0j} && \forall j &&\text{(M9) first move}
\end{aligned}
$$

Travel times, simultaneous axes (sequential: replace by
$\tau_{ij}\ge \xi_{ij}/v_x+\lvert y_{k(i)}-y_{k(j)}\rvert/v_y$ with
$\xi_{ij}\ge\pm(X_j-X_i)$):

$$
\tau_{ij}\ge \frac{X_j-X_i}{v_x},\quad \tau_{ij}\ge \frac{X_i-X_j}{v_x},\quad
\tau_{ij}\ge\frac{\lvert y_{k(i)}-y_{k(j)}\rvert}{v_y},\qquad
\tau_{0j}\ge\frac{\pm(X_j-x_0)}{v_x},\ \ \tau_{0j}\ge\frac{\lvert y_{k(j)}-y_0\rvert}{v_y}.
$$

Only lower bounds are needed: a larger $\tau$ only tightens (M8)–(M9), so at an optimum
(or in any feasible solution, after lowering it) $\tau$ equals the true travel time.
(M8) is imposed for all ordered pairs; by the triangle inequality this is equivalent to
imposing it only between consecutive parts.

**Non-idling conveyors (C5)**, omitted for $\mathcal P^R_\Delta$:

$$
\begin{aligned}
&u_{kt}\ \ge\ 1-\textstyle\sum_{j:\sigma_{k\,k(j)}=1}\sum_{t'=t-\bar p_j+1}^{t}s_{jt'}-\sum_{j\in J_k}\beta_{jt} && \forall k,t\\
&\beta_{jt}\ \le\ 1-\textstyle\sum_{t'\le t}s_{jt'} && \forall j,t \quad(\text{not started yet})\\
&D_{k(j),t}-E_j\ \ge\ b-v_{k(j)}\Delta-M(1-\beta_{jt}) && \forall j,t\quad(\text{at the zone end})
\end{aligned}
$$

The slack $v_k\Delta$ is the discretization tolerance: a conveyor may stop one step
early instead of overshooting $b$ (the simulator stops exactly at $b$ with a partial
step).

### Objectives (linear forms)

* $C_{\max}\ge C_j\ \forall j$; $\ L_{\max}\ge C_j-d_j\ \forall j$;
* flow / waiting: $\sum_j (C_j-r_j)$, $\sum_j (S_j-r_j)$;
* downtime: $\Delta\sum_{k,t}(1-u_{kt})$;
* blocking (0.4.0): with the variables $\beta_{jt}$ of (C5),
  $T_B=\Delta\sum_{j,t}\beta_{jt}$ and $N_B=\sum_j\gamma_j$ with
  $\gamma_j\in\lbrace 0,1\rbrace$, $\gamma_j\ge\beta_{jt}\ \forall t$; penalty
  $\Pi=\lambda_N N_B+\lambda_T T_B$. Under (C5) $\beta_{jt}$ is forced to 1 exactly when
  the belt must stop for $j$ (otherwise $u_{kt}=1$ would carry $j$ beyond $b$,
  contradicting (M4)–(M5)); minimizing $\Pi$ keeps it at 0 otherwise. The CP-SAT
  implementation (`solvers/cpsat.py`) does not include these terms yet: for the
  penalized objectives it is skipped and the simulation-based methods are used.
* travel distance: add immediate-successor binaries $z_{ij}$ (with a dummy start $0$):
  $\sum_i z_{ij}=1$, $\sum_j z_{ij}\le 1$, $z_{ij}\le w_{ij}$,
  $w_{il}+w_{lj}\le 2-z_{ij}\ \forall l$ (nothing between $i$ and $j$), and
  $$
  \lambda^{\ell}_{ij}\ge \lvert y_{k(i)}-y_{k(j)}\rvert+c_x\,\xi_{ij}-M(1-z_{ij}),\qquad
  \lambda^{\ell}_{ij}\ge \xi_{ij}+c_y\lvert y_{k(i)}-y_{k(j)}\rvert-M(1-z_{ij}),
  $$
  $\xi_{ij}\ge\pm(X_j-X_i)$, objective $\sum_{i,j}\lambda^{\ell}_{ij}$. This is exact
  because $\ell$ is convex in $\Delta x$ (§0).

### Size and practical notes

$O(nH)$ binaries $s_{jt}$, $O(NH)$ binaries $u_{kt}$, $O(n^2)$ order binaries. With
$\Delta=0.1$ s and a 60 s window this is a few thousand binaries for 10–15 parts:
solvable with CP-SAT / HiGHS for small instances and inside a rolling horizon. In
CP-SAT, (M7)–(M8) are naturally written with interval variables and `NoOverlap`,
and (M4) with `AddElement` on the odometer array.

---

## 3. Sequence-only view (decoder)

Under (C5), once the order $\pi$ is fixed, the **earliest** schedule compatible with
(C1)–(C5) is computed by a deterministic forward pass: the laser leaves $P_{\pi(i)}$
at $C_{\pi(i)}$, heads to the earliest interception point of $\pi(i+1)$ (the smallest
$t$ with $\tau(P_{\pi(i)},(x_{\pi(i+1)}(t),y))\le t-C_{\pi(i)}$, recomputed whenever a
conveyor changes state), and the belts evolve by (C3)–(C5). In the simulator the
laser additionally starts towards $\pi(i+1)$ only once that part is READY, i.e. the
pass respects (C7). This map $\Phi:\pi\mapsto(S,u)$ — the earliest schedule of
$\mathcal P^{\mathrm{sim}}$ for the order $\pi$ — is exactly what the simulator computes
with the `fixed_sequence` policy, so

$$
\min_{\pi\in\mathfrak S_n} f\big(\Phi(\pi)\big)
$$

is a permutation problem that genetic algorithms, simulated annealing, tabu search or
exhaustive search (small $n$) can attack with the simulator as evaluator. Caveat:
because positions depend on start times, delaying a start can move a part closer to
the laser; the earliest (semi-active) schedule is therefore not guaranteed to dominate
for every objective. $\Phi$ is therefore a heuristic restriction of
$\mathcal P^{\mathrm{sim}}$ (and of $\mathcal P$), while $\mathcal P_\Delta$ explores all
start times and pre-positioning.

A dynamic program over subsets $(\text{processed set},\ \text{last part})$ is **not**
exact here: the belt odometers $D_k$ depend on the whole history, so the state would
also have to include them.

---

## 4. Online version

In the cell, $r_j$ (and possibly $p_j$) are only revealed when the part arrives. A
**policy** maps the information available at a decision epoch (laser free) to the next
part. The offline problem $\mathcal P$ with full knowledge of the future gives the
*clairvoyant* benchmark $f^\ast$; a policy $\mu$ is evaluated by its expected
performance over the arrival distribution and by the empirical ratio
$\mathbb E[f(\mu)]/\mathbb E[f^\ast]$ on generated instances. With the default
`laser.prepositioning: none` the implemented policies satisfy (C7) — the laser only
selects parts that are already in the laser zone — and never wait deliberately. Hence
they produce feasible solutions of $\mathcal P^{\mathrm{sim}}\subseteq\mathcal P$; with
pre-positioning (`when_idle`, `always`) they produce feasible solutions of
$\mathcal P$. In every case the optimum of $\mathcal P$ (or the valid bounds of
`optimization/lower_bounds.py`) is a lower bound for all of them.

**Belt-order policy (`nearest_head`, 0.4.0).** By property 2 of §5, processing a
conveyor out of belt order makes the earlier part travel on towards $b$ and block. The
policy `nearest_head` only considers the *head* of every conveyor (its most downstream
unstarted part) and picks the head with the smallest interception time. Its schedules
keep belt order inside every conveyor, which is the structure of the good sequences
found by the metaheuristics.

**Reinforcement learning (0.5.0).** The online problem is a semi-Markov decision
process: decision epochs are the instants at which the laser is free and at least two
conveyors have a dispatch candidate; the sojourn time between epochs is random. The
action is the conveyor $k$ whose head is served (actions without candidates are
masked). The cost accrued between epochs $t$ and $t'$ is

$$
c(t,t')=\int_t^{t'} \lvert\lbrace j:\ r_j\le s<C_j\rbrace\rvert\,ds+\lambda_T\,T_B(t,t')+\lambda_N\,N_B(t,t'),
$$

and since $\int_0^\infty \lvert\lbrace j: r_j\le s<C_j\rbrace\rvert\,ds=\sum_j F_j$
(Little), the undiscounted return of an episode is exactly
$-(\sum_j F_j+\Pi)=-n\,\overline F_\Pi$. See `docs/methods/rl.md`.

**Rolling horizon:** at each decision epoch, solve $\mathcal P_\Delta$ restricted to
the parts present (plus forecast arrivals, if any) over a window of $H$ steps, apply
the first decision and re-solve at the next epoch.

---

## 5. Structural properties and complexity

1. **Rigid belts.** On each conveyor the relative order of parts never changes and the
   distance between two parts is fixed after both have entered:
   $x_i(t)-x_j(t)=E_j-E_i\ge\delta$ for $i$ before $j$. The laser, however, may process
   them in any order.
2. **Blocking cascades.** By (C4), on each conveyor the binding cap at time $t$ is the
   most downstream unstarted part: $D_k(t)\le \min\lbrace E_j+b:\ j\in J_k,\ S_j>t\rbrace$.
   Processing parts of a conveyor out of belt order therefore causes blocking.
3. **Interception is always possible in blocking mode.** If $v_k\ge v_x$ the laser may
   not catch a part ahead of it while it moves, but by (C4) the part stops at $b$, so
   every part is eventually reachable and $\mathcal P$ is always feasible.
4. **Lower bounds.** For every instance,
   $C_{\max}\ \ge\ \max_j\big(r_j + a/v_{k(j)} + p_j\big)$ and
   $C_{\max}\ \ge\ \min_j\big(r_j+a/v_{k(j)}\big)+\sum_j p_j$. Adding a lower bound on
   the travel between consecutive processing points (e.g. $\lvert y_k-y_{k'}\rvert/v_y$
   for changes of conveyor) strengthens the second one.
5. **Complexity.** $\mathcal P$ contains, as limiting cases, classical NP-hard
   problems. With travel negligible ($v_x,v_y\to\infty$) and a laser zone long enough
   that blocking never binds, and with one part per conveyor (so that the belt stops of
   (C3) do not delay other parts), part $j$ becomes available at $r_j+a/v_{k(j)}$ and
   (C1)–(C3) reduce to $1\mid r_j\mid L_{\max}$ and $1\mid r_j\mid\sum C_j$, both
   strongly NP-hard (Lenstra, Rinnooy Kan & Brucker, 1977). With all parts already in
   the zone and conveyors stopped, the makespan version is a shortest Hamiltonian path
   in the metric $\tau$ (a TSP-path). Hence exact methods are realistic only for small
   instances or inside a rolling horizon, and heuristics are needed for the full
   dynamic problem.

Classification (Graham-style):
$1\mid r_j,\ s_{ij}(t),\ \text{blocking},\ \text{time windows}\mid C_{\max},\ \sum F_j,\ L_{\max}$,
where the travel "setup" $s_{ij}(t)$ is sequence- **and** time-dependent.

---

## 6. Correspondence with the simulator (validated)

| model element | simulator |
|---|---|
| odometer $D_k(t)$ | `Conveyor.displacement` |
| $E_j$, $D_k(r_j)$, $D_k(S_j)$, $D_k(C_j)$, $X_j$ | `Part.entry_coordinate`, `arrival_displacement`, `start_displacement`, `completion_displacement`, `processing_x` (exported in `parts.csv`) |
| upstream queue / spacing (§1.1) | `Conveyor.can_admit` / `admit` |
| (C1) window | candidates are READY parts (plus entry-zone parts with pre-positioning); processing starts only in contact inside $[a,b]$ |
| (C7) / pre-positioning | `laser.prepositioning` (`none` = (C7), `always` = $\mathcal P$, `when_idle`), `Simulation._dispatch`, `_revoke_prepositioning` |
| (C2) interception | `motion.find_intercept` + re-planning in `Simulation._drive_laser` |
| (C3) stop rule $\sigma$ | `stop_policy.py` (`processing_conveyor_only` = $I$, `all_conveyors` = all ones) |
| (C4)–(C5) blocking, non-idling | `Conveyor.advance(block_unprocessed=True)` and `_update_blocking` |
| $\tau$, $\ell$ | `motion.travel_time`; travelled distance accumulated in `Laser.move_towards` |
| $B_j$, $\beta_j$, $N_B$, $T_B$, $\Pi$ | `Part.block_time`, `Part.caused_block`, `blocking_events`, `total_blocked_time`, `blocking_penalty` (summary); $T_B=\sum_j(S_j-B_j)^+$ is checked by `tests/test_blocking_prepositioning.py` |
| objectives | `metrics.summarize`, `optimization/objective_functions.py` |

`src/optimization/schedule_check.py` takes a simulated run and verifies the identities
and constraints of §1 — entry coordinates (§1.1), $X_j=D_k(S_j)-E_j$, (C1), (C2) for
every consecutive pair, (C3) through $D_k(C_j)=D_k(S_j)$ and the belt-kinematics
bound $D_k(S_j)-D_k(S_i)\le v_k\cdot(\text{running time})$ — with tolerances of the
order of the simulator step `dt`. `tests/test_formulation.py` runs it for every policy,
for stochastic instances and for the `all_conveyors` stop rule. Any solution produced
by a policy is therefore a feasible solution of $\mathcal P$ (up to `dt`), and an
optimal solution of $\mathcal P$ (or $\mathcal P_\Delta$) is a valid benchmark for the
policies.

---

## 7. Results obtained with this formulation

The four specialist methods built on this model are documented in `docs/methods/`
(see `docs/agent_architecture.md` for how they were developed and integrated):

| document | content |
|---|---|
| [theory.md](methods/theory.md) | valid lower bounds (L1–L6) with proofs; special cases (single-conveyor counterexample, light-traffic optimality theorems, when FIFO / nearest job fail, decoder vs $\mathcal P$) |
| [exact_cpsat.md](methods/exact_cpsat.md) | CP-SAT models of $\mathcal P_\Delta$ and $\mathcal P^{\mathrm{sim}}_\Delta$; proven optima for $n\le 8$ |
| [metaheuristics.md](methods/metaheuristics.md) | simulated annealing and iterated local search over $\Phi$ |
| [online_rollout.md](methods/online_rollout.md) | rollout (look-ahead) policy for the online problem |
| [nearest_head_prepositioning.md](methods/nearest_head_prepositioning.md) | version 0.4.0: blocking penalty, belt-order policy `nearest_head`, pre-positioning modes; 30-seed study |
