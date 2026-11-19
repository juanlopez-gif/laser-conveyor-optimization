# Theory: valid lower bounds and special cases

Owner: THEORY agent. Code: [`src/optimization/lower_bounds.py`](../../src/optimization/lower_bounds.py),
tests: [`tests/test_lower_bounds.py`](../../tests/test_lower_bounds.py).
Model and notation: [`docs/formal_formulation.md`](../formal_formulation.md) (§0–§1).

Every statement is labelled **Theorem / Proposition / Lemma** (proved here),
**Conjecture** (supported by computational evidence, the amount is stated) or
**Counterexample** (an explicit instance, reproduced by a test).

All bounds are valid for **every feasible schedule of the exact model $\mathcal P$**
(constraints (C1)–(C5)), hence also for the relaxed variant $\mathcal P^R$ (none of
the proofs uses (C5)), for every policy and for the decoder $\Phi$; they bound the
time-indexed model $\mathcal P_\Delta$ to the extent that its solutions are feasible for
$\mathcal P$ (the $\Delta$-rounding of positions is not analysed here). The simulator
integrates with a step dt, so simulated values are exact only up to $O(dt)$; the tests
compare with a tolerance of $2\,dt$. The only model facts used are:

* (F1) $D_k$ is non-decreasing and $D_k(t)-D_k(s)=v_k\,\lvert\lbrace\theta\in[s,t):u_k(\theta)=1\rbrace\rvert\le v_k(t-s)$;
* (F2) $E_j\ge D_{k(j)}(r_j)$ and $E_j\ge E_{\mathrm{pred}(j)}+\delta$;
* (C1) $a\le X_j=D_{k(j)}(S_j)-E_j\le b$, (C2) with its pairwise consequence
  $S_j\ge C_i+\tau(P_i,P_j)$ for every $i$ processed before $j$, (C3) with $\sigma_{kk}=1$,
  (C4);
* $\tau(P,Q)\ge \lvert x-x'\rvert/v_x$ and $\tau(P,Q)\ge\lvert y-y'\rvert/v_y$ (both motion models).

All parameters are read from `instance_config(instance)`: $a$ = `geometry.active_start`,
$b$ = `geometry.active_end`, $y_k, v_k$ = `conveyors`, $\delta$ =
`simulation.min_part_spacing`, $v_x,v_y$, motion model and $P_0$ = `laser`, the stop rule
$\sigma$ = `simulation.conveyor_stop_policy` and the blocking flag.

## 0. Additional notation

For part $j$ on conveyor $k=k(j)$, $m_j\in\lbrace 0,1,\dots\rbrace$ is its index in belt (arrival)
order on $k$; $h\preceq_k j$ means $h\in J_k$, $m_h\le m_j$. Since $\mathrm{pred}$ follows
arrival order, $h\preceq_k j\Rightarrow r_h\le r_j$. Iterating (F2):

$$
E_j\ \ge\ E_h+(m_j-m_h)\,\delta\ \ge\ D_k(r_h)+(m_j-m_h)\,\delta\qquad (h\preceq_k j). \tag{0.1}
$$

Minimum first move: $\tau^0_j=\min_{X\in[a,b]}\tau\big(P_0,(X,y_{k(j)})\big)$, i.e.
$\max(d_x/v_x,\lvert y_k-y_0\rvert/v_y)$ (simultaneous) or the sum (sequential), with
$d_x=\mathrm{dist}(x_0,[a,b])$.

Forced predecessors (blocking): $F_j=\lbrace i\in J_k: m_i<m_j,\ a+(m_j-m_i)\delta>b\rbrace$
(empty in `pass` mode).

---

## 1. Lower bounds

### Lemma L1 (earliest start)

For every feasible schedule and every part $j$, $S_j\ge e_j$, where

$$
e_j=\max\Big\lbrace \tau^0_j,\ \max_{h\preceq_k j}\Big[r_h+\frac{a+(m_j-m_h)\delta}{v_k}+\sum_{i\in F_j,\ m_i\ge m_h}p_i\Big],\ \max_{i\in F_j}\big(e_i+p_i+s_{ij}\big)\Big\rbrace
$$

($s_{ij}$ from Lemma L2).

*Proof.* (i) $S_j\ge\tau^0_j$: let $i_1$ be the first processed part; by (C2) and the triangle
inequality $S_j\ge \tau(P_0,P_{i_1})+p_{i_1}+\tau(P_{i_1},P_j)\ge\tau(P_0,P_j)\ge\tau^0_j$
(for $j=i_1$ directly), because $P_j=(X_j,y_k)$ with $X_j\in[a,b]$.

(ii) Forced predecessors complete before $S_j$: for $i\in F_j$, by (0.1) and (C1)
$D_k(S_j)-E_i\ge D_k(S_j)-E_j+(m_j-m_i)\delta\ge a+(m_j-m_i)\delta>b$. If $S_i\ge S_j$, (C4)
($D_k(t)-E_i\le b$ for $t\le S_i$) fails at $t=S_j$. So $S_i<S_j$ and, the processing
intervals being disjoint, $C_i\le S_j$; with Lemma L2, $S_j\ge C_i+s_{ij}\ge e_i+p_i+s_{ij}$.

(iii) Belt kinematics. Fix $h\preceq_k j$. By (C1) and (0.1),
$D_k(S_j)-D_k(r_h)\ge X_j+(m_j-m_h)\delta\ge a+(m_j-m_h)\delta$. For $i\in F_j$ with
$m_i\ge m_h$: $S_i\ge r_i\ge r_h$ and $C_i\le S_j$, so $[S_i,C_i)\subset[r_h,S_j)$ and, by
(C3) with $\sigma_{kk}=1$, $u_k=0$ there; these intervals are pairwise disjoint. By (F1),
$v_k\big(S_j-r_h-\sum p_i\big)\ge D_k(S_j)-D_k(r_h)$, which is the claim. $\square$

With $h=j$ this gives the trivial availability $e_j\ge r_j+a/v_k$; with $h$ = the first part
of a burst it captures the upstream queue: the $m$-th part of a conveyor cannot be
processed before the belt has moved $a+(m-1)\delta$ after the arrival of the first one.

### Lemma L2 (minimum gap between two processings)

If $i$ is processed before $j$ (not necessarily consecutively), $S_j-C_i\ge s_{ij}$ with

$$
s_{ij}=\begin{cases}
\lvert y_{k(i)}-y_{k(j)}\rvert/v_y & k(i)\ne k(j)\\[0.5ex]
(m_j-m_i)\,\delta/(v_x+v_k) & k(i)=k(j)=k,\ m_j>m_i\ (j\text{ behind } i)\\[0.5ex]
(m_i-m_j)\,\delta/v_x & k(i)=k(j)=k,\ m_j<m_i\ (j\text{ ahead of } i).
\end{cases}
$$

*Proof.* Different conveyors: (C2) and $\tau\ge\lvert\Delta y\rvert/v_y$. Same conveyor:
let $g=S_j-C_i$ and $z_j(t)=D_k(t)-E_j$, non-decreasing with slope $\le v_k$ (F1), with
$X_j=z_j(S_j)$. By (C3) $D_k(C_i)=D_k(S_i)$, so $z_j(C_i)=X_i+E_i-E_j$. From (C2),
$\lvert X_j-X_i\rvert\le v_x g$. If $j$ is behind, $E_j-E_i\ge(m_j-m_i)\delta=:d$ by (0.1),
so $X_j\le z_j(C_i)+v_k g\le X_i-d+v_k g$ and $X_j\ge X_i-v_x g$, hence $d\le(v_x+v_k)g$.
If $j$ is ahead, $E_i-E_j\ge(m_i-m_j)\delta=:d$ and $X_j\ge z_j(C_i)\ge X_i+d$, so
$d\le v_x g$. $\square$

### Lemma H (gaps along a whole sequence)

For a set $U$ with $n_k$ parts on conveyor $k$, $K'$ conveyors present, span
$Y=\max y_k-\min y_k$ and minimum distance $g_{\min}=\min_{k\ne k'}\lvert y_k-y_{k'}\rvert$
over the present conveyors, $c_k=\delta/(v_x+v_k)$, $c_{\max}=\max_k c_k$: in any processing
order of $U$, the sum of the $\lvert U\rvert-1$ gaps between consecutive $U$-parts is at least

$$
H(U)=\min_{T=K'-1}^{\lvert U\rvert-1}\Big[\sum_k (n_k-1)c_k-(T-K'+1)\,c_{\max}+\frac{\max(Y,\ T g_{\min})}{v_y}\Big].
$$

*Proof.* Let $T$ be the number of changes of conveyor and $w_k\ge1$ the number of maximal
runs on conveyor $k$, $\sum_k w_k=T+1$. Within-conveyor transitions: $n_k-w_k$ on $k$, each
$\ge c_k$ (L2, $\lvert\Delta m\rvert\ge1$); their total is
$\sum_k(n_k-1)c_k-\sum_k(w_k-1)c_k\ge\sum_k(n_k-1)c_k-(T-K'+1)c_{\max}$. Changes of conveyor:
each $\ge g_{\min}/v_y$, and together they visit every present conveyor, so their $y$-length
is $\ge Y$. Minimizing over the unknown $T$ gives $H$. $\square$

### Proposition L3 (Jackson tail bound with travel)

For every $t$, with $U_t=\lbrace j:e_j\ge t\rbrace$:
$\ C_{\max}\ \ge\ t+\sum_{j\in U_t}p_j+H(U_t).$

*Proof.* List $U_t$ in processing order $u_1,\dots,u_m$. Then
$C_{u_m}=S_{u_1}+\sum_{U_t}p+\sum_{l<m}(S_{u_{l+1}}-C_{u_l})$, $S_{u_1}\ge e_{u_1}\ge t$ (L1), and
every difference is $\ge s_{u_l u_{l+1}}$ (L2, pairwise version), whose sum is $\ge H(U_t)$. $\square$

It is applied to $t\in\lbrace e_j\rbrace$; without $H$ it is the optimum of the relaxation
$1\mid r_j{=}e_j\mid C_{\max}$, which already dominates both bounds of formulation §5.4
(`benchmark.trivial_lower_bound`); the code also keeps the trivial bound in the maximum.

### Proposition L4 (per-conveyor bounds)

(a) For each conveyor $k$ with last part $\ell$ and each $h\in J_k$:
$\ C_{\max}\ge r_h+\big(a+(m_\ell-m_h)\delta\big)/v_k+\sum_{i\in J_k,\ m_i\ge m_h}p_i.$

(b) Let $C_{(l)}^k$ be the $l$-th smallest completion time among the parts of $J_k$
(indexed $1..n_k$ in belt order). Then
$C_{(l)}^k\ge\max_{h\le l}\big[r_h+(a+(l-h)\delta)/v_k+\text{(sum of the }l-h+1\text{ smallest } p_i,\ i\ge h)\big]$,
and $\sum_j C_j\ge\sum_k\sum_l(\text{that bound})$.

*Proof.* (a) Split the parts $i\succeq h$ of $J_k$, $i\ne\ell$, into $B$ (processed before $\ell$)
and $A$ (after). As in L1(iii), the belt is stopped on the disjoint intervals
$[S_i,C_i)\subset[r_h,S_\ell)$, $i\in B$, so $S_\ell\ge r_h+(a+(m_\ell-m_h)\delta)/v_k+\sum_B p_i$;
and $C_{\max}\ge C_\ell+\sum_A p_i$. (b) Let $Q$ be the first $l$ completed parts of $J_k$ and
$j$ the one of largest belt index, $j\ge l$. The same argument with $B=Q\cap\lbrace i\ge h\rbrace$
processed before $j$ and the rest of $Q$ after $j$ gives
$C^k_{(l)}\ge r_h+(a+(j-h)\delta)/v_k+\sum_{i\in Q,i\ge h}p_i$; at least $l-h+1$ members of $Q$
have index $\ge h$. $\square$

### Proposition L5 (preemptive SRPT bound)

$\sum_jC_j\ \ge\ \mathrm{OPT}\big(1\mid r_j{=}e_j,\ \mathrm{pmtn}\mid\sum C_j\big)$, computed
exactly by SRPT (Schrage 1968; Baker 1974). Moreover, with $g_j=\min_{i\ne j}s_{ij}$,
also $\sum_jC_j\ge\mathrm{OPT}\big(1\mid r_j{=}e_j-g_j,\ p_j{+}g_j,\ \mathrm{pmtn}\mid\sum C_j\big)$.

*Proof.* The processing intervals of a feasible schedule are disjoint and start after $e_j$
(L1): a non-preemptive, hence preemptive, single-machine schedule. For the second version,
the intervals $[S_j-g_j,C_j)$ are still disjoint: if $i$ immediately precedes $j$,
$S_j-g_j\ge C_i+s_{ij}-g_j\ge C_i$ (L2); and $S_j-g_j\ge e_j-g_j$. $\square$

### Theorem L6 (sequence relaxation, exact DP over subsets)

For a set $B$ of parts processed before $j$, define
$\hat e_j(B)=+\infty$ if $F_j\not\subset B$, otherwise

$$
\hat e_j(B)=\max\Big\lbrace\tau^0_j,\ \max_{h\preceq_k j}\Big[r_h+\frac{a+(m_j-m_h)\delta}{v_k}+\sum_{i\in B\cap M_h}p_i\Big]\Big\rbrace,
$$

with $M_h=\lbrace i\in J_k: m_i\ge m_h\rbrace$ for $\sigma=I$ and
$M_h=\lbrace i\in J: r_i\ge r_h\rbrace$ for $\sigma=$ all ones. For an order $\pi$ define
$\hat C_{\pi(1)}=\hat e_{\pi(1)}(\emptyset)+p_{\pi(1)}$ and
$\hat C_{\pi(q)}=\max\big(\hat e_{\pi(q)}(\lbrace\pi(1..q{-}1)\rbrace),\ \hat C_{\pi(q-1)}+s_{\pi(q-1)\pi(q)}\big)+p_{\pi(q)}$.
Then every feasible schedule with order $\pi$ has $C_{\pi(q)}\ge\hat C_{\pi(q)}$ for all $q$.
Consequently $C_{\max}\ge\min_\pi\max_q\hat C$ and $\sum C\ge\min_\pi\sum_q\hat C$.

*Proof.* Induction on $q$. $S_{\pi(q)}\ge C_{\pi(q-1)}+s\ge\hat C_{\pi(q-1)}+s$ by L2; the
release part is L1(ii)–(iii) with the parts of $B\cap M_h$ in place of $F_j$: they complete
before $S_j$ (they are processed before $j$), start after $r_i\ge r_h$, and stop belt $k$
during processing ($\sigma_{kk}=1$, resp. $\sigma_{k\,k(i)}=1$). $\square$

The recursion depends on the past only through (processed set, last part, $\hat C_{\text{last}}$)
and is non-decreasing in $\hat C_{\text{last}}$, so $\min_\pi\max_q\hat C$ is computed
**exactly** by dynamic programming over (set, last) in $O(2^n n^2)$. For $\sum C$ the DP keeps two
labels per state, $\min\hat C_{\text{last}}$ and $\min\sum\hat C$ (each a lower bound for every
path into the state, and the successor's $\hat C$ is monotone in $\hat C_{\text{last}}$), which is a
valid (possibly slightly weaker) bound.

**Corollary L6′ (windows).** For any $U\subset J$, the restriction of a feasible schedule to
$U$ satisfies the recursion built on $U$ alone (sums over $B\cap U$, forced parts outside $U$
ignored), so $C_{\max}\ge\mathrm{DP}_{C}(U)$ and $\sum_{j\in U}C_j\ge\mathrm{DP}_\Sigma(U)$. For a partition
$J=U_1\cup\dots\cup U_q$, $\sum_jC_j\ge\sum_b\mathrm{DP}_\Sigma(U_b)$. The code uses the exact DP for
$n\le 10$ and, beyond, windows of 8 consecutive parts in $e$-order (4 shifted partitions;
the best is kept), plus the last window for $C_{\max}$.

### Objectives

`best_lower_bound(instance, objective)` returns the maximum of the bounds above:

* makespan: trivial, $\max_j(e_j+p_j)$, L3 (with and without $H$), L4(a), L6;
* `avg_flow_time` $=(\sum C_j-\sum r_j)/n$ with $\sum C_j\ge\max(\sum_j(e_j+p_j),$ L4(b), L5, L6$)$;
* `avg_waiting_time` $=(\sum S_j-\sum r_j)/n$ with $S_j=C_j-p_j$: same bound minus $\sum p_j/n$;
* any other objective (or a composite dict) returns `None`.

Runtime of `all_lower_bounds`: ≤ 0.1 s for $n\le 10$, ≈ 0.25 s for $n=40$.

---

## 2. Special cases

### (a) A single conveyor

**Counterexample A (belt order is not optimal for total flow time).** One conveyor
($N=1$, $y_1=0$, other parameters default: $a=10$, $b=45$, $v=3$, $\delta=4$,
$v_x=v_y=10$, $P_0=(27.5,0)$). Parts: $A$: $r=0$, $p=10$; $B$: $r=1.5$, $p=1$.

* Any schedule of $\mathcal P$ in belt order ($A$ then $B$): by L1 $C_A\ge a/v+10=13.33$; by
  L6 (B is behind A, the belt stops while A is processed)
  $S_B\ge r_A+(a+\delta)/v+p_A=14.67$, so $C_A+C_B\ge 29.0$ and mean flow $\ge 13.75$ s.
* The reverse order, simulated (dt = 0.01) and verified by `schedule_check`: mean flow
  **11.77 s**. Hence FIFO / belt order is strictly suboptimal for $\sum F_j$ in $\mathcal P$ (by
  ≥ 1.98 s per part), whatever the start times. (Its makespan is better: 16.16 vs 17.84 s.)
  Test: `test_counterexample_single_conveyor_fifo_not_optimal_for_flow_time`.

The mechanism is SPT-like: when parts have different $p_j$ and are close together, a short
part behind a long one should go first; the stopped belt keeps the long part inside the zone.

**Conjecture A1 (constant processing times).** With one conveyor and equal $p_j$, belt order
is optimal for both makespan and total flow time over the decoder $\Phi$.
*Evidence:* exhaustive enumeration of all permutations (simulator, dt = 0.05):
random instances (inter-arrival times exponential with mean drawn from
$\lbrace1.5,3,6\rbrace$ s, $p_j=2.5$ s): **100/100** instances with $n=3$, **25/25** with $n=4$,
**40/40** with $n=5$, **10/10** with $n=6$ (all 720 orders) — belt order attained the decoder optimum (within 0.06 s) for both objectives
in every case.

**Conjecture A2 (makespan, any $p_j$).** With one conveyor, belt order is optimal for the
makespan over $\Phi$. *Evidence:* same generator with $p_j\in\lbrace1,2.5,6,10\rbrace$ s: belt order was
makespan-optimal in **25/25** ($n=4$) and **40/40** ($n=5$) instances, while it was *not*
flow-optimal in 7/25 ($n=4$) and 33/40 ($n=5$) instances (consistent with Counterexample A).


Both are stated for $\Phi$ (what the simulator can evaluate); for $\mathcal P$ they remain
open. Intuition: in belt order the laser always serves the most downstream remaining part,
which approaches the laser at relative speed $v_x+v$ (L2 with $j$ behind), whereas
serving a part ahead of the laser costs $\delta/v_x$ per spacing and leaves upstream parts
blocked behind processed ones.

### (b) Light traffic

Let $e^0_j=r_j+a/v_{k(j)}$ (free-flow availability; $e_j\ge e^0_j$ by L1 with $h=j$).

**Theorem LT-P (light traffic, model $\mathcal P$).** Assume $\sigma=I$ and
1. $a\ge\delta$;
2. consecutive parts $h,i$ of the same conveyor satisfy $r_i\ge r_h+a/v_k+p_h$;
3. in $e^0$-order $(1),(2),\dots$: $e^0_{(1)}\ge\tau(P_0,(a,y_{(1)}))$ and
   $e^0_{(q+1)}\ge e^0_{(q)}+p_{(q)}+\lvert y_{(q+1)}-y_{(q)}\rvert/v_y$ (simultaneous model; add
   nothing in $x$ since both points are at $x=a$).

Then the schedule $S_j=e^0_j$, $X_j=a$ (laser pre-positioned at $(a,y_{k(j)})$, belts stopped only
while processing) is feasible for $\mathcal P$ and **simultaneously optimal for every regular
objective** (makespan, flow, waiting, $L_{\max}$, ...), and our bound is exact:
`best_lower_bound` $=\max_j e^0_j+p_j$ resp. $\frac1n\sum_j(e^0_j+p_j-r_j)$.

*Proof.* Feasibility: by 2, when $i$ arrives its predecessor $h$ on the belt has been
completed ($C_h=e^0_h+p_h\le r_i$) at position $a\ge\delta$ and the belt is running, so
$E_i=D_k(r_i)$; no other part of $k$ is processed during $[r_i,e^0_i)$ (earlier ones are done,
later ones are processed at $e^0\ge e^0_i$), so $x_i(e^0_i)=a$. (C1) holds with $X=a$, (C2) is
condition 3, (C3) by construction, (C4) holds since unprocessed parts are at $x\le a<b$, (C5)
since belts stop only for processing. Optimality: every feasible schedule has
$S_j\ge e_j\ge e^0_j$ (L1). $\square$

(For $\sigma$ = all ones replace 2 by: $[e^0_h,e^0_h+p_h)\cap[r_i,e^0_i)=\emptyset$ for all $h\ne i$.)

**Theorem LT-Φ (light traffic, policies).** Simultaneous axes, $\sigma=I$, $P_0\in W$,
$v_k\le v_x$ for all $k$, and let

$$
I^\ast=\max\Big(\frac{b-a}{v_x+\min_kv_k},\ \frac{\max_ky_k-\min_ky_k}{v_y}\Big),\qquad v_kI^\ast\le b-a .
$$

If 1 holds and 2–3 are strengthened to $r_i\ge r_h+a/v_k+I^\ast+p_h$ and
$e^0_{(q+1)}\ge e^0_{(q)}+I^\ast+p_{(q)}$, then at every decision epoch of the simulator exactly
one part is READY, so **all policies (fifo, nearest_job, round_robin, weighted_score, any
fixed sequence that is non-idling) produce the same schedule**, and it satisfies
$e^0_j\le S_j\le e^0_j+I^\ast$. Hence its makespan and mean flow/waiting time exceed the
optimum of $\mathcal P$ by at most $I^\ast$.

*Proof.* Interception lemma: a laser at $(x_L,y_L)\in W$ meets a part entering at $(a,y)$ with
speed $v\le v_x$ within $t^\ast=\max\big((x_L-a)/(v_x+v),\lvert y_L-y\rvert/v_y\big)\le I^\ast$: at
$x_m=a+vt^\ast\le b$, if $x_L\ge x_m$ then $x_L-x_m\le v_xt^\ast$ by the choice of $t^\ast$, otherwise
$x_m-x_L\le vt^\ast\le v_xt^\ast$ (as $x_L\ge a$). Induction in $e^0$-order: part $(q)$ reaches $a$ at
$e^0_{(q)}$ exactly as in LT-P; the laser finished $(q-1)$ by $e^0_{(q-1)}+I^\ast+p\le e^0_{(q)}$,
no other part is READY (earlier ones are done, later ones are upstream of $a$), so every policy
selects $(q)$ and intercepts it within $I^\ast$ (before $b$, so no blocking). $\square$

Default numbers: $I^\ast=\max(35/13,\,24/10)=2.69$ s ($vI^\ast=8.1\le35$). LT-Φ needs
inter-arrivals $\ge 8.53$ s on each conveyor and $\ge 5.19$ s between consecutive availabilities
globally; LT-P needs $\ge 5.83$ s and $\ge 2.5+\lvert\Delta y\rvert/10\le 4.9$ s. Under the default
Poisson arrivals (total rate $1/10+1/12+1/14=0.255$ s$^{-1}$) a global gap is $\ge 5.19$ s with
probability $e^{-0.255\cdot5.19}\approx0.27$, so long light-traffic stretches are rare: the
conditions describe the *uncontended regime* but the default instances are contended.
Test: `test_light_traffic_all_policies_coincide_and_bound_is_exact` (instance: 6 parts, one every
6 s cycling over the three conveyors; all four policies give identical start times with
$S_j-e^0_j\in\lbrace1.36,1.21,1.21,2.41,1.21,1.21\rbrace\le I^\ast$; bound = $\mathcal P$-optimum
= 35.83 s makespan, 5.83 s mean flow; policies 37.04 s and 7.27 s).

Note that under light traffic **no** policy of the simulator is optimal for $\mathcal P$: the
optimum pre-positions the laser (see (d)).

### (c) When are nearest_job / FIFO optimal?

* **Proposition C1.** Under the conditions of Theorem LT-Φ, fifo, nearest_job,
  round_robin and weighted_score coincide and are within $I^\ast$ (makespan and mean
  flow/waiting) of the optimum of $\mathcal P$ (Theorem LT-Φ). Whether they are then also
  optimal over $\Phi$ is not claimed: another order changes the point from which the laser
  intercepts later parts.
* **Remark C2 (single conveyor).** On one conveyor, once the laser has served parts in belt
  order, all remaining parts are behind its position, so the most downstream READY part is also
  the nearest one (it is the closest to the laser in $x$); nearest_job and fifo then make the same
  choices. With Conjecture A1 this suggests both are optimal over $\Phi$ on one conveyor with
  constant $p_j$; Counterexample A shows they are not for flow time with unequal $p_j$.
* **Counterexample C3 (myopia: waiting beats serving).** `generate_instance(5, 3)` (default
  cell): P001–P003 on conveyor 2 at $r\approx0.56,0.74,2.41$; P004 on conveyor 1 at $r=19.678$;
  P005 on conveyor 2 at $r=19.881$. After P003 the laser sits on conveyor 2. P004 enters the zone
  0.2 s before P005, so **every** policy (one candidate at that epoch) serves P004 first:
  makespan 30.42 s, mean flow 9.27 s (dt = 0.01). The order P001, P002, P003, P005, P004 (wait
  0.2 s for P005, which is on the laser's conveyor, then go to P004) gives **29.45 s** and
  **8.88 s**, feasible and verified by `schedule_check`. Hence neither fifo nor nearest_job (nor
  the other two policies) is optimal, over $\Phi$ and hence over $\mathcal P$; the improvement
  needs deliberate idling, which no implemented policy does.
  Test: `test_counterexample_nearest_and_fifo_not_optimal`.
* **Evidence (how often they are optimal).** Exhaustive decoder optimum (all 120 orders,
  dt = 0.05) on `generate_instance(5, seed)`, seeds 1–30: fifo optimal in 22/30 (makespan) and
  21/30 (flow) instances, nearest_job in 20/30 and 21/30 (a policy counts as optimal if it is
  within $10^{-6}$ of the enumerated optimum at the same dt).

### (d) Decoder $\Phi$ versus $\mathcal P$

**Counterexample D1 (anticipation; Φ and every policy are strictly suboptimal for $\mathcal P$).**
Default 3-conveyor cell, one part on conveyor 1 ($y=0$) at $r=0$, $p=2.5$. In $\mathcal P$ the
laser leaves $P_0=(27.5,12)$ at $t=0$, reaches $(10,0)$ after $\max(17.5,12)/10=1.75$ s, and
starts at $S=a/v=3.33$ s when the part arrives there: $C=5.83$ s, which equals our lower bound
(so it is optimal). The simulator only dispatches parts already in the zone: the laser stays at
$P_0$ until $t=3.33$ and intercepts the moving part 1.35 s later, $C=7.19$ s for every policy and
for $\Phi$ (+23 %). Test: `test_counterexample_decoder_gap_anticipation`.

In general, whenever a part enters the zone while the laser is idle away from $(a,y_k)$,
the simulator loses up to $I^\ast$ with respect to $\mathcal P$. This is a structural part of every
gap between our bounds and the policies (≈ 1.2–1.4 s per part in light traffic), not looseness
of the bounds.

**Proposition D2 (delays cannot pay through travel alone).** Assume $v_k\le v_x$. Delaying the
start of $j$ by $\varepsilon$ (same sequence, everything else unchanged) delays $C_j$ by
$\varepsilon$ and moves $X_j$ by at most $v_k\varepsilon$, so $\tau(P_j,P_l)$ for the successor $l$
decreases by at most $v_k\varepsilon/v_x\le\varepsilon$: the travel constraint (C2) of the successor is
never relaxed. *Proof:* $\tau$ is $1/v_x$-Lipschitz in the $x$ coordinate of either end point. $\square$

What a delay *does* change is the belt: delaying a stop of conveyor $k$ makes $D_k$ pointwise
larger (the stop interval is shifted right), so parts behind $j$ advance earlier. Whether this can
strictly improve a regular objective (with the same sequence) is left **open**; we have
neither a proof nor a counterexample. Combined with D1 and C3: the
non-idling, non-anticipative schedules of the simulator are not dominant in $\mathcal P$ — what
pays is anticipative laser motion (D1) and deliberate idling of the laser for a part that is
about to enter (C3, which is still inside $\Phi$), rather than later starts of the same sequence.

---

## 3. Quality of the bounds

Gap $=100\,(\text{UB}-\text{LB})/\text{LB}$, where LB = `best_lower_bound` and UB = best known
schedule value at dt = 0.01: for $n\le 8$ the optimum over $\Phi$ (all $n!$ orders, enumerated
with pruning: an order is simulated only if its L6 sequence bound is below the incumbent
+ 0.1 s, valid up to the simulator's $O(dt)$ error; the best of this and the four policies is
reported); for $n\ge10$ the best of the four online policies (`evaluate_policy`). Instances
`generate_instance(n, seed)`, default configuration. "bound" is the bound attaining the maximum.

| $n$ | seed | objective | lower bound | attained by | best known (UB) | UB source | gap % |
|---|---|---|---|---|---|---|---|
| 6 | 1 | makespan | 40.63 | L6_relaxation | 41.53 | exhaustive(phi) | 2.2 |
| 6 | 1 | avg_flow_time | 6.38 | L6_relaxation | 7.17 | exhaustive(phi) | 12.3 |
| 6 | 2 | makespan | 28.64 | L6_relaxation | 30.47 | exhaustive(phi) | 6.4 |
| 6 | 2 | avg_flow_time | 9.90 | L6_relaxation | 11.30 | exhaustive(phi) | 14.2 |
| 6 | 3 | makespan | 32.02 | L3_jackson_gaps | 33.15 | exhaustive(phi) | 3.5 |
| 6 | 3 | avg_flow_time | 8.67 | L6_relaxation | 9.23 | exhaustive(phi) | 6.6 |
| 8 | 1 | makespan | 47.14 | L6_relaxation | 48.25 | exhaustive(phi) | 2.4 |
| 8 | 1 | avg_flow_time | 7.45 | L6_relaxation | 8.29 | exhaustive(phi) | 11.3 |
| 8 | 2 | makespan | 35.89 | L6_relaxation | 38.93 | exhaustive(phi) | 8.5 |
| 8 | 2 | avg_flow_time | 9.31 | L6_relaxation | 11.13 | exhaustive(phi) | 19.6 |
| 8 | 3 | makespan | 38.53 | L3_jackson_gaps | 39.93 | exhaustive(phi) | 3.6 |
| 8 | 3 | avg_flow_time | 8.29 | L6_relaxation | 9.07 | exhaustive(phi) | 9.4 |
| 10 | 1 | makespan | 53.46 | L6_relaxation | 55.69 | round_robin | 4.2 |
| 10 | 1 | avg_flow_time | 8.73 | L6_relaxation | 9.74 | fifo | 11.6 |
| 10 | 2 | makespan | 48.93 | trivial | 49.41 | fifo | 1.0 |
| 10 | 2 | avg_flow_time | 8.62 | L6_relaxation | 10.45 | fifo | 21.3 |
| 10 | 3 | makespan | 44.26 | L6_relaxation | 49.16 | fifo | 11.1 |
| 10 | 3 | avg_flow_time | 8.38 | L6_relaxation | 10.74 | fifo | 28.2 |
| 20 | 1 | makespan | 80.51 | L3_jackson_gaps | 92.19 | nearest_job | 14.5 |
| 20 | 1 | avg_flow_time | 12.89 | L5_srpt | 17.93 | nearest_job | 39.2 |
| 20 | 2 | makespan | 91.55 | L6_relaxation | 94.36 | nearest_job | 3.1 |
| 20 | 2 | avg_flow_time | 8.82 | L6_relaxation | 10.89 | fifo | 23.5 |
| 20 | 3 | makespan | 88.18 | L3_jackson_gaps | 91.43 | nearest_job | 3.7 |
| 20 | 3 | avg_flow_time | 8.94 | L6_relaxation | 13.42 | weighted_score | 50.1 |
| 40 | 1 | makespan | 184.19 | L6_relaxation | 186.33 | fifo | 1.2 |
| 40 | 1 | avg_flow_time | 10.29 | L5_srpt | 15.55 | nearest_job | 51.1 |
| 40 | 2 | makespan | 177.28 | L6_relaxation | 187.07 | fifo | 5.5 |
| 40 | 2 | avg_flow_time | 8.37 | L6_relaxation | 11.83 | nearest_job | 41.4 |
| 40 | 3 | makespan | 165.70 | L6_relaxation | 172.57 | weighted_score | 4.1 |
| 40 | 3 | avg_flow_time | 9.37 | L6_relaxation | 13.59 | weighted_score | 45.1 |

Mean gap per $n$ (3 seeds) and time of one `all_lower_bounds` call:

| $n$ | makespan | mean flow time | bound time (s) |
|---|---|---|---|
| 6 | 4.0 % | 11.0 % | 0.03 |
| 8 | 4.8 % | 13.4 % | 0.03 |
| 10 | 5.4 % | 20.4 % | 0.08 |
| 20 | 7.1 % | 37.6 % | 0.11 |
| 40 | 3.6 % | 45.9 % | 0.25 |

Reading the table:

* **Makespan:** gaps 1.0–8.5 % in 13 of the 15 instances (11.1 % and 14.5 % in the other
  two). L6 is the best bound in most cases; the Jackson bound with travel (L3) wins on
  instances with a long contended stretch.
* **Mean flow time:** 6.6–19.6 % for $n\le8$ (L6, exact DP), 11.6–28.2 % for $n=10$, and
  23.5–51.1 % for $n=20$–$40$. Two sources:
  (i) part of every gap is the *true* difference between $\mathcal P$ and the simulator
  (anticipation, (d): up to $I^\ast$ per part, about 1.3 s on a mean flow of 7–10 s, i.e.
  10–20 %); (ii) for large $n$ the window decomposition of L6 ignores contention between
  windows and SRPT (L5) ignores belt stops and travel, so the bound is loose in heavy traffic.
  For large $n$ the UB is only the best policy, so part of the gap is also suboptimality of
  the policies.

---

## 4. Remarks on `docs/formal_formulation.md`

1. **§3 vs. the simulator (inaccuracy).** §3 defines $\Phi$ through "the smallest $t$ with
   $\tau(P_{\pi(i)},(x_{\pi(i+1)}(t),y))\le t-C_{\pi(i)}$" and states that this map "is exactly what
   the simulator does with a fixed sequence". When $\pi(i+1)$ has not yet entered the laser zone at
   $C_{\pi(i)}$, that formula lets the laser travel towards $(a,y)$ in advance, whereas the simulator
   (`fixed_sequence` and all policies) only dispatches READY parts and keeps the laser idle at
   $P_{\pi(i)}$. Counterexample D1 shows the difference (start 3.33 s vs. 4.68 s). The decoder in the
   code is therefore "$\Phi$ with non-anticipative travel", consistent with §4 but not with the
   formula of §3. Either §3 should say so or the decoder could pre-position the laser.
2. **§5.5 (minor).** The reduction to $1\mid r_j\mid\sum C_j$ also needs that processing a part
   does not delay the availability of other parts of the same conveyor (C3 stops the belt while
   upstream parts are still travelling to $a$); e.g. one part per conveyor. The NP-hardness
   claim is unaffected (take one part per conveyor).
3. §5.4 bounds are valid (they are L1 with $h=j$ and the case $t=\min_j e^0_j$ of L3) and are
   dominated by `best_lower_bound`.

## 5. Reproducing

```
python -m pytest tests/test_lower_bounds.py
python -c "from optimization.instances import generate_instance as g; from optimization.lower_bounds import all_lower_bounds as L; print(L(g(8,1),'makespan'))"
```

(with `PYTHONPATH=src`). The gap table was produced with the scripts described in §3 (decoder
enumeration with L6 pruning, policies via `evaluate_policy` at dt = 0.01).
