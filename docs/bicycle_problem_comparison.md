# Relation to the "bicycle problem"

## The bicycle problem

Three people must reach a destination. Each person walks at speed `w_i` and cycles at
`c_i > w_i`. There is **only one bicycle**, and only one person can ride it at a time.
People and bicycle can move forwards and backwards (someone can leave the bicycle on
the road for another person to pick up, or ride it back). Goal: minimize the time at
which the last person arrives.

## Why there is an analogy

Both are instances of **shared moving resource scheduling**:

| | Bicycle problem | Laser + conveyors |
|---|---|---|
| agents / jobs | 3 people | parts on N conveyors |
| single resource | bicycle | laser head |
| exclusivity | one rider at a time | one part processed at a time |
| the resource must travel | the bicycle must reach whoever uses it next | the laser travels to the part (sequence-dependent travel time) |
| decision | who rides, when, and where the bicycle is left | which part gets the laser, and when |
| objective | minimize the arrival time of the last person | minimize makespan / waiting / travel / downtime |
| progress without the resource | walking (slower) | the part moves with its conveyor but is not processed |

In both, assigning the resource to one agent benefits that agent and forces the
resource to "reposition" before serving the next one; the optimum balances immediate
benefit against repositioning cost. The intuitions (never leave the resource idle, serve
first whoever is falling behind, the cost of repositioning) carry over.

## Why they are NOT mathematically identical

1. **Continuous arrivals.** The bicycle problem has a fixed set of 3 agents known from
   the start. Here jobs keep arriving (potentially without end): it is an
   *online*/dynamic problem, not a closed instance.
2. **Queues.** Parts accumulate; queue lengths, waiting times and stability (load ρ < 1)
   appear, none of which exist in the bicycle problem.
3. **Processing times.** The laser does not just "carry" an advantage: it is occupied
   for `p_j` on each part, during which it cannot move.
4. **Conveyor dynamics.** Job positions change by themselves over time (and depend on
   the stops caused by the laser itself). In the bicycle problem the agents decide how
   they move; here they do not.
5. **Blocking / stopping.** Processing a part stops its conveyor, and an unprocessed part
   at the end of the zone can block it. Decisions change the future geometry of every job
   on that conveyor.
6. **Accessibility windows.** Each part is reachable only inside the laser zone during a
   time interval: there are implicit *time windows*.
7. **Multiple objectives.** Makespan, waiting, laser travel, conveyor downtime, lateness
   and throughput conflict; the bicycle problem has a single objective.
8. **Scale.** The 3-person bicycle problem has a (near) closed-form solution; here the
   number of jobs grows with the horizon and the problem is NP-hard in general.

## Conclusion

The bicycle problem is a **simplified conceptual model** of *shared moving resource
scheduling*: useful for intuition and for explaining the problem, but its formulation
cannot be reused directly. The right formulation for the cell is single-machine
sequencing with sequence-dependent setup times, dynamic arrivals, time windows and
blocking (see `src/optimization/problem_formulation.md`).
