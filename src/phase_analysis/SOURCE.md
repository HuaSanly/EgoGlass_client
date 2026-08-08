# Phase Analysis Source

The phase boundaries and short-run cleanup are adapted from HumanEgo
`preprocess/AriaPhases.py` and `preprocess/AriaPhasesOps.py` at commit
`18fb1082abb87b79f88e53f2abb5bfb9f61de19b`.

This implementation is a clean-room adaptation for EgoGlass `session_time_ns`,
Basalt VIO, and typed hand kinematics. It does not import Project Aria tools or
copy HumanEgo I/O and visualization code. See `src/hand_tracking/HUMANEGO-LICENSE.txt`
for the upstream license and required notice.
