"""Generate C code for the Preview MPC CasADi QP."""

import os
import re
import shlex
import subprocess

import casadi
import numpy as np


HORIZON = 20
DT = 0.5
V_MAX_MPH = 40.0
CONTROL_COST = 4000.0
DISTANCE_GAP = 7.0
FUNCTION_NAME = "preview_mpc"


def build_opti_problem():
    """Build the same fixed-horizon QP used by the Python Preview MPC."""
    opti = casadi.Opti("conic")

    state = opti.variable(3, HORIZON)
    control = opti.variable(HORIZON - 1)
    slack_gap = opti.variable(HORIZON)
    slack_speed = opti.variable(HORIZON)
    slack_unused = opti.variable(HORIZON)
    reference = opti.parameter(3, HORIZON)
    initial_state = opti.parameter(3, 1)

    q_acceleration = casadi.MX(CONTROL_COST)
    q_control = casadi.MX(1)
    q_gap = casadi.MX(1)
    q_gap_slack = casadi.MX(1e6)
    q_speed_slack = casadi.MX(1e6)

    transition = casadi.MX(
        np.array(
            [
                [1.0, DT, DT**2 / 2.0],
                [0.0, 1.0, DT],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    input_matrix = casadi.MX(
        np.array([[DT**3 / 6.0], [DT / 2.0], [DT]])
    )
    max_speed = V_MAX_MPH * 0.44704

    opti.subject_to(state[:, 0] == initial_state)
    cost = 0

    for step in range(HORIZON - 1):
        cost += (
            q_control * control[step] ** 2
            + q_gap
            * (reference[0, step] - DISTANCE_GAP - state[0, step]) ** 2
            + q_acceleration * state[2, step] ** 2
        )
        opti.subject_to(
            state[:, step + 1]
            == transition @ state[:, step] + input_matrix @ control[step]
        )
        opti.subject_to(state[1, step] <= max_speed + slack_speed[step])
        opti.subject_to(state[1, step] >= -slack_speed[step])
        opti.subject_to(
            state[0, step]
            <= reference[0, step] - DISTANCE_GAP + slack_gap[step]
        )
        cost += (
            q_gap_slack * slack_gap[step] ** 2
            + q_speed_slack * slack_speed[step] ** 2
        )

    final_step = HORIZON - 1
    cost += (
        q_gap
        * (reference[0, final_step] - DISTANCE_GAP - state[0, final_step]) ** 2
        + q_acceleration * state[2, final_step] ** 2
        + q_gap_slack * slack_gap[final_step] ** 2
        + q_speed_slack * slack_speed[final_step] ** 2
    )
    opti.subject_to(state[1, final_step] <= max_speed + slack_speed[final_step])
    opti.subject_to(state[1, final_step] >= -slack_speed[final_step])
    opti.subject_to(
        state[0, final_step]
        <= reference[0, final_step] - DISTANCE_GAP + slack_gap[final_step]
    )
    opti.minimize(cost)

    # Expose state and control guesses as function inputs for the existing
    # shifted-trajectory warm start. Slack guesses remain zero, as before.
    opti.set_initial(slack_gap, 0.0)
    opti.set_initial(slack_speed, 0.0)
    opti.set_initial(slack_unused, 0.0)
    opti.solver("osqp", {"print_time": 0}, {"verbose": False})
    function = opti.to_function(
        FUNCTION_NAME,
        [reference, initial_state, state, control],
        [state, control],
        ["reference", "initial_state", "state_guess", "control_guess"],
        ["state_solution", "control_solution"],
    )
    return function


def generate_and_compile():
    """Generate the CasADi C function and compile it as a shared library."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    casadi_dir = os.path.dirname(os.path.abspath(casadi.__file__))
    source_file = FUNCTION_NAME + ".c"
    header_file = FUNCTION_NAME + ".h"
    bridge_file = FUNCTION_NAME + "_bridge.c"
    library_file = "lib" + FUNCTION_NAME + ".so"

    function = build_opti_problem()
    previous_dir = os.getcwd()
    try:
        os.chdir(output_dir)
        function.generate(source_file, {"with_header": True})
    finally:
        os.chdir(previous_dir)

    with open(os.path.join(output_dir, source_file), "r") as generated_source:
        source_text = generated_source.read()
    memory_match = re.search(r"static int (casadi_\w+)_mem_counter = 0;", source_text)
    if memory_match is None:
        raise RuntimeError("Could not find the generated OSQP memory manager")
    memory_prefix = memory_match.group(1)
    if not re.search(
        r"static (?:OSQPWorkspace|OSQPSolver)\* "
        + re.escape(memory_prefix)
        + r"_mem\[CASADI_MAX_NUM_THREADS\];",
        source_text,
    ):
        raise RuntimeError("Generated CasADi solver memory layout is not recognized")

    bridge_source = (
        '#include "' + header_file + '"\n'
        '#include "' + source_file + '"\n'
        "CASADI_SYMBOL_EXPORT int preview_mpc_step(\n"
        "    const double* reference, const double* initial_state,\n"
        "    const double* state_guess, const double* control_guess,\n"
        "    double* state_solution, double* control_solution) {\n"
        "  static const casadi_real* arguments[preview_mpc_SZ_ARG];\n"
        "  static casadi_real* results[preview_mpc_SZ_RES];\n"
        "  static casadi_int integer_work[preview_mpc_SZ_IW];\n"
        "  static casadi_real real_work[preview_mpc_SZ_W];\n"
        "  int i;\n"
        "  for (i = 0; i < preview_mpc_SZ_ARG; ++i) arguments[i] = 0;\n"
        "  for (i = 0; i < preview_mpc_SZ_RES; ++i) results[i] = 0;\n"
        "  arguments[0] = reference;\n"
        "  arguments[1] = initial_state;\n"
        "  arguments[2] = state_guess;\n"
        "  arguments[3] = control_guess;\n"
        "  results[0] = state_solution;\n"
        "  results[1] = control_solution;\n"
        "  return preview_mpc(arguments, results, integer_work, real_work, 0);\n"
        "}\n"
        "CASADI_SYMBOL_EXPORT void preview_mpc_cleanup(void) {\n"
        "  int memory;\n"
        "  for (memory = 0; memory < " + memory_prefix + "_mem_counter; ++memory) {\n"
        "    " + memory_prefix + "_free_mem(memory);\n"
        "  }\n"
        "  " + memory_prefix + "_mem_counter = 0;\n"
        "  " + memory_prefix + "_unused_stack_counter = -1;\n"
        "}\n"
    )
    with open(os.path.join(output_dir, bridge_file), "w") as bridge_output:
        bridge_output.write(bridge_source)

    command = shlex.split(os.environ.get("CC", "cc"))
    command.extend(
        [
            "-O3",
            "-fPIC",
            "-shared",
            bridge_file,
            "-I" + os.path.join(casadi_dir, "include"),
            "-L" + casadi_dir,
            "-Wl,-rpath," + casadi_dir,
            "-losqp",
            "-lm",
            "-o",
            library_file,
        ]
    )
    subprocess.run(command, cwd=output_dir, check=True)
    print("Generated:", os.path.join(output_dir, source_file))
    print("Generated:", os.path.join(output_dir, header_file))
    print("Built:", os.path.join(output_dir, library_file))


if __name__ == "__main__":
    generate_and_compile()
