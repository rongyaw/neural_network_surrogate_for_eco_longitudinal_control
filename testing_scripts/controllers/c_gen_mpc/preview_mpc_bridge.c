#include "preview_mpc.h"
#include "preview_mpc.c"
CASADI_SYMBOL_EXPORT int preview_mpc_step(
    const double* reference, const double* initial_state,
    const double* state_guess, const double* control_guess,
    double* state_solution, double* control_solution) {
  static const casadi_real* arguments[preview_mpc_SZ_ARG];
  static casadi_real* results[preview_mpc_SZ_RES];
  static casadi_int integer_work[preview_mpc_SZ_IW];
  static casadi_real real_work[preview_mpc_SZ_W];
  int i;
  for (i = 0; i < preview_mpc_SZ_ARG; ++i) arguments[i] = 0;
  for (i = 0; i < preview_mpc_SZ_RES; ++i) results[i] = 0;
  arguments[0] = reference;
  arguments[1] = initial_state;
  arguments[2] = state_guess;
  arguments[3] = control_guess;
  results[0] = state_solution;
  results[1] = control_solution;
  return preview_mpc(arguments, results, integer_work, real_work, 0);
}
CASADI_SYMBOL_EXPORT void preview_mpc_cleanup(void) {
  int memory;
  for (memory = 0; memory < casadi_f3_mem_counter; ++memory) {
    casadi_f3_free_mem(memory);
  }
  casadi_f3_mem_counter = 0;
  casadi_f3_unused_stack_counter = -1;
}
