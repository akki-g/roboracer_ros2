#!/bin/bash
# Same as tools/sim_gru_linux.sh (kept so earlier notes still work).
exec bash "$(dirname "${BASH_SOURCE[0]}")/sim_gru_linux.sh" "$@"
