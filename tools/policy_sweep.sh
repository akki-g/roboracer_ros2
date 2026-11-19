#!/bin/bash
# One independently guarded/recorded run per cap. Ctrl-C closes this run; q quits at the next prompt.
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$WS/tools/car_tests.sh" policy-sweep "$@"
