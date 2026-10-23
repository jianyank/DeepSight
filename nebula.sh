#!/usr/bin/env bash
# Compatibility entry point; actual launch scripts live under scripts/train/.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/scripts/train/train.sh" "$@"
