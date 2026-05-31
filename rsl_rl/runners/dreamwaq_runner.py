# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.algorithms import DreamWaQ
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class DreamWaQRunner(OnPolicyRunner):
    """Compatibility runner for DreamWaQ.

    The standard :class:`OnPolicyRunner` already supports DreamWaQ through the algorithm ``class_name`` entry, so this
    subclass exists mainly for configs that still refer to ``DreamWaQRunner`` by name.
    """

    alg: DreamWaQ
