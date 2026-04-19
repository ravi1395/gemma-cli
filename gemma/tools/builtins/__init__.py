"""Built-in tool implementations.

Importing this package triggers the ``@tool`` decorator on each module
below, populating the global registry. Order of import doesn't affect
correctness — every module registers independently — but is kept
deterministic for easier debugging.

If you are adding a new builtin, give it its own module here and
import it below. Keep tools one-to-a-file so unit tests can target
them in isolation and the registry diff is easy to review.
"""

from __future__ import annotations

# Importing each module runs its decorator and registers the tool.
# Explicit imports (rather than a glob) keep the registry contents
# visible at a glance.
from gemma.tools.builtins import fs_read    # noqa: F401
from gemma.tools.builtins import fs_write   # noqa: F401
from gemma.tools.builtins import fs_archive  # noqa: F401
from gemma.tools.builtins import lint       # noqa: F401
from gemma.tools.builtins import tests      # noqa: F401
from gemma.tools.builtins import net_fetch  # noqa: F401
