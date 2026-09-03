"""Launcher that patches `mcp_snowflake_server`'s JSON serialization.

The upstream `mcp_snowflake_server` serializes query results with a bare
``json.dumps(output)`` (no ``default=`` handler), so any result containing a
pandas ``Timestamp``, ``date``/``datetime``, ``Decimal``, ``bytes`` or other
non-JSON-native value raises::

    Object of type Timestamp is not JSON serializable

That bug fires on results we can't control from SQL (e.g. metadata returned by
``list_tables``/``describe_table``) and forces the agent into error-recovery.

Because ``server.py`` calls ``json.dumps`` via attribute lookup at call time,
replacing the attribute on the ``json`` module here — before the server is
imported/run — makes every call site fall back to ``str`` for exotic types.
All connection/CLI arguments are passed straight through via ``sys.argv``.
"""

import datetime
import decimal
import json


def _json_default(obj):
    """Serialize types the stdlib JSON encoder can't handle, losslessly enough."""
    if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        # Keep integers as ints; otherwise fall back to float for JSON numbers.
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    # pandas Timestamp/Timedelta/NaT and anything else: stringify.
    return str(obj)


_original_dumps = json.dumps


def _safe_dumps(obj, *args, **kwargs):
    kwargs.setdefault("default", _json_default)
    return _original_dumps(obj, *args, **kwargs)


json.dumps = _safe_dumps


def main() -> None:
    # Imported after the patch so the server picks up the patched json.dumps.
    from mcp_snowflake_server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
