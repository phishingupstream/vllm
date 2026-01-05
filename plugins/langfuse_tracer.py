# SPDX-License-Identifier: Apache-2.0
# Backward compatibility shim — use plugins.otel_tracer.OtelTracerMiddleware
from plugins.otel_tracer import OtelTracerMiddleware as LangfuseTracerMiddleware  # noqa: F401

__all__ = ["LangfuseTracerMiddleware"]
