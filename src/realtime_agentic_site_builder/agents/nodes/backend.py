"""Writes endpoint handlers, then binds the UI components that submit to them."""

from typing import Any, Final

from realtime_agentic_site_builder.agents.runtime import NodeContext
from realtime_agentic_site_builder.agents.schemas import ComponentSpec, EndpointSpec, HandlerOutput
from realtime_agentic_site_builder.agents.state import AgentWorkflowState
from realtime_agentic_site_builder.domain.backend import ApiEndpointNode
from realtime_agentic_site_builder.domain.swarm import AgentAction

SYSTEM_PROMPT: Final = (
    "You write one FastAPI endpoint handler. Return Python source whose top level defines a single "
    "async function. Do not include imports for the framework or the route decorator."
)


def _handler_targets(state: AgentWorkflowState, context: NodeContext) -> list[EndpointSpec]:
    if state["iteration"] > 0:
        retry = set(state["retry_backend"])
        return [spec for spec in state["plan"].endpoints if spec.endpoint_id in retry]
    existing = {endpoint.id for endpoint in context.bundle().backend.endpoints}
    return [spec for spec in state["plan"].endpoints if spec.endpoint_id not in existing]


async def _write_handler(context: NodeContext, spec: EndpointSpec) -> None:
    endpoint_id = spec.endpoint_id
    async with context.locks() as locks:
        await context.emit(
            AgentAction.ACQUIRING_LOCK,
            f"Locking {endpoint_id}.",
            tool_name="lock.acquire",
            tool_args={"target_id": endpoint_id},
        )
        await locks.acquire(endpoint_id)
        feedback = context.feedback_for(endpoint_id)
        prompt = (
            f"{spec.method.value} {spec.path}\nSummary: {spec.summary}\nBrief: {spec.brief}\n"
            f"Request model: {spec.request_model_ref}\nResponse model: {spec.response_model_ref}"
        )
        if feedback:
            prompt = f"{prompt}\nYour previous attempt was rejected:\n{feedback}"
        result = await context.generate(SYSTEM_PROMPT, prompt, HandlerOutput)
        await context.emit(AgentAction.COMPILING_BACKEND, f"Compiling {endpoint_id}.")
        endpoint = ApiEndpointNode.create(
            path=spec.path,
            method=spec.method,
            summary=spec.summary,
            handler_code=result.output.handler_code,
            request_model_ref=spec.request_model_ref,
            response_model_ref=spec.response_model_ref,
        )
        patches: list[dict[str, Any]]
        existing = [item.id for item in context.bundle().backend.endpoints]
        if endpoint_id in existing:
            index = existing.index(endpoint_id)
            patches = [
                {"op": "test", "path": f"/backend/endpoints/{index}/id", "value": endpoint_id},
                {
                    "op": "replace",
                    "path": f"/backend/endpoints/{index}/handler_code",
                    "value": endpoint.handler_code,
                },
            ]
        else:
            value = endpoint.model_dump(mode="json")
            patches = [{"op": "add", "path": "/backend/endpoints/-", "value": value}]
        await context.commit("backend", endpoint_id, patches, lock=locks.lock_on(endpoint_id))


async def _bind(context: NodeContext, spec: ComponentSpec, endpoint_id: str) -> None:
    async with context.locks() as locks:
        await locks.acquire(spec.id)
        index = context.component_index(spec.id)
        await context.commit(
            "frontend",
            spec.id,
            [
                {"op": "test", "path": f"/frontend/components/{index}/id", "value": spec.id},
                {
                    "op": "replace",
                    "path": f"/frontend/components/{index}/component_type",
                    "value": spec.component_type.value,
                },
                {
                    "op": "replace",
                    "path": f"/frontend/components/{index}/connected_endpoint_id",
                    "value": endpoint_id,
                },
            ],
            lock=locks.lock_on(spec.id),
        )


async def backend_node(state: AgentWorkflowState, context: NodeContext) -> dict[str, Any]:
    for spec in _handler_targets(state, context):
        with context.recoverable(f"endpoint {spec.endpoint_id}"):
            await _write_handler(context, spec)
    for component in state["plan"].components:
        bundle = context.bundle()
        bound = {node.id: node.connected_endpoint_id for node in bundle.frontend.components}
        endpoints = {endpoint.id for endpoint in bundle.backend.endpoints}
        endpoint_id = component.endpoint_id
        # An endpoint that failed to commit is reported by the critic, and bound on its retry.
        if endpoint_id in endpoints and bound.get(component.id) != endpoint_id:
            with context.recoverable(f"binding of {component.id}"):
                await _bind(context, component, str(endpoint_id))
    return {}
