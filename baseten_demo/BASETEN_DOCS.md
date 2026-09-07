# Baseten interface documentation citations

Every Baseten API/config interface used in this package was verified against
the **current** Baseten documentation before coding (docs.baseten.co, fetched
2026-09-07) and the official `baseten` Python SDK **v0.10.0** (the current
PyPI release; its `managementapi`/`inferenceapi` modules are generated from
the Baseten OpenAPI spec). No interface is used from memory alone.

## Wake (scale-from-zero)

Used by: `baseten_demo/client.py :: BasetenClient.wake`, `deployment.py :: wake_url`.

| Interface | Shape |
| --- | --- |
| `POST {model_host}/deployment/{deployment_id}/wake` | returns `202 Accepted`, empty body |

Docs:

- <https://docs.baseten.co/api-reference/non-regional/wake-a-specific-deployment-of-a-model-by-deployment-id>
- <https://docs.baseten.co/deployment/manage/scaling> ("Wake a scaled-to-zero deployment": `curl -i -X POST "https://model-{model_id}.api.baseten.co/deployment/{deployment_id}/wake" ...` → `HTTP/2 202`)
- SDK: `baseten.client.inferenceapi.ApiClient.wake_deployment` (`POST /deployment/{deployment_id}/wake`, `success_code=202`)
- Named-environment / production wake alternatives: <https://docs.baseten.co/api-reference/non-regional/wake-a-named-environment-of-a-model>, <https://docs.baseten.co/api-reference/non-regional/wake-the-production-environment-of-a-model>

## Readiness / health signal

Used by: `BasetenClient.deployment_status` / `.is_ready`, `Harness._poll_ready`.

| Interface | Shape |
| --- | --- |
| `GET https://api.baseten.co/v1/models/{model_id}/deployments/{deployment_id}` | `Deployment.status` enum + `active_replica_count` |

Readiness is the documented health/readiness signal: `status == "ACTIVE"` and
`active_replica_count >= 1`. The documented wake lifecycle is
`SCALED_TO_ZERO` → `WAKING_UP` → `ACTIVE`.

Docs:

- <https://docs.baseten.co/deployment/manage/scaling> ("poll `baseten model deployment describe` until `status` is `ACTIVE`"; "moving from `SCALED_TO_ZERO` through `WAKING_UP` to `ACTIVE`")
- <https://docs.baseten.co/deployment/manage/scaling#scale-to-zero>
- SDK `DeploymentStatus` enum (`DeploymentStatus.ACTIVE`, `SCALED_TO_ZERO`, `WAKING_UP`, `LOADING_MODEL`, ...) and `Deployment` model (`status`, `active_replica_count`) in `baseten/client/managementapi/_models.py`
- Container-level readiness probes / `/health` endpoint: <https://docs.baseten.co/development/model/health-checks>

This slice adds **one trivial inference probe** on top of the documented
health signal (feature lead decision 2); the probe doubles as the first
inference and is recorded as `first_inference`.

## Autoscaling (min/max replicas, idle scale-down)

Used by: `deployment.py :: build_autoscaling_settings`, `autoscaling_url`.

| Interface | Shape |
| --- | --- |
| `PATCH https://api.baseten.co/v1/models/{model_id}/deployments/{deployment_id}/autoscaling_settings` | `{min_replica, max_replica, autoscaling_window, scale_down_delay, concurrency_target, target_utilization_percentage}` → `{status: "ACCEPTED", message}` |

Docs:

- <https://docs.baseten.co/reference/management-api/deployments/autoscaling/updates-a-deployments-autoscaling-settings>
- <https://docs.baseten.co/deployment/autoscaling/overview> (parameter table: Min replicas `≥ 0` default `0` = scale to zero; Max replicas `≥ 1` default `1`; Scale-down delay `0–3600s` default `900s`, BIS-LLM default `300s`)
- <https://docs.baseten.co/deployment/autoscaling/cold-starts>
- SDK: `managementapi.ApiClient.patch_models_deployments_autoscaling_settings` (`PATCH /v1/models/{}/deployments/{}/autoscaling_settings`) with `UpdateAutoscalingSettings` (`min_replica`, `max_replica`, `autoscaling_window`, `scale_down_delay`, `concurrency_target`, `target_utilization_percentage`, `max_scale_down_rate`).

### Conflicts / decisions of record

- **`min_replicas` / `max_replicas` (feature record) vs `min_replica` / `max_replica` (documented).** The customer/feature wording uses the plural; the documented Baseten field names are singular. **The documented interface wins.** The profile file keeps the readable plural keys, and `deployment.py` maps them to the documented singular payload fields.
- **Idle-scale-down range.** The documented `scale_down_delay` range is `0–3600` seconds. The customer asked for ~2–5 minutes, so `profiles.py` validates each profile's idle timeout to the stricter closed `[120, 300]`-second window. Default `300`s sits inside both ranges. This tightening is a product decision layered on the documented range, recorded here.

## Deployment configuration (model definition)

Used by: `deployment.py :: build_config_yaml`.

The deployment definition is a Truss `config.yaml` (documented model
configuration format), built so model/adapter artifacts are **pre-deployed**
(BDN `weights` mount + `--served-model-name`) and waking only provisions/loads
a replica — there is **no build/package/redeploy step** in the definition.

Docs:

- <https://docs.baseten.co/development/model/configuration> (config.yaml structure)
- <https://docs.baseten.co/examples/models/llm/qwen3> (canonical vLLM `config.yaml`: `model_name`, `model_metadata.tags: [openai-compatible]`, `base_image`, `docker_server` with `start_command`/`readiness_endpoint: /health`/`liveness_endpoint: /health`/`predict_endpoint: /v1/chat/completions`/`server_port`, `weights`, `resources`)
- <https://docs.baseten.co/development/model/bdn> (BDN `weights` block; the documented Qwen3-8B pinned source is `hf://Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`)
- <https://docs.baseten.co/development/model/health-checks> (`/health` readiness/liveness endpoints)
- <https://docs.baseten.co/reference/truss-configuration> (Truss config reference)

## GPU resources (A10G-class 24 GB and A100 for 14B)

Used by: `profiles.yaml` (`gpu.instance_type`).

Docs:

- <https://docs.baseten.co/deployment/resources> (instance-type reference).
  `A10Gx8x32` = 1 × NVIDIA A10G, 24 GiB VRAM, 8 vCPU, 32 GiB RAM. A10G SKU
  format for a single GPU is `<GPU>x<vCPU>x<MEMORY>` (no colon), unlike
  `L4:4x16`.
- 14B fp16 (~28 GiB of weights) does not fit 24 GiB VRAM, so the
  `qwen3-14b` benchmark profile targets `A100:12x144` (1 × NVIDIA A100,
  80 GiB VRAM). Live 14B benchmarking is a follow-up.

## Inference (OpenAI-compatible chat completions)

Used by: `BasetenClient.infer`, `deployment.py :: chat_completions_url`.

| Interface | Shape |
| --- | --- |
| `POST {model_host}/environments/{environment}/sync/v1/chat/completions` | OpenAI-compatible body (`model`, `messages`, `max_tokens`, `temperature`, `stream`); streaming SSE with `data:` lines and a final `data: [DONE]` |

Docs:

- <https://docs.baseten.co/inference/calling-your-model> (server-side only; "Call your model from server-side code to avoid exposing your Baseten API key")
- <https://docs.baseten.co/examples/models/llm/qwen3> ("vLLM serves an OpenAI-compatible API"; base URL `https://model-{model_id}.api.baseten.co/environments/{environment}/sync/v1`)
- <https://docs.baseten.co/reference/inference-api/chat-completions>
- Standard synchronous predict endpoints (alternative): <https://docs.baseten.co/reference/inference-api/overview>

### Conflict / decision of record

- `voice_proto/model.py` currently appends `/v1/chat/completions` to
  `BASETEN_MODEL_URL` (a remembered legacy shape). The current documented
  OpenAI-compatible path is `/environments/{environment}/sync/v1/chat/completions`.
  **The documented interface wins** for this package; `voice_proto` is
  unchanged this slice and can adopt the new deployment later via
  `BASETEN_MODEL_URL` (its endpoint is credential-gated, not part of this
  slice).

## Scale-to-zero observation

Used by: `Harness.observe_scale_to_zero`.

`status == "SCALED_TO_ZERO"` or `active_replica_count == 0` from the same
`GET .../deployments/{deployment_id}` endpoint is the documented observable
signal. Observation is bounded (best-effort); nothing polls forever.

Docs:

- <https://docs.baseten.co/deployment/manage/scaling#scale-to-zero>
- <https://docs.baseten.co/deployment/autoscaling/overview> (scale-to-zero: `min_replica: 0`)

## SDK version verified

- `baseten` **0.10.0** on PyPI (<https://pypi.org/project/baseten/>) — the
  generated `managementapi` / `inferenceapi` clients were read to confirm the
  exact verbs, paths, success codes, and field names for wake, predict,
  deployment status, and autoscaling settings.
