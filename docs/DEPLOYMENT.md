# Deployment planning — voice intake service

> Research/planning estimate only. Prices, service availability, BAA terms,
> concurrency, regional telephony rates, and transcription quality must be
> revalidated with vendors before any deployment. This is not a production
> clinical-system design or a clinical safety claim.

## Proposed call path

```text
Caller
  -> AWS Connect
  -> AWS Transcribe
  -> AWS orchestration
  -> Baseten Qwen + aLoRA adapters
  -> Rime Mist
  -> AWS Connect
  -> Caller
```

Keep AWS and Baseten under BAAs. Rime's publicly listed BAA availability is
Enterprise-only; confirm contractual scope and pricing directly with Rime.

## Estimated variable cost per live call-minute

| Component | Approximate cost |
| --- | ---: |
| Amazon Connect voice | **$0.0180/min** |
| US inbound telephony | **$0.0022/min** |
| AWS Transcribe streaming | **$0.0240/min** at low-volume Tier 1 |
| Rime Mist | **~$0.007–0.010/min of call** |
| Baseten GPU | **~$0.0004–0.001/min** amortized |
| AWS orchestration | negligible |
| **Total** | **~$0.052–0.055/call-minute** |

Rime's public pricing is approximately $0.03 per **minute of generated
speech**, rather than per minute of call duration. If the assistant speaks for
roughly 25–35% of a call, that is approximately $0.0075–0.0105 per call-minute
at public pricing. Enterprise/BAA pricing is custom.

Amazon Connect Basic voice is estimated at $0.018/min plus approximately
$0.0022/min for inbound US DID telephony in `us-west-2`. Standard Transcribe
Tier 1 streaming is estimated at about $0.024/min at lower volume, with
material volume discounts available.

**Planning estimate:** a 15-minute voice intake is approximately **$0.80** in
variable infrastructure cost.

## Baseten GPU starting point

For an approximately 8B quantized Qwen deployment, start with:

```text
1 × NVIDIA L4, 24 GB VRAM
```

Planning estimate for Baseten L4 capacity:

```text
$0.01414/min = $0.848/hour
```

Suggested autoscaling schedule:

```text
Clinic hours:   min replicas = 1
                max replicas = 2 or 3

After hours:    min replicas = 0
                max replicas = 2 or 3
```

At roughly 10 clinic hours/day × 22 days/month, one warm L4 costs:

```text
220 hours × $0.848 ≈ $187/month
```

Additional GPUs incur cost only while autoscaled up.

## Concurrency assumptions

Until the aLoRA cache-fork architecture is benchmarked, conservatively design
for about **20 simultaneous active voice interviews per 24 GB L4** for an
8B-class model.

At 20 concurrent calls:

```text
$0.848/hour ÷ 20 = $0.0424 per caller-hour
                    = $0.00071 per caller-minute
```

The GPU is therefore not expected to dominate variable cost; telephony and
speech-to-text are likely to dominate. For 40+ simultaneous calls, add a
horizontal replica. A two-L4 instance provides 48 GB aggregate VRAM, but prefer
horizontal replicas unless benchmarking demonstrates a need for a larger shared
memory footprint.

## Transcription cost decision

Amazon Transcribe Medical is estimated around **$0.075/minute**, compared with
approximately **$0.024/minute** for standard streaming. A 15-minute call would
therefore add roughly **$0.77** if Medical is used.

Before selecting it, benchmark:

```text
standard Transcribe + custom medical vocabulary
vs.
Transcribe Medical
```

Measure terminology accuracy, clinically relevant transcription errors,
latency, and total cost. Medical transcription accuracy may justify the higher
price for this use case, but should not be assumed.

## Rough planning model

```text
~$200/month fixed warm-GPU capacity
+ ~$0.80 per 15-minute voice intake
+ automatic GPU scale-out beyond one L4's benchmarked safe concurrency
```

## Sources to revalidate

- [Baseten BAA](https://www.baseten.co/baa/)
- [Rime pricing](https://www.rime.ai/pricing)
- [Amazon Connect pricing appendix](https://aws.amazon.com/products/connect/customer/pricing/appendix/)
- [Amazon Transcribe pricing](https://aws.amazon.com/transcribe/pricing/)
- [Baseten cloud pricing](https://www.baseten.co/pricing/)
- [Baseten deployment resources](https://docs.baseten.co/deployment/resources)
