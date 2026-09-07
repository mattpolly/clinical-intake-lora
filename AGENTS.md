# Project instructions

- The customer speaks to the configured Agentic Devshop lead in ordinary language.
- Keep feature records in features/<feature-id>.md.
- The lead plans small, observable MVP tasks and uses independent review only when it has a clear benefit.
- Do not require the customer to operate workflow state or recovery commands.

## Cost control

- Default to direct lead implementation for small, medium, or low-risk work. Use the multi-role flow only on an explicit customer request or a large, complex, or high-risk score.
- Dispatch scoring once per requirements hash. Fill Decisions and assumptions and Definition of enough before the first score.
- Make one multi-role dispatch attempt per feature. Any re-run requires explicit customer approval and must state the expected request cost.
- Poll a long-running dispatch at most three times; verify delivered work locally, because tests are free.
- Close every feature record with one tally line: `planned / actual / waste / accepted` requests, using the spend tally.
