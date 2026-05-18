# Local module: `system` (System Loop / Orchestrator)

## 📌 Responsibilities
*   Handle the main loop (`loop`) of the inference node.
*   Define the strict execution order, based on the pure contracts of `aideen-core`:
    1.  `Reasoning::step`
    2.  `Control::decide`
    3.  `Ethics::violates`
    4.  `Mathematical integration` (S ← S + tanh(α · Δ))

## 🚫 Restrictions (Constitutional)
*   Does **NOT** know local GPU implementation details.
*   Does **NOT** know the details of the port, sockets or the network protocol (delegates these tasks to `engine` and `network` respectively).
