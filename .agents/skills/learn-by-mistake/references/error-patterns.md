# Error Patterns Reference

> Detection patterns, root causes, and fix strategies for common errors.
> Used by learn-by-mistake to classify and deduplicate errors.

---

## build

### Missing module / import not found
- **Regex:** `Cannot find module '([^']+)'|ModuleNotFoundError: No module named '([^']+)'|error\[E0432\]: unresolved import`
- **Root cause:** Package not installed, typo in import path, or missing `exports` field in package.json.
- **Fix:** Install the missing package or correct the import path.
- **Learnable:** Yes

### Type mismatch / compilation error
- **Regex:** `Type '([^']+)' is not assignable to type '([^']+)'|expected .+, found .+|incompatible types`
- **Root cause:** Passing a value of the wrong type; API contract changed after an upgrade.
- **Fix:** Align types at the call-site or update the type definition.
- **Learnable:** Yes

### Syntax error
- **Regex:** `SyntaxError:|error: expected .+ but found|parse error|unexpected token`
- **Root cause:** Malformed code — missing bracket, comma, semicolon, etc.
- **Fix:** Fix the syntax at the reported location.
- **Learnable:** No (usually a one-off typo)

### Dependency version conflict
- **Regex:** `ERESOLVE|Could not resolve dependency|version solving failed|incompatible version`
- **Root cause:** Two packages require conflicting versions of a shared dependency.
- **Fix:** Align versions, use overrides/resolutions, or upgrade the conflicting packages.
- **Learnable:** Yes

### Build tool misconfiguration
- **Regex:** `Configuration error|Invalid option|Unknown compiler option|webpack .+ error`
- **Root cause:** Wrong or outdated setting in build config (tsconfig, webpack, vite, cargo, etc.).
- **Fix:** Correct the config key/value per the tool's docs.
- **Learnable:** Yes

---

## test

### Assertion failure
- **Regex:** `Expected .+ but received .+|AssertionError|assert.*failed|expect\(.*\)\.(toBe|toEqual)`
- **Root cause:** Code behavior drifted from the test expectation, or the test is wrong.
- **Fix:** Update the code or the test to match the intended behavior.
- **Learnable:** Yes

### Async timeout
- **Regex:** `Timeout - Async callback was not invoked|exceeded timeout of \d+ms|TimeoutError`
- **Root cause:** Missing `await`, unresolved promise, or genuinely slow operation without increased timeout.
- **Fix:** Add missing `await`, resolve the promise, or increase the test timeout.
- **Learnable:** Yes

### Mock / stub misconfiguration
- **Regex:** `is not a function|mockImplementation|Cannot spy .+ because it is not a function|stub .+ not called`
- **Root cause:** Mock target doesn't match actual module path, or mock wasn't restored between tests.
- **Fix:** Verify mock path matches import path; add `afterEach(jest.restoreAllMocks)`.
- **Learnable:** Yes

### Flaky test (race condition)
- **Regex:** `intermittent|flaky|sometimes fails|order-dependent`
- **Root cause:** Shared mutable state between tests, or timing-dependent assertions.
- **Fix:** Isolate state per test, use deterministic waits instead of fixed delays.
- **Learnable:** Yes (pattern, not specific instance)

---

## runtime

### Null / undefined reference
- **Regex:** `TypeError: Cannot read propert(y|ies) of (null|undefined)|NoneType has no attribute|NullPointerException|unwrap.*None`
- **Root cause:** Accessing a property on a value that is null/undefined/None.
- **Fix:** Add a null check, use optional chaining, or fix the data source.
- **Learnable:** Yes

### Unhandled promise rejection
- **Regex:** `UnhandledPromiseRejection|Unhandled promise rejection|unhandled rejection`
- **Root cause:** An async error without a `.catch()` or `try/catch` around `await`.
- **Fix:** Add error handling to the async call chain.
- **Learnable:** Yes

### Out of memory
- **Regex:** `FATAL ERROR: .+ JavaScript heap out of memory|MemoryError|OOM|OutOfMemoryError`
- **Root cause:** Unbounded data accumulation, large file loaded entirely into memory, or memory leak.
- **Fix:** Stream data, paginate, increase heap size, or fix the leak.
- **Learnable:** Yes

### Stack overflow
- **Regex:** `Maximum call stack size exceeded|StackOverflowError|stack overflow|thread .+ has overflowed its stack`
- **Root cause:** Infinite or excessively deep recursion without a base case.
- **Fix:** Add/fix the base case or convert to an iterative approach.
- **Learnable:** Yes

---

## types

### Type assignability error
- **Regex:** `Type '([^']+)' is not assignable to type '([^']+)'|TS2322|TS2345`
- **Root cause:** Value shape doesn't match the declared type.
- **Fix:** Correct the value, widen the type, or add a type guard.
- **Learnable:** Yes

### Generic constraint violation
- **Regex:** `does not satisfy the constraint|TS2344|trait bound .+ is not satisfied`
- **Root cause:** Type argument doesn't meet the generic's `extends`/trait constraint.
- **Fix:** Pass a type that satisfies the constraint or relax the constraint.
- **Learnable:** Yes

### Missing type definitions
- **Regex:** `Could not find a declaration file for module|TS7016|Cannot find type definition`
- **Root cause:** Package lacks built-in types and no `@types/*` package is installed.
- **Fix:** Install `@types/<package>` or declare the module manually.
- **Learnable:** Yes

---

## linter

### Unused variable
- **Regex:** `is declared but .+ never (used|read)|no-unused-vars|unused variable|dead_code`
- **Root cause:** Leftover from refactoring or premature declaration.
- **Fix:** Remove the variable or prefix with `_` if intentionally unused.
- **Learnable:** No (cleanup noise)

### Import order
- **Regex:** `import/order|Import statements must be sorted|unsorted imports`
- **Root cause:** Auto-import placed the statement in the wrong group.
- **Fix:** Run the auto-fixer (`eslint --fix`, `isort`, etc.).
- **Learnable:** No (auto-fixable)

### Style violation
- **Regex:** `prettier|formatting|indent|trailing (space|comma|whitespace)|max-len`
- **Root cause:** Code doesn't match the project's style config.
- **Fix:** Run the formatter.
- **Learnable:** No (auto-fixable)

---

## git

### Merge conflict
- **Regex:** `CONFLICT \(content\)|Automatic merge failed|merge conflict`
- **Root cause:** Two branches modified the same lines.
- **Fix:** Manually resolve the conflict markers, then stage and commit.
- **Learnable:** Yes (if same files keep conflicting)

### Diverged branches
- **Regex:** `have diverged|Your branch and .+ have diverged`
- **Root cause:** Local and remote both have commits the other doesn't.
- **Fix:** Rebase or merge to reconcile the histories.
- **Learnable:** Yes

### Detached HEAD
- **Regex:** `HEAD detached at|You are in 'detached HEAD' state`
- **Root cause:** Checked out a commit/tag directly instead of a branch.
- **Fix:** Create a branch from the current state or checkout an existing branch.
- **Learnable:** No (intentional workflow)

### Permission denied on push
- **Regex:** `Permission denied|permission to .+ denied|fatal: Could not read from remote`
- **Root cause:** SSH key not configured, token expired, or no write access.
- **Fix:** Re-authenticate or request repository access.
- **Learnable:** No (transient / credentials)

---

## dependency

### Package not found
- **Regex:** `404 Not Found|No matching version|no matching package|package .+ was not found`
- **Root cause:** Package name typo, private registry not configured, or package was unpublished.
- **Fix:** Verify the package name and registry configuration.
- **Learnable:** Yes

### Version incompatibility
- **Regex:** `requires a peer of .+ but none is installed|engine .+ is incompatible|requires node`
- **Root cause:** Package needs a different runtime or peer dependency version.
- **Fix:** Upgrade/downgrade the runtime or install the required peer.
- **Learnable:** Yes

### Peer dependency warning
- **Regex:** `WARN .+peer dep|peerDependencies|peer dependency .+ not installed`
- **Root cause:** A package expects a peer that isn't listed in your dependencies.
- **Fix:** Install the peer at the required version.
- **Learnable:** Yes

### Lock file conflict
- **Regex:** `EINTEGRITY|Lock file .+ conflict|integrity checksum failed|ERR_PNPM_LOCKFILE`
- **Root cause:** Lock file out of sync with package.json, or corrupted cache.
- **Fix:** Delete the lock file and `node_modules`, then reinstall cleanly.
- **Learnable:** Yes

---

## config

### Environment variable missing
- **Regex:** `is not defined|env .+ not set|Missing required .+ environment|Error: .+_KEY is required`
- **Root cause:** `.env` file missing, not loaded, or variable name typo.
- **Fix:** Add the variable to `.env` or the deployment config.
- **Learnable:** Yes

### Port already in use
- **Regex:** `EADDRINUSE|address already in use|port \d+ is already in use|bind: address already in use`
- **Root cause:** Another process is occupying the port.
- **Fix:** Kill the other process or change the port.
- **Learnable:** No (transient)

### File permission error
- **Regex:** `EACCES|Permission denied|PermissionError|EPERM`
- **Root cause:** Process lacks read/write/execute permission on the target file.
- **Fix:** Fix file permissions (`chmod`/`chown`) or run with appropriate privileges.
- **Learnable:** No (transient)

### Path resolution error
- **Regex:** `ENOENT|No such file or directory|FileNotFoundError|path .+ does not exist`
- **Root cause:** Hardcoded or miscalculated path; file was moved or never created.
- **Fix:** Use relative paths from a known root, or verify the file exists before access.
- **Learnable:** Yes
