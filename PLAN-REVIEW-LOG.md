# Plan Review Log: PLANETES 科学处理与 ESO 接口整理

Status: the user subsequently approved implementation on 2026-10-27. The text below records the planning stage. See docs/CHANGE_REPORT.md for the completed implementation.

Act 1 (grill) complete — 14 decisions locked with the user. MAX_ROUNDS=5.
PLAN_FILE=PLAN.md
LOG_FILE=PLAN-REVIEW-LOG.md

Interviewer and plan author: Codex. Reviewer: a separate Codex CLI session using its configured default model. This is an independent-session review; no cross-model claim is made without verifying the actual models.

Implementation has not started. Only these planning documents may be written in this phase.

Review runtime: codex-cli 0.155.1, configured default model, sandbox read-only.
Review session: 01a17cfb-15be-75e2-8829-6d5876d2d47f.


## Round 1 — Codex

The fourteen decisions are coherent and should remain settled. However, the following contract gaps need resolution before implementation. This review uses the current checkout, `38341e0`, and the external GUI; no files were modified or hardware accessed.

### 1. P1 — Calibration compatibility has no enforceable definition

`PLAN.md:59` requires compatible flat/wave/P2VM geometry and scientific roles, but the proposed fields do not specify how to establish that relationship. A wavelength solution depends on the flat’s extraction geometry; P2VM also depends on the extraction mask and flat response. Matching shapes, instrument order and wavelength coverage cannot distinguish incompatible calibrations. The current products contain no mandatory dependency identity (`planetes/product.py:33`).

**Fix:** Persist explicit wave→flat and P2VM→flat/wave dependency fingerprints, validate them at preparation boundaries, and test rejection of same-shaped but mismatched calibrations.

### 2. P1 — Independent darks need explicit roles within the P2VM task

“Each task explicitly selects a DarkCalibration” does not fully specify P2VM’s inputs (`PLAN.md:55`). Currently, telescope and baseline cubes share one dark, while the phase reference can use a different dark (`planetes/processing.py:46`, `planetes/processing.py:70`). The new Python signature and SOF contract must preserve that distinction after removing the flat’s embedded dark.

There is also an adapter conflict: `select_inputs()` permits one file per tag and determines CALIB/RAW classification by whether the tag equals a product category (`planetes/eso/common.py:33`). Role-specific dark tags would therefore need different classification logic.

**Fix:** Define separate main-calibration and phase-reference dark roles, their required/optional and explicit-reuse rules, and map each SOF role to its expected product category and CPL group.

### 3. P1 — The GUI’s dark-input contract remains unresolved

The selected GUI currently accepts a directory’s newest file, a raw cube, or a 2-D image; it reads generic FITS data and computes cube means itself (`/Users/zexwu/zexwu_lib/devel/camview_p2vm.py:687`). That does not establish whether a file is a schema-2 DarkCalibration, and its float32 mean calculation is not the offline chunked float64 accumulation.

A 2-D image also cannot supply measured temporal RMS. Keeping this loader while replacing only the reducer would leave a second scientific dark path, despite the shared-core requirement.

**Fix:** Specify that P2VM consumes a validated DarkCalibration through the public loader/shared dark task, explicitly resolve raw-capture and 2-D-image handling without fabricating RMS, and test rejection of a same-shaped wrong-category FITS file.

### 4. P2 — Opening extraction width exposes an existing rectangular-array assumption

The flat builder selects pixels using a strict spatial inequality, then constructs rectangular arrays directly from the per-output lists (`planetes/algorithms/flat.py:63`, `planetes/algorithms/flat.py:76`). Valid non-default widths can produce unequal list lengths.

A small read-only counterexample: width **3**, two columns, and straight traces at **y=4.5** versus **y=8.0** select **4 versus 6 pixels**. The existing array construction cannot represent those rows as the required numeric matrix. Edge-clipped traces create the same problem.

**Fix:** Pad unequal sparse profiles with the existing paired `(-1, -1)` sentinel, preserving selected pixels and accumulation order, and test non-default widths and detector-edge traces.

### 5. P2 — The expanded grid/GD parameters lack essential scientific semantics

The coordinate checks in `PLAN.md:59` are insufficient for all channel counts. One or two target channels leave respectively zero or one channel in the calibration’s central-band GD calculation (`planetes/algorithms/p2vm.py:66`); a single channel cannot identify group delay.

The calibration GD parameter also needs a defined scope: there are searches in the baseline fit and two phase-reference stages (`planetes/algorithms/p2vm.py:71`, `planetes/algorithms/p2vm.py:219`, `planetes/algorithms/p2vm.py:237`). Finally, current search bounds constrain only coarse initialization; Newton refinement can leave them (`planetes/numerics.py:59`, `planetes/numerics.py:81`). Silently turning them into final-result bounds would change the retained algorithm.

**Fix:** Define shared parameter types/domains, require at least two central-band wavelength samples and a nonempty finite GD grid, specify propagation through all calibration searches, and document unchanged unconstrained Newton refinement.

### 6. P2 — Schema migration omits the packaged wavelength-reference product

The packaged `wavelength_lines.fits` currently has `PLTSCHEM=1`. It is loaded as `WavelengthReference` through the same strict product decoder (`planetes/io/reference.py:17`, `planetes/io/products.py:176`). The plan introduces schema 2 and lists the new science models, but does not explicitly retain or migrate this static-product contract. Updating the shared decoder alone would break the default wave task.

**Fix:** Explicitly include the static line-list contract in the migration, regenerate its FITS from the retained ECSV without changing wavelengths, and run the installed-wheel wave test using the default packaged reference.

These fixes fit within the proposed small package; none requires a workflow framework or reversal of the fourteen decisions.

VERDICT: REVISE

### Plan author's response

Accept all six findings as contract clarifications within the fourteen settled decisions.

1. Add enforceable wave-to-flat and P2VM-to-flat/wave dependency identities/fingerprints and same-shape mismatch tests. Allow science dark to differ from flat's dark by explicit role.
2. Specify main calibration dark and independent optional phase-reference dark; use explicit input-role/category/group tables instead of classifying frames solely by tag text.
3. Reopen one user-facing decision: whether an explicitly selected 2-D dark may be used for subtraction without measured RMS/bad-pixel calibration, or whether P2VM must pause until a complete dark calibration/3-D dark cube is selected. Do not fabricate temporal statistics. Shared processing handles raw cubes; wrong-category files must be rejected.
4. Pad sparse profiles using paired (-1, -1), preserve masks and addition order, and test non-default widths and detector edges.
5. Specify parameter domains, central-band sample requirements and coarse-GD semantics; pass calibration GD settings through baseline and phase-reference searches without constraining Newton updates.
6. Migrate and test the packaged static reference, retaining the ECSV and exact wavelength values.

Additional author checks to incorporate in the same revision: define the partial in-memory realtime result versus the complete reduced FITS contract, and make the atomic no-overwrite publication test explicit.

No implementation changes have been made. The next revision and resumed review wait for the additional dark-input decision.

## Additional grill decision 15

The user selected A: keep a 2-D dark for scientific subtraction, label it as subtraction-only, and do not invent temporal RMS or bad-pixel statistics. Flat construction still requires a complete dark calibration.

### Revision submitted for round 2

All six round-1 findings are incorporated in PLAN.md. The revision specifies scientific-content dependency fingerprints, explicit SOF role/category/group rules, main/phase dark reuse, DarkImage versus full DarkCalibration and shared loading, paired-sentinel padding without changing masks, parameter domains and unchanged GD Newton semantics, and schema-2 static-reference migration. It also distinguishes partial realtime VisibilityObservables from complete ReducedObservables and makes atomic no-overwrite failure tests concrete.

The original fourteen choices remain unchanged. No implementation or external GUI file has been modified.

## Round 2 — Codex

All six prior findings are resolved **at the plan level**:

- **Calibration compatibility:** Dependency fingerprints, canonical array encoding, recomputation and read-only preparation snapshots make the compatibility checks enforceable (`PLAN.md:48`).
- **Dark roles:** Main versus phase-reference darks, explicit reuse, invalid combinations and SOF role/category mappings are specified (`PLAN.md:61`, `PLAN.md:85`).
- **GUI dark loading:** The shared loader distinguishes validated products, raw cubes and subtraction-only `DarkImage` inputs. Decision 15 is respected without fabricated RMS/BPM; flat still requires complete `DarkCalibration` (`PLAN.md:44`, `PLAN.md:60`, `PLAN.md:101`).
- **Ragged extraction profiles:** Paired sentinel padding preserves selected pixels and accumulation order, with targeted counterexamples in acceptance tests (`PLAN.md:63`, `PLAN.md:109`).
- **Grid/GD parameters:** Domains exclude the identified degenerate channel counts; calibration settings reach all GD searches, while Newton refinement retains its existing unconstrained semantics (`PLAN.md:69`, `PLAN.md:80`).
- **Static reference migration:** Schema-2 regeneration, exact wavelength preservation, origin labels and the installed-wheel default-reference path are covered (`PLAN.md:52`, `PLAN.md:118`).

The partial/full observable distinction and no-overwrite publication fault tests also close the associated acceptance gaps (`PLAN.md:51`, `PLAN.md:111`).

After rechecking the relevant code boundaries, I find no remaining material contradiction with the fifteen settled decisions. The plan is sufficiently concrete to implement without adding a framework or reopening those preferences.

VERDICT: APPROVED

### Plan author

Accepted. All six prior contract findings are resolved; no further implementation change is introduced at this planning stage.

## Resolution

APPROVED after 2 of MAX_ROUNDS=5 independent-session read-only review rounds. Fifteen user decisions are locked.

Approved PLAN.md SHA256: 5ec521026c3cb8045992bfa20a853a0a24dc221df142149889dfa7a473dc848f

Awaiting the user's final implementation sign-off under grill-me-codex. Only PLAN.md and this review log were created; package code, external GUI, raw data and existing scientific products are unchanged.
