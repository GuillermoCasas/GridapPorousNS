# ---------------------------------------------------------------------------
#  Makefile for the Coq formalisation (PNSFormal).
#
#  Requires only a Coq installation (coqc, and optionally coqchk), version
#  8.16--8.20; developed and tested against 8.18.0.  No external libraries.
#
#  Targets:
#    make          -- compile all files in dependency order
#    make check    -- additionally re-verify all proofs with the trusted
#                     kernel checker coqchk (slower, strongest guarantee)
#    make clean    -- remove compiled artefacts
# ---------------------------------------------------------------------------

COQC    := coqc
COQCHK  := coqchk
COQDEP  := coqdep
COQFLAGS := -Q . PNSFormal -q
# coqdep resolves the same `From PNSFormal Require' graph through the -Q logical
# mapping, but it does NOT accept coqc-only flags such as -q, so it is given its
# own flag set rather than reusing COQFLAGS.
COQDEPFLAGS := -Q . PNSFormal

# [known-fragility]  The file list lives in _CoqProject and is READ from there,
# never restated.  It used to be duplicated here and in run_all.sh; when
# AbstractInterpolation moved ahead of AbstractStability (the latter now DEFINES
# B_S from the former's eighteen-term expression instead of assuming it), the
# copies drifted and run_all.sh stopped compiling on a clean tree while `make`
# stayed green.  One list, one source of truth.
#
# The SAME rule governs MODULES below, and for a while it did not: MODULES was a
# second restated literal that had silently drifted to twelve of the fifteen
# modules.  Limits and DerivKit were still covered transitively (Asymptotics,
# ManufacturedSolution and PlateauBump Require them), but NonVacuity is a LEAF --
# nothing Requires it -- so `make check` never kernel-checked the one file that
# certifies the hypothesis bundle is consistent.  Derive, never restate.
VFILES := $(shell sed -n 's/^\([A-Za-z0-9_]*\.v\)[[:space:]]*$$/\1/p' _CoqProject)

ifeq ($(strip $(VFILES)),)
$(error No .v files found in _CoqProject -- refusing to "succeed" by building nothing)
endif

VOFILES := $(VFILES:.v=.vo)

MODULES := $(VFILES:%.v=PNSFormal.%)

all: $(VOFILES)

%.vo: %.v
	$(COQC) $(COQFLAGS) $<

# Intra-project dependencies are DERIVED by coqdep, never hand-restated -- the
# same "derive, never restate" rule that governs VFILES and MODULES above.
#
# [known-fragility]  Two hazards were fixed here, in order.
#
#   (1) A hand-maintained literal edge list (`AbstractStability.vo: ...
#       AbstractInterpolation.vo', and so on).  It records only the `From
#       PNSFormal Require' edges someone remembered, so every MISSING edge is
#       invisible to `make' precisely because the SERIAL order of VFILES in
#       _CoqProject happens to compile the prerequisite first.  That makes file
#       order the sole guarantor of correctness and `make -j' UNSAFE.  coqdep
#       reads the REAL Require graph from the sources, so the edge set can never
#       fall behind the code.  (The old list named none of the two
#       NonVacuity{Interp,Conv} witnesses' eleven edges.)
#
#   (2) Loading coqdep's graph via `-include .coq-deps' plus a rule that remakes
#       it.  On a COLD tree .coq-deps is absent, so GNU make remakes it and then
#       RESTARTS -- it reads the makefiles a SECOND time ("Re-executing[1]") to
#       pick up the freshly written edges.  That restart is a race window under
#       `-j': make can schedule a .vo against the still-edgeless graph before the
#       restart fences the goal targets, so a file is compiled before a
#       prerequisite .vo exists.  It is intermittent and timing-dependent
#       (signatures: "Cannot find a physical path bound to logical path
#       StabilityAlgebra"; "DerivKit.vo: No such file or directory").  Serial
#       `make', `-j1' and run_all.sh never raced -- file order alone sufficed --
#       so the bug was specifically `make -j' from a clean tree.
#
# Fix for (2): generate the graph at PARSE TIME with $(shell), then load it with
# a plain `include' -- there is NO rule that remakes .coq-deps, hence no restart.
# make's parse phase is single-threaded and finishes BEFORE any target is
# scheduled, so the complete inter-.vo edge set is already in the database when
# the first .vo is considered; `-j' has no edgeless window to race on.  coqdep
# runs once per invocation (~1s for these files), which also stops the graph
# ever going stale.
#
# Verified empirically (a single green `-j' run cannot prove the absence of a
# race, so the check is a REPEATED cold-start loop): `make -n --debug=m' on a
# cold tree reads the makefiles exactly ONCE ("Re-executing" count 0), and
# `for i in 1..N; do make clean && make -jN; done' at N in {2,8,16} is green on
# every iteration.  Because the parse-time $(shell) leaves no restart window, the
# edge set is complete before scheduling and there is nothing left to race on;
# the loop is confirmation, not the mechanism.  Approach kept over the honest
# fallback `.NOTPARALLEL:' precisely because it verifies clean.
#
# The $(shell) is SKIPPED for `make clean' (nothing to compile, and no point
# regenerating a file about to be deleted).  .coq-deps is a build artefact:
# `make clean' removes it and it is gitignored (root .gitignore, "Coq proof
# verification auxiliary outputs").
ifeq ($(filter clean,$(MAKECMDGOALS)),)
COQDEP_OK := $(shell $(COQDEP) $(COQDEPFLAGS) $(VFILES) > .coq-deps 2>/dev/null && echo ok)
ifneq ($(COQDEP_OK),ok)
$(error coqdep failed to compute the dependency graph into .coq-deps -- refusing to build against an unknown edge set (is coqdep installed and on PATH?))
endif
include .coq-deps
endif

check: all
	$(COQCHK) -Q . PNSFormal -silent -o $(MODULES)
	@echo "coqchk: all modules kernel-verified."

# `clean' removes every COMPILATION artefact so a subsequent build (and, above
# all, run_all.sh) starts from a genuinely cold tree -- see run_all.sh invariant
# (2): a leftover .vo from an older source revision can satisfy a later Require
# and turn a real breakage into a false [PASS].
#
#   *.vo *.vos *.vok  -- coqc object / native / kernel-check outputs
#   *.glob            -- cross-reference tables
#   .*.aux            -- coqc per-file auxiliary state
#   .coq-deps         -- the coqdep graph regenerated at parse time (see above)
#
# NOT removed -- `.*.cache` (the Coq micromega SOS caches: .lia.cache, .nia.cache,
# .nra.cache written by the lia/nia/nra tactics).  This is DELIBERATE, for two
# reasons, and matching a "fix" that deletes them would be a regression:
#
#   (i)  It is a content-addressed cache, not stale state.  Each entry stores a
#        Positivstellensatz certificate keyed by the exact arithmetic goal, and
#        coqc re-runs the trusted kernel over that certificate on EVERY build --
#        so a surviving .nra.cache can never manufacture a false [PASS] the way a
#        surviving .vo can.  It is therefore outside the scope of invariant (2).
#   (ii) Deleting it forces every nra/nia goal to re-derive its certificate from
#        scratch via the (slow) SOS search on the next build -- measured here at
#        roughly 15x the compile time (minutes, not seconds), on every clean
#        rebuild.  run_all.sh's own pre-run cleanup deliberately preserves it for
#        the same reason (it removes only .vo/.vos/.vok/.glob/.aux); `make clean'
#        matches that policy so the two entry points behave identically.
#
# The caches are gitignored (root .gitignore) so they never pollute `git status'.
clean:
	rm -f *.vo *.vos *.vok *.glob .*.aux .coq-deps

.PHONY: all check clean
