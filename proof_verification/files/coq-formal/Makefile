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
COQFLAGS := -Q . PNSFormal -q

# Dependency order matters: Limits and DerivKit are shared prerequisites.
VFILES := Limits.v \
          DerivKit.v \
          StabilityAlgebra.v \
          ContinuityAlgebra.v \
          TauDesign.v \
          Asymptotics.v \
          ManufacturedSolution.v \
          PlateauBump.v

VOFILES := $(VFILES:.v=.vo)

MODULES := PNSFormal.StabilityAlgebra \
           PNSFormal.ContinuityAlgebra \
           PNSFormal.TauDesign \
           PNSFormal.Asymptotics \
           PNSFormal.ManufacturedSolution \
           PNSFormal.PlateauBump

all: $(VOFILES)

%.vo: %.v
	$(COQC) $(COQFLAGS) $<

# Explicit intra-project dependencies.
StabilityAlgebra.vo:     Limits.vo
TauDesign.vo:            Limits.vo
Asymptotics.vo:          Limits.vo
ManufacturedSolution.vo: DerivKit.vo
PlateauBump.vo:          Limits.vo DerivKit.vo

check: all
	$(COQCHK) -Q . PNSFormal -silent -o $(MODULES)
	@echo "coqchk: all modules kernel-verified."

clean:
	rm -f *.vo *.vos *.vok *.glob .*.aux

.PHONY: all check clean
