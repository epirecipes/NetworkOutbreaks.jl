/-!
# NetworkOutbreaks event-semantics invariants

These proofs formalise the count-level invariants that every SSA algorithm in
`NetworkOutbreaks.jl` must preserve.  They intentionally avoid stochastic
claims; the checked facts are deterministic consequences of applying one
compartment-changing event to a compartment-count snapshot.
-/

namespace NetworkOutbreaksProofs.EventSemantics

/-- Count snapshot for SIS trajectories. -/
structure SISCounts where
  S : Int
  I : Int

/-- Count snapshot for SIR trajectories. -/
structure SIRCounts where
  S : Int
  I : Int
  R : Int

namespace SISCounts

def total (c : SISCounts) : Int := c.S + c.I

def infection (c : SISCounts) : SISCounts :=
  { S := c.S - 1, I := c.I + 1 }

def recovery (c : SISCounts) : SISCounts :=
  { S := c.S + 1, I := c.I - 1 }

def valid (c : SISCounts) : Prop :=
  0 ≤ c.S ∧ 0 ≤ c.I

theorem infection_preserves_total (c : SISCounts) :
    (infection c).total = c.total := by
  simp [total, infection]
  omega

theorem recovery_preserves_total (c : SISCounts) :
    (recovery c).total = c.total := by
  simp [total, recovery]
  omega

theorem infection_preserves_valid (c : SISCounts)
    (hvalid : c.valid) (hS : 1 ≤ c.S) :
    (infection c).valid := by
  rcases hvalid with ⟨_, hI⟩
  simp [valid, infection]
  omega

theorem recovery_preserves_valid (c : SISCounts)
    (hvalid : c.valid) (hI : 1 ≤ c.I) :
    (recovery c).valid := by
  rcases hvalid with ⟨hS, _⟩
  simp [valid, recovery]
  omega

end SISCounts

namespace SIRCounts

def total (c : SIRCounts) : Int := c.S + c.I + c.R

def infection (c : SIRCounts) : SIRCounts :=
  { S := c.S - 1, I := c.I + 1, R := c.R }

def recovery (c : SIRCounts) : SIRCounts :=
  { S := c.S, I := c.I - 1, R := c.R + 1 }

def valid (c : SIRCounts) : Prop :=
  0 ≤ c.S ∧ 0 ≤ c.I ∧ 0 ≤ c.R

theorem infection_preserves_total (c : SIRCounts) :
    (infection c).total = c.total := by
  simp [total, infection]
  omega

theorem recovery_preserves_total (c : SIRCounts) :
    (recovery c).total = c.total := by
  simp [total, recovery]
  omega

theorem infection_preserves_valid (c : SIRCounts)
    (hvalid : c.valid) (hS : 1 ≤ c.S) :
    (infection c).valid := by
  rcases hvalid with ⟨_, hI, hR⟩
  simp [valid, infection]
  omega

theorem recovery_preserves_valid (c : SIRCounts)
    (hvalid : c.valid) (hI : 1 ≤ c.I) :
    (recovery c).valid := by
  rcases hvalid with ⟨hS, _, hR⟩
  simp [valid, recovery]
  omega

/-- If the initial seed allocation consumes `seeded` nodes from a population of
`N`, filling the unassigned nodes into a default compartment preserves `N`. -/
theorem seed_fill_preserves_total (N seeded : Int) :
    seeded + (N - seeded) = N := by
  omega

end SIRCounts

end NetworkOutbreaksProofs.EventSemantics

