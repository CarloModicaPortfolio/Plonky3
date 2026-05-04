//! HVZK variant of the WHIR sumcheck (Construction 6.3 of eprint 2026/391).
//!
//! Companion to [`super::single`]: same reduction, but with `k` random
//! univariate masks committed under a ZK encoding so that the prover's `k`
//! round-polynomials no longer leak linear functions of the secret message.
//!
//! ⚠️ This file is a **scaffold**. Type shape, decision block, and constructor
//! signatures are stable; method bodies are stubbed with `unimplemented!()`
//! and land in subsequent commits on PR #1605.
//!
//! See `paper-db-2026-391/21-section-6-zk-sumcheck.md` for the protocol spec
//! and `22-section-6.1-zk.md` for the HVZK proof sketch.
//!
//! # Decision block (frozen for this PR)
//!
//! 1. **Architecture** — `pub struct ZkSumcheck;` (namespace, mirrors
//!    [`super::single::SingleSumcheck`]) plus
//!    [`ZkSumcheckProver`]`<F, EF, Enc>` (state, mirrors
//!    [`super::strategy::SumcheckProver`] plus mask bookkeeping). Considered
//!    alternative: WizardOfMenlo/whir#241 used a runtime `mask_length: usize`
//!    flag on the existing `Config` and branched at runtime; we picked the
//!    type-split to match Plonky3's existing namespace+state pattern and keep
//!    `single.rs` byte-frozen.
//!
//! 2. **MVP scope** — only [`ZkSumcheck::new_classic_unpacked`] in this PR.
//!    SIMD-packed (`new_classic_packed`), SVO (`new_svo`), and the dispatcher
//!    (`new`) are follow-up PRs that hang off the same `ZkSumcheck` namespace
//!    and reuse `ZkSumcheckProver`. SVO is structurally compatible: only the
//!    `(c0, c_inf)` computation of the plain piece differs.
//!
//! 3. **Transcript ordering** — for each mask oracle: MMCS-commit the
//!    codeword and observe the commitment (binds the mask to ε); after all
//!    k masks: observe `μ̃`; sample `ε`; per round j: observe `ĥ_j`
//!    (`ℓ_zk - 1` field elements; see #4); grind; sample `γ_j`. The
//!    protocol pins the relative order of prover-sends and verifier-samples
//!    (Construction 6.3, with HVZK forcing `μ̃` before `ε` and RBR forcing
//!    `ĥ_j` before `γ_j`); the per-message observe granularity is a
//!    Fiat-Shamir realization choice and we pick granular for debuggability.
//!    Mask oracles are bound by their MMCS commitment rather than by
//!    absorbing codeword bytes — this is the same pattern as
//!    [`p3_fri::HidingFriPcs`] and keeps `ZkEncoding::Codeword` opaque
//!    (preserving the dependency isolation set up in #1601).
//!
//! 4. **Wire format — skip the linear coefficient.** Per round, send
//!    `(c0, c2, c3, …, c_{ℓ_zk-1})`; verifier derives `c1` from the affine
//!    consistency check `ĥ_j(0) + ĥ_j(1) = ε·μ + μ̃` (round 1) or
//!    `= ĥ_{j-1}(γ_{j-1})` (subsequent rounds). Matches Lemma 6.4's affine
//!    subspace dimension `1 + k·(ℓ_zk - 1)` and mirrors the existing
//!    plain-path convention in [`super::single`].
//!
//! 5. **Field constraints** — `char(F) ≠ 2` (Lemma 6.4 rank-nullity argument)
//!    and `ℓ_zk ≥ 2`. Enforced via runtime `assert!` in the constructor for
//!    now; promote to compile-time when/if `Field` exposes characteristic as
//!    a `const`.
//!
//! 6. **Round-loop sharing** — the per-round arithmetic is duplicated from
//!    `super::single` in this PR (≈20 LoC). Refactor into a shared
//!    `sumcheck::core` helper in a follow-up PR. Rationale: keeps `single.rs`
//!    byte-frozen for free, narrows review surface, defers the cross-cutting
//!    refactor to when both paths are stable.
//!
//! # Per-round formula (Construction 6.3 step 4(a))
//!
//! For round `j` (1-indexed), with `γ = (γ_1, …, γ_{j-1})` already sampled.
//! The running `sum_future_endpoints` field follows a uniform "decrement at
//! the *start* of each round" convention so there is no special case for
//! entry into round 1: see field doc on [`ZkSumcheckProver::sum_future_endpoints`].
//!
//! At the start of round `j`, before building `ĥ_j`:
//! - `sum_future_endpoints -= s_j(0) + s_j(1)`.
//!   At this point `sum_future_endpoints == Σ_{l > j} (s_l(0) + s_l(1))`.
//!
//! Then build `ĥ_j(X)`:
//!
//! ```text
//! ĥ_j(X) = 2^{k-j}   · s_j(X)                                  (live mask)
//!        + 2^{k-j}   · Σ_{l < j} s_l(γ_l)                      (past masks, cached)
//!        + 2^{k-j-1} · sum_future_endpoints                    (future masks, running)
//!        + ε         · plain_piece(X)                          (base sumcheck round)
//! ```
//!
//! After observing `ĥ_j` (minus c1) and sampling `γ_j` the prover:
//! - pushes `s_j(γ_j)` onto `mask_evals_at_gamma`,
//! - calls `base.fix_prefix_var_mut(γ_j)`.
//!
//! No after-round bookkeeping for `sum_future_endpoints` — the next round's
//! start-decrement handles the transition.
//!
//! The closed-form `μ̃ = 2^{k-1} · sum_future_endpoints_initial` is checked
//! against the naive `Σ_{b ∈ {0,1}^k} ŝ(b)` form via `debug_assert!` in the
//! constructor (catches the multiplicity bug class that the reference impl
//! shipped at first; see `_search_log.md`).
//!
//! # References
//!
//! - eprint 2026/391, §6 Construction 6.3, Lemma 6.4 (HVZK), Lemma 6.5 (RBR).
//! - Local digest: `paper-db-2026-391/21-section-6-zk-sumcheck.md`,
//!   `22-section-6.1-zk.md`, `23-section-6.2-rbr.md`.
//! - Reference impl: WizardOfMenlo/whir#241 (merged 2026-03-31).
//! - Issue: Plonky3/Plonky3#1586 (HVZK-WHIR 3/6).
//! - Depends on: #1584/#1601 (ZK encoding traits in `p3-zk-codes`).
//! - Tracked under: #1590 (HVZK-WHIR effort).

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_zk_codes::ZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::constraints::statement::EqStatement;
use crate::sumcheck::SumcheckData;
use crate::sumcheck::strategy::SumcheckProver;

/// Namespace for the HVZK variant of the WHIR sumcheck.
///
/// Mirrors [`super::single::SingleSumcheck`]: a unit struct hosting the static
/// constructors for each sumcheck strategy. In this PR only
/// [`Self::new_classic_unpacked`] is exposed; the SIMD-packed and SVO
/// variants land in follow-up PRs and will hang off the same namespace.
pub struct ZkSumcheck;

/// Stateful prover for the HVZK sumcheck (Construction 6.3).
///
/// Mirrors [`SumcheckProver`] (the plain prover state) and adds the mask
/// bookkeeping required by the construction. The plain prover handles the
/// witness-side polynomial fold exactly as in the non-ZK path; everything
/// mask-related lives here.
//
// Fields are populated by stubs that aren't implemented yet; the
// `dead_code` allow lifts in commit 2/3 of PR #1605 once the prelude and
// round logic actually read them.
#[allow(dead_code)]
pub struct ZkSumcheckProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
{
    /// Plain sumcheck state (poly + claimed sum). Folded at each `γ_j`
    /// exactly like the non-ZK path — fold logic is unchanged.
    /// In the reference paper:
    /// Construction 6.3, step 4: the Ĝ(X_1,…,X_k) polynomial
    base: SumcheckProver<F, EF>,
    /// ZK encoding used for the masks.
    /// in the reference paper:
    /// Theorem 6.2 ingredients: C_zk ⊆ Σ_zk^{m_zk}; Enc is C_zk
    encoding: Enc,
    /// `k` mask polynomials `s_1, …, s_k` as coefficient vectors of length
    /// `ell_zk`
    ///  in the reference paper:
    ///  Construction 6.3, step 1: "P samples s_1, …, s_k ∈ F^{<ℓ_zk}[X]"
    masks: Vec<Vec<F>>,
    /// MMCS commitment + prover data for each encoded mask codeword.
    ///
    /// - The `Commitment` is what gets observed on the challenger (binds
    ///   the masks to ε; see decision block item 3).
    /// - The `ProverData` is kept so downstream consumers (committed
    ///   sumcheck relation, §2.4 / §5 of the paper) can produce opening
    ///   proofs for queries to the mask oracles.
    ///
    /// Pattern matches [`p3_fri::HidingFriPcs`] — `Enc::Codeword` stays
    /// opaque on the encoding trait; binding goes through the MMCS layer.
    mask_oracles: Vec<(M::Commitment, M::ProverData<Enc::Codeword>)>,
    /// Combination challenge `ε` sampled after observing `μ̃`. Used in every
    /// subsequent round to scale the plain piece. In the paper:
    ///  ε ← F.
    ///  here challenges come from the extension field EF
    eps: EF,
    /// Running scalar tracking the future-mask endpoint sum.
    ///
    /// **Convention (Option B):** at the start of round `j`, *before* the
    /// per-round decrement, the field holds `Σ_{l ≥ j} (s_l(0) + s_l(1))`.
    /// Each `round()` decrements by `s_j(0) + s_j(1)` *at its start* —
    /// after the decrement the field holds `Σ_{l > j}`, which is what the
    /// per-round `ĥ_j` formula expects (see module-level docstring).
    ///
    /// At construction the field is initialised to `Σ_{l=1}^k (s_l(0)+s_l(1))`
    /// (= `Σ_{l ≥ 1}`, matching the convention at the entry to round 1).
    /// The same value is used in the closed-form `μ̃ = 2^{k-1} · this`.
    /// In the paper: Construction 6.3 step 2 + per-round formula step 4(a).
    sum_future_endpoints: F,
    /// `s_l(γ_l)` for `l < current_round`, accumulated as rounds progress.
    mask_evals_at_gamma: Vec<EF>,
    /// Number of rounds remaining (`k` initially, decremented per round).
    /// Standard bookkeeping
    rounds_left: usize,
}

impl ZkSumcheck {
    /// HVZK sumcheck via the classic unpacked (scalar) strategy.
    ///
    /// Mirrors [`super::single::SingleSumcheck::new_classic_unpacked`] in
    /// shape. Runs steps 1–3 of Construction 6.3 (sample masks, observe mask
    /// oracles, send `μ̃`, sample `ε`) plus the first masked sumcheck round,
    /// returning the prover state and the first verifier challenge `γ_1`.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` (Lemma 6.4 requires `char(F) ≠ 2`).
    /// - If `encoding.message_len() < 2` (Lemma 6.4 requires `ℓ_zk ≥ 2`).
    /// - All assertions inherited from
    ///   [`super::single::SingleSumcheck::new_classic_unpacked`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_classic_unpacked<F, EF, Enc, M, Challenger, R>(
        poly: &Poly<F>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
        encoding: &Enc,
        mmcs: &M,
        rng: &mut R,
    ) -> (ZkSumcheckProver<F, EF, Enc, M>, Point<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Enc: ZkEncoding<F>,
        Enc::Codeword: Matrix<F>,
        M: Mmcs<F>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let k = folding_factor;
        let ell_zk = encoding.message_len();

        // ----- Field constraints (decision block item 5) -----
        assert!(
            F::TWO != F::ZERO,
            "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
        );
        assert!(
            ell_zk >= 2,
            "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
        );
        assert!(k >= 1, "sumcheck requires at least one round");

        // ----- Step 1 — sample k masks `s_1, …, s_k ∈ F^{<ell_zk}[X]` -----
        // Each mask is a coefficient vector of length `ell_zk` over F.
        let masks: Vec<Vec<F>> = (0..k)
            .map(|_| (0..ell_zk).map(|_| rng.random()).collect())
            .collect();

        // ----- Step 1 (continued) — encode, MMCS-commit, observe -----
        // For each mask: encode under C_zk, MMCS-commit the codeword, observe
        // the commitment. The commitment is what binds the masks to the
        // ε challenge sampled later (decision block item 3 — pattern matches
        // `p3_fri::HidingFriPcs`). Encoding randomness is consumed inside
        // `Enc::encode` and not stored here; downstream composition that
        // needs `r'_j` will require bumping to `ZkEncodingWithRandomness`
        // and adding a `mask_randomness` field (see `_search_log.md`).
        let mask_oracles: Vec<(M::Commitment, M::ProverData<Enc::Codeword>)> = masks
            .iter()
            .map(|mask| {
                let codeword = encoding.encode(mask, rng);
                let (commit, prover_data) = mmcs.commit_matrix(codeword);
                challenger.observe(commit.clone());
                (commit, prover_data)
            })
            .collect();

        // ----- Step 2 — compute μ̃ via the closed form -----
        // For separable masks `ŝ(b) = Σ_l ŝ_l(b_l)`,
        //     μ̃ := Σ_{b ∈ {0,1}^k} ŝ(b) = 2^{k-1} · Σ_l (s_l(0) + s_l(1)).
        //
        // For a polynomial `s(X) = c_0 + c_1·X + … + c_{ell_zk-1}·X^{ell_zk-1}`
        // over F, `s(0) = c_0` and `s(1) = Σ c_i`, so
        //     `s(0) + s(1) = c_0 + Σ c_i = mask[0] + mask.iter().sum()`.
        let sum_future_endpoints: F = masks
            .iter()
            .map(|mask| mask[0] + mask.iter().copied().sum::<F>())
            .sum();
        let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        let mu_tilde: F = two_to_k_minus_1 * sum_future_endpoints;

        // Cross-check the closed form against the naive `Σ_{b ∈ {0,1}^k} ŝ(b)`.
        // Catches the multiplicity-bug class that the reference impl
        // (WizardOfMenlo/whir#241) shipped at first; see `_search_log.md`.
        // Bounded loop (only runs in debug) — k is at most a few dozen in
        // realistic settings.
        #[cfg(debug_assertions)]
        {
            let mut naive = F::ZERO;
            for bits in 0..(1u64 << k) {
                for (l, mask) in masks.iter().enumerate() {
                    let b_l = (bits >> l) & 1;
                    let s_l_eval = if b_l == 0 {
                        mask[0]
                    } else {
                        mask.iter().copied().sum::<F>()
                    };
                    naive += s_l_eval;
                }
            }
            debug_assert_eq!(
                mu_tilde, naive,
                "μ̃ closed form does not match naive Σ_{{b ∈ {{0,1}}^k}} ŝ(b)",
            );
        }

        // Observe μ̃ on the challenger. μ̃ lives in F (mask coefficients are
        // in F, b ∈ {0,1} so all `s_l(b)` stay in F); lift to EF for the
        // observation to match the rest of the codebase's algebra-element
        // transcript convention.
        challenger.observe_algebra_element(EF::from(mu_tilde));

        // ----- Step 3 — sample ε from the challenger -----
        let _eps: EF = challenger.sample_algebra_element();

        // Suppress unused-binding warnings for state we'll wire up in
        // commit 3; keeping them in scope so the call sites are visible.
        let _ = (
            poly,
            sumcheck_data,
            pow_bits,
            statement,
            masks,
            mask_oracles,
            sum_future_endpoints,
        );

        // TODO(commit 3 of PR #1605): Step 4, round 1.
        // - Build ĥ_1 from `base.sumcheck_coefficients(...)` (plain piece) +
        //   mask piece (live mask `s_1` weighted by `2^{k-1}`, plus
        //   future-mask endpoints weighted by `2^{k-2}`).
        // - Wire format: send (c0, c2, …, c_{ell_zk-1}); verifier derives c1
        //   from `ĥ_1(0) + ĥ_1(1) = ε·μ + μ̃`.
        // - Observe ĥ_1 coefficients (minus c1) on transcript.
        // - Grind PoW; sample γ_1.
        // - Fold base prover at γ_1; push s_1(γ_1) onto mask_evals_at_gamma;
        //   decrement sum_future_endpoints by `s_2(0) + s_2(1)` if k ≥ 2.
        // - Construct and return `(ZkSumcheckProver { … }, Point::new(vec![γ_1]))`.
        unimplemented!("commit 3 of PR #1605: round 1 + ZkSumcheckProver construction")
    }

    // TODO(follow-up commits): verifier counterpart (`verify_classic_unpacked`)
    // and HVZK simulator (`simulate_classic_unpacked`).
}

impl<F, EF, Enc, M> ZkSumcheckProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
{
    /// Runs one masked sumcheck round (rounds `2..=k`).
    ///
    /// Computes `ĥ_j` per the per-round formula in this module's docstring,
    /// sends the `ell_zk - 1` non-linear coefficients on the transcript,
    /// samples `γ_j`, folds the base prover, and updates the running mask
    /// bookkeeping.
    pub fn round<Challenger>(
        &mut self,
        _sumcheck_data: &mut SumcheckData<F, EF>,
        _challenger: &mut Challenger,
        _sum: &mut EF,
        _pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // TODO(commit 3): per-round arithmetic + bookkeeping update.
        unimplemented!("commit 3 of PR #1605: round() impl")
    }

    /// Read-only access to the encoded mask oracles, for downstream
    /// protocols (committed sumcheck relation; see §2.4 / §5 of eprint
    /// 2026/391).
    ///
    /// Returns `(MMCS commitment, prover data)` per mask. Downstream
    /// callers can produce opening proofs by passing the prover data back
    /// into the same MMCS instance.
    pub fn mask_oracles(&self) -> &[(M::Commitment, M::ProverData<Enc::Codeword>)] {
        &self.mask_oracles
    }
}
