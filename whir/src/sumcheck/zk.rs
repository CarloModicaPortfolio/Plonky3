//! HVZK variant of the WHIR sumcheck (Construction 6.3, eprint 2026/391).
//!
//! Companion to [`super::single`]: same sumcheck reduction, but with `k` random
//! univariate masks committed under a ZK encoding so that the prover's `k`
//! round-polynomials no longer leak linear functions of the secret message.
//!
//! # Protocol overview
//!
//! 1. **Masks.** Prover samples `s_1, …, s_k ∈ F^{<ℓ_zk}[X]` and commits
//!    each encoded codeword `Enc_{C_zk}(s_j)` under MMCS, observing each
//!    commitment on the transcript.
//! 2. **New target.** Prover sends `μ̃ := Σ_{b ∈ {0,1}^k} (s_1(b_1) + … + s_k(b_k))`.
//! 3. **Combination randomness.** Verifier samples `ε`.
//! 4. **Sumcheck.** For `j = 1, …, k`: prover sends `ĥ_j` (formula below),
//!    verifier samples `γ_j`.
//!
//! Decision-phase verifier checks:
//!
//! - Round 1: `ĥ_1(0) + ĥ_1(1) = ε·μ + μ̃`.
//! - Round `j > 1`: `ĥ_j(0) + ĥ_j(1) = ĥ_{j-1}(γ_{j-1})`.
//!
//! # Per-round polynomial
//!
//! For round `j` with `γ = (γ_1, …, γ_{j-1})` already sampled:
//!
//! ```text
//! ĥ_j(X) = 2^{k-j}   * s_j(X)                          (live mask)
//!        + 2^{k-j}   * Σ_{l < j} s_l(γ_l)              (past masks, cached)
//!        + 2^{k-j-1} * Σ_{l > j} (s_l(0) + s_l(1))     (future-mask endpoints)
//!        + ε         * plain_piece(X)                  (base sumcheck round)
//! ```
//!
//! Combined degree is `max(ℓ_zk - 1, 2)`: the mask piece has degree `ℓ_zk - 1`,
//! the plain piece is degree 2 (multilinear × multilinear).
//!
//! # `μ̃` closed form
//!
//! For separable masks `ŝ(b) := s_1(b_1) + … + s_k(b_k)`, each `b_l ∈ {0,1}`
//! is independent and contributes `s_l(0) + s_l(1)` over `2^{k-1}` of the `2^k`
//! strings, giving
//!
//! ```text
//! μ̃ = Σ_{b ∈ {0,1}^k} ŝ(b) = 2^{k-1} * Σ_l (s_l(0) + s_l(1)).
//! ```
//!
//! For `s(X) = c_0 + c_1·X + … + c_{ℓ_zk-1}·X^{ℓ_zk-1}` we have
//! `s(0) + s(1) = c_0 + Σ c_i = mask[0] + mask.iter().sum()`.
//!
//! # Wire format (skip-linear-coefficient)
//!
//! Per round the prover sends `max(ℓ_zk - 1, 2)` field elements
//! `(c_0, c_2, c_3, …, c_d)` where `d = max(ℓ_zk - 1, 2)`. The linear
//! coefficient `c_1` is omitted: the verifier reconstructs it from the affine
//! check above using `ĥ_j(0) + ĥ_j(1) = 2·c_0 + Σ_{i ≥ 1} c_i`.
//!
//! Lemma 6.4's rank-nullity argument shows the affine subspace of valid
//! transcripts `(μ̃, ĥ_1, …, ĥ_k)` has dimension `1 + k(ℓ_zk - 1)`, so the `k`
//! linear coefficients are exactly the redundant degrees of freedom.
//!
//! Mirrors the same convention used by [`super::single`] for the plain
//! (degree-2) round polynomial.
//!
//! # Field constraints (Lemma 6.4)
//!
//! - `char(F) ≠ 2` — required by the rank-nullity argument that drives the
//!   HVZK simulator's affine-subspace surjectivity.
//! - `ℓ_zk ≥ 2` — needed for the mask piece to carry non-trivial information.
//!
//! Both are enforced at constructor entry.
//!
//! # References
//!
//! - eprint 2026/391, §6 Construction 6.3, Lemma 6.4 (HVZK), Lemma 6.5 (RBR).

use alloc::vec;
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
use crate::sumcheck::extrapolate_01inf;
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};

/// Per-round transcript records for the HVZK sumcheck.
///
/// HVZK's per-round polynomial has combined degree `max(ℓ_zk - 1, 2)`, so each
/// round's wire payload carries `max(ℓ_zk - 1, 2)` field elements (after
/// skipping `c_1`). The length is constant within a proof but only known at
/// runtime (derived from `encoding.message_len()`), hence the inner `Vec<EF>`.
#[derive(Default, Debug, Clone)]
pub struct ZkSumcheckData<F, EF> {
    /// Per-round wire coefficients of `ĥ_j` with the linear term skipped.
    /// Layout per entry: `[c_0, c_2, c_3, …, c_d]` where `d = max(ℓ_zk - 1, 2)`.
    pub round_coefficients: Vec<Vec<EF>>,
    /// Per-round proof-of-work witnesses (one entry per round if `pow_bits > 0`).
    pub pow_witnesses: Vec<F>,
}

/// Namespace for the HVZK variant of the WHIR sumcheck.
///
/// Mirrors [`super::single::SingleSumcheck`]: a unit struct hosting the static
/// constructors for each sumcheck strategy.
pub struct ZkSumcheck;

/// Stateful prover for the HVZK sumcheck (Construction 6.3).
///
/// Wraps a plain [`SumcheckProver`] (which handles the witness-side polynomial
/// fold exactly as in the non-ZK path) and adds the mask bookkeeping required
/// to build `ĥ_j` per the per-round formula in the module docs.
#[allow(dead_code)]
pub struct ZkSumcheckProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
{
    /// Plain sumcheck state (poly + claimed sum). Tracks the plain piece only;
    /// folded at each `γ_j` exactly like the non-ZK path. Corresponds to the
    /// `Ĝ(X_1, …, X_k)` polynomial of Construction 6.3 step 4.
    base: SumcheckProver<F, EF>,
    /// ZK encoding `Enc_{C_zk}` used for the masks (Theorem 6.2 ingredient `C_zk`).
    encoding: Enc,
    /// The `k` mask polynomials `s_1, …, s_k ∈ F^{<ℓ_zk}[X]` as coefficient
    /// vectors of length `ℓ_zk` (Construction 6.3 step 1).
    masks: Vec<Vec<F>>,
    /// MMCS commitment + prover data for each encoded mask codeword.
    ///
    /// - The commitment is observed on the challenger, binding the masks to
    ///   the `ε` challenge sampled later.
    /// - The prover data is kept so downstream consumers (committed sumcheck
    ///   relation, §2.4 / §5 of the paper) can produce opening proofs for
    ///   queries to the mask oracles.
    mask_oracles: Vec<(M::Commitment, M::ProverData<Enc::Codeword>)>,
    /// Combination challenge `ε` sampled after `μ̃`; multiplies the plain
    /// piece in every round polynomial. Widened to the extension field for
    /// soundness margin (the paper writes `ε ← F`).
    eps: EF,
    /// Running future-mask endpoint sum.
    ///
    /// At the start of round `j`, before that round's start-of-round
    /// decrement, this field holds
    ///
    /// ```text
    /// Σ_{l ≥ j} (s_l(0) + s_l(1)).
    /// ```
    ///
    /// Each round subtracts `s_j(0) + s_j(1)` first, leaving `Σ_{l > j}`,
    /// which is the future-mask term in `ĥ_j`'s formula. The same quantity at
    /// `j = 1` drives the closed-form `μ̃ = 2^{k-1} · Σ_{l=1}^k (s_l(0) + s_l(1))`.
    sum_future_endpoints: F,
    /// `s_l(γ_l)` for `l < current_round`, accumulated as rounds progress.
    /// Drives the past-mask term `Σ_{l < j} s_l(γ_l)` of `ĥ_j`.
    mask_evals_at_gamma: Vec<EF>,
    /// Number of rounds remaining; decremented per `round()` call.
    rounds_left: usize,
}

impl ZkSumcheck {
    /// HVZK sumcheck via the classic unpacked (scalar) strategy.
    ///
    /// Mirrors [`super::single::SingleSumcheck::new_classic_unpacked`] in
    /// shape. Runs Construction 6.3 steps 1–3 (sample and commit masks, send
    /// `μ̃`, sample `ε`) plus round 1 of step 4 (build `ĥ_1`, sample `γ_1`,
    /// fold the base polynomial).
    ///
    /// # Algorithm
    ///
    /// 1. Sample a batching challenge `α` and combine multiple `EqStatement`
    ///    constraints into a single weight polynomial.
    /// 2. Sample masks `s_1, …, s_k ∈ F^{<ℓ_zk}[X]`; for each, encode under
    ///    `Enc_{C_zk}`, MMCS-commit the codeword, and observe the commitment.
    /// 3. Compute and observe `μ̃ = 2^{k-1} · Σ_l (s_l(0) + s_l(1))`.
    /// 4. Sample `ε`.
    /// 5. Build `ĥ_1` per the per-round formula; observe its non-linear
    ///    coefficients on the transcript; grind; sample `γ_1`.
    /// 6. Cache `s_1(γ_1)` and fold the base polynomial at `γ_1`.
    ///
    /// # Returns
    ///
    /// - The HVZK prover state, ready for rounds 2..=k via [`ZkSumcheckProver::round`].
    /// - The first verifier challenge `γ_1`.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` (Lemma 6.4 requires `char(F) ≠ 2`).
    /// - If `encoding.message_len() < 2` (Lemma 6.4 requires `ℓ_zk ≥ 2`).
    /// - If `folding_factor` is 0 or exceeds `poly.num_variables()`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_classic_unpacked<F, EF, Enc, M, Challenger, R>(
        poly: &Poly<F>,
        zk_data: &mut ZkSumcheckData<F, EF>,
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
        Enc: ZkEncoding<F> + Clone,
        Enc::Codeword: Matrix<F>,
        M: Mmcs<F>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let k = folding_factor;
        let ell_zk = encoding.message_len();
        let n_vars = poly.num_variables();

        assert!(
            F::TWO != F::ZERO,
            "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
        );
        assert!(
            ell_zk >= 2,
            "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
        );
        assert!(k >= 1, "sumcheck requires at least one round");
        assert!(
            k <= n_vars,
            "folding_factor must be <= poly.num_variables()",
        );

        // Sample a batching challenge for combining multiple equality
        // constraints into a single weight polynomial. Construction 6.3
        // assumes a single-claim input (relation `R_{C, C_zk, sl}`,
        // Definition 5.8); we collapse here before proceeding.
        let alpha: EF = challenger.sample_algebra_element();
        let mut weights = Poly::zero(n_vars);
        let mut sum = EF::ZERO;
        statement.combine_hypercube::<F, false>(&mut weights, &mut sum, alpha);

        // --- Construction 6.3 step 1: sample, encode, commit, observe ---
        // Sample masks `s_1, …, s_k ∈ F^{<ell_zk}[X]` as coefficient vectors;
        // for each, encode the codeword under `Enc_{C_zk}`, MMCS-commit, and
        // observe the commitment. The commitment is what binds the masks to
        // the `ε` challenge sampled later. Encoding randomness is consumed
        // inside `Enc::encode` and not stored.
        let masks: Vec<Vec<F>> = (0..k)
            .map(|_| (0..ell_zk).map(|_| rng.random()).collect())
            .collect();
        let mask_oracles: Vec<(M::Commitment, M::ProverData<Enc::Codeword>)> = masks
            .iter()
            .map(|mask| {
                let codeword = encoding.encode(mask, rng);
                let (commit, prover_data) = mmcs.commit_matrix(codeword);
                challenger.observe(commit.clone());
                (commit, prover_data)
            })
            .collect();

        // --- Construction 6.3 step 2: send μ̃ ---
        // μ̃ = Σ_{b ∈ {0,1}^k} ŝ(b) = 2^{k-1} · Σ_l (s_l(0) + s_l(1))
        // where s_l(0) + s_l(1) = c_0 + Σ c_i = mask[0] + Σ mask
        // for s_l(X) = c_0 + c_1·X + … + c_{ell_zk-1}·X^{ell_zk-1}.
        let sum_future_endpoints: F = masks
            .iter()
            .map(|mask| mask[0] + mask.iter().copied().sum::<F>())
            .sum();
        let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        let mu_tilde: F = two_to_k_minus_1 * sum_future_endpoints;

        // Cross-check the closed form against the naive 2^k-term sum.
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

        // Lift `μ̃` to EF for transcript observation (codebase convention).
        challenger.observe_algebra_element(EF::from(mu_tilde));

        // --- Construction 6.3 step 3: sample ε ---
        let eps: EF = challenger.sample_algebra_element();

        // --- Construction 6.3 step 4, round 1: build ĥ_1, fold base ---

        // Start-of-round decrement: subtract `s_1`'s endpoints so the running
        // future-mask sum holds `Σ_{l > 1} (s_l(0) + s_l(1))`, the value the
        // per-round formula uses at `j = 1`.
        let s_1_endpoints = masks[0][0] + masks[0].iter().copied().sum::<F>();
        let sum_future_endpoints_state = sum_future_endpoints - s_1_endpoints;

        // Plain piece (degree-2): returns (c_0, c_∞); derive c_1 from the
        // affine constraint h(0) + h(1) = sum, i.e. c_1 = sum - 2·c_0 - c_∞.
        let (plain_c0, plain_c_inf) =
            VariableOrder::Prefix.sumcheck_coefficients(poly.as_slice(), weights.as_slice());
        let plain_c1 = sum - plain_c0.double() - plain_c_inf;

        // Build `ĥ_1` of length `max(ell_zk, 3)`:
        //   indices 0..ell_zk : live-mask piece           = 2^{k-1} · s_1(X)
        //   index 0           : future-mask contribution += 2^{k-2} · Σ_{l>1}(s_l(0)+s_l(1))
        //   indices 0..3      : plain piece              += ε · (c_0 + c_1·X + c_∞·X²)
        // The future-mask term is only present when `k ≥ 2`.
        let h1_size = core::cmp::max(ell_zk, 3);
        let mut h1: Vec<EF> = vec![EF::ZERO; h1_size];

        let two_pow_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        for (i, &c) in masks[0].iter().enumerate() {
            h1[i] += EF::from(two_pow_k_minus_1 * c);
        }
        if k >= 2 {
            let two_pow_k_minus_2 = F::TWO.exp_u64((k - 2) as u64);
            h1[0] += EF::from(two_pow_k_minus_2 * sum_future_endpoints_state);
        }

        h1[0] += eps * plain_c0;
        h1[1] += eps * plain_c1;
        h1[2] += eps * plain_c_inf;

        // Round-1 affine consistency check:
        //   h(0) + h(1) = c_0 + (c_0 + c_1 + … + c_d) = 2·c_0 + Σ_{i ≥ 1} c_i,
        // which must equal μ̃ + ε·μ.
        debug_assert_eq!(
            h1[0].double() + h1[1..].iter().copied().sum::<EF>(),
            EF::from(mu_tilde) + eps * sum,
            "ĥ_1 should satisfy h(0) + h(1) = μ̃ + ε·μ",
        );

        // Wire format: send (c_0, c_2, c_3, …, c_d), skipping c_1; verifier
        // reconstructs c_1 from the affine consistency check above.
        let mut h1_wire: Vec<EF> = Vec::with_capacity(h1_size - 1);
        h1_wire.push(h1[0]);
        for i in 2..h1_size {
            h1_wire.push(h1[i]);
        }

        challenger.observe_algebra_slice(&h1_wire);
        zk_data.round_coefficients.push(h1_wire);

        // Proof-of-work grind, then sample γ_1.
        if pow_bits > 0 {
            zk_data.pow_witnesses.push(challenger.grind(pow_bits));
        }
        let gamma_1: EF = challenger.sample_algebra_element();

        // Cache `s_1(γ_1)` via Horner for the past-mask term in future rounds.
        let s1_at_gamma1: EF = masks[0]
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |acc, c| acc * gamma_1 + EF::from(c));
        let mask_evals_at_gamma: Vec<EF> = vec![s1_at_gamma1];

        // Fold base polynomial and weights at γ_1; update plain sum to
        // plain_h(γ_1) via quadratic extrapolation. `base.sum` tracks the
        // plain-piece sum only — mask-side bookkeeping lives in this struct's
        // other fields, multiplied by `ε` when assembled into `ĥ_j`.
        weights.fix_prefix_var_mut(gamma_1);
        let folded_poly = poly.fix_prefix_var(gamma_1);
        let new_sum = extrapolate_01inf(plain_c0, sum - plain_c0, plain_c_inf, gamma_1);

        let product_poly =
            ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, folded_poly, weights);
        debug_assert_eq!(product_poly.dot_product(), new_sum);
        let base = SumcheckProver::new(product_poly, new_sum);

        let prover = ZkSumcheckProver {
            base,
            encoding: encoding.clone(),
            masks,
            mask_oracles,
            eps,
            // After round 1's start-decrement, this holds Σ_{l ≥ 2}, which is
            // the state round 2 expects at its start (before its own decrement).
            sum_future_endpoints: sum_future_endpoints_state,
            mask_evals_at_gamma,
            rounds_left: k - 1,
        };

        (prover, Point::new(vec![gamma_1]))
    }
}

impl<F, EF, Enc, M> ZkSumcheckProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
{
    /// Runs one masked sumcheck round for `j ∈ 2..=k`.
    ///
    /// Computes `ĥ_j` per the per-round formula in the module docs, observes
    /// its `max(ell_zk - 1, 2)` non-linear coefficients on the transcript,
    /// grinds, samples `γ_j`, folds the base prover, and updates the running
    /// mask bookkeeping. Returns `γ_j`.
    pub fn round<Challenger>(
        &mut self,
        _zk_data: &mut ZkSumcheckData<F, EF>,
        _challenger: &mut Challenger,
        _pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        unimplemented!("HVZK sumcheck round (j > 1) not yet implemented")
    }

    /// Read-only access to the encoded mask oracles for downstream protocols
    /// (committed sumcheck relation; §2.4 / §5 of eprint 2026/391).
    ///
    /// Returns `(MMCS commitment, prover data)` per mask. Callers produce
    /// opening proofs by passing the prover data back into the same MMCS
    /// instance.
    pub fn mask_oracles(&self) -> &[(M::Commitment, M::ProverData<Enc::Codeword>)] {
        &self.mask_oracles
    }
}
