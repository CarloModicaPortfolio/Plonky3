//! Delayed-reduction variant of `compress_hi_dot` for the Packed case.
//!
//! For each output row, decomposes the inner N-element `PackedEF × F::Packing` dot
//! product into `dim` separate `F::Packing × F::Packing` dot products (one per
//! extension-field basis coefficient). Each per-coefficient dot product uses
//! Plonky3's specialized `PackedMontyField31::dot_product<N>` which accumulates
//! u64 products via `dot_product_4` (VPMULUDQ + VPADDQ) and applies a single
//! Monty reduction per 4-chunk — giving a ~4× reduction in the number of
//! Montgomery reductions relative to the eager path.

use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};

use crate::poly::Poly;

/// Delayed-reduction `compress_hi_dot` for the Packed variant.
///
/// Specialized for `inner_size == 32` (the common 22-var SVO shape with a 4-var
/// low-half eq on AVX-512: `2^(split_eq - k_pack) = 2^5 = 32`). Falls back to a
/// scalar-accumulator default for other sizes. Adding further specializations is
/// possible but risks code-bloat regressions in unrelated hot paths.
pub(super) fn compress_hi_dot_delayed_packed<F, EF>(
    eq1_packed: &[<EF as ExtensionField<F>>::ExtensionPacking],
    chunk: &[F],
    eq0: &Poly<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let inner_size = eq1_packed.len();
    if inner_size == 0 {
        return EF::ZERO;
    }
    if <EF as BasedVectorSpace<F>>::DIMENSION != 4 {
        return fallback::<F, EF>(eq1_packed, chunk, eq0);
    }

    match inner_size {
        32 => specialized::<F, EF, 32>(eq1_packed, chunk, eq0),
        _ => fallback::<F, EF>(eq1_packed, chunk, eq0),
    }
}

#[inline]
fn specialized<F, EF, const N: usize>(
    eq1_packed: &[<EF as ExtensionField<F>>::ExtensionPacking],
    chunk: &[F],
    eq0: &Poly<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(eq1_packed.len(), N);

    let chunk_packed = F::Packing::pack_slice(chunk);

    // Transpose eq1 into four stack arrays, one per extension-field basis coefficient.
    // Stack-allocated to avoid polluting the allocator's per-thread page cache, which
    // heap allocation here was observed to do (with 2× regressions in unrelated benches).
    let mut c0: [F::Packing; N] = [F::Packing::default(); N];
    let mut c1: [F::Packing; N] = [F::Packing::default(); N];
    let mut c2: [F::Packing; N] = [F::Packing::default(); N];
    let mut c3: [F::Packing; N] = [F::Packing::default(); N];
    for (i, item) in eq1_packed.iter().enumerate() {
        let coefs = item.as_basis_coefficients_slice();
        c0[i] = coefs[0];
        c1[i] = coefs[1];
        c2[i] = coefs[2];
        c3[i] = coefs[3];
    }

    let mut acc = EF::ExtensionPacking::default();
    for (j, w0) in eq0.as_slice().iter().enumerate() {
        let start = j * N;
        let chunk_piece: &[F::Packing; N] =
            chunk_packed[start..start + N].try_into().unwrap();

        let inner_j = EF::ExtensionPacking::from_basis_coefficients_fn(|k| match k {
            0 => F::Packing::dot_product::<N>(&c0, chunk_piece),
            1 => F::Packing::dot_product::<N>(&c1, chunk_piece),
            2 => F::Packing::dot_product::<N>(&c2, chunk_piece),
            3 => F::Packing::dot_product::<N>(&c3, chunk_piece),
            _ => unreachable!(),
        });

        acc += inner_j * (*w0);
    }

    <EF::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter([acc]).sum()
}

#[inline]
fn fallback<F, EF>(
    eq1_packed: &[<EF as ExtensionField<F>>::ExtensionPacking],
    chunk: &[F],
    eq0: &Poly<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    use p3_field::dot_product;
    let chunk_packed = F::Packing::pack_slice(chunk);
    let inner_size = eq1_packed.len();

    let sum: EF::ExtensionPacking = (0..eq0.as_slice().len())
        .map(|j| {
            let chunk_piece = &chunk_packed[j * inner_size..(j + 1) * inner_size];
            let d: EF::ExtensionPacking = dot_product::<EF::ExtensionPacking, _, _>(
                eq1_packed.iter().copied(),
                chunk_piece.iter().copied(),
            );
            d * eq0.as_slice()[j]
        })
        .sum();
    <EF::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter([sum]).sum()
}
