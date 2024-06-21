//! The scalar field of the Vesta curve, defined as `F_p` where
//! `p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001`
//! is the base field of the Pallas curve, which is equivalent to the scalar field of the vesta
//! curve
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use std::fmt::{Debug, Display, Formatter};
use ff::{Field as FField, PrimeField as PrimeFField};
use num_bigint::BigUint;
use num_traits::Num;
use p3_field::{AbstractField, Field, Packable, PrimeField};
use pasta_curves::Fp;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct FpFF {
    pub value: Fp,
}

impl FpFF {
    const fn new(value: Fp) -> Self {
        Self { value }
    }

}

/// Serializes bytes to human readable or compact representation.
///
/// Depending on whether the serializer is a human readable one or not, the bytes are either
/// encoded as a hex string or a list of bytes.
fn serialize_bytes<S: Serializer>(bytes: [u8; 32], s: S) -> Result<S::Ok, S::Error> {
    if s.is_human_readable() {
        hex::serde::serialize(bytes, s)
    } else {
        bytes.serialize(s)
    }
}

/// Deserialize bytes from human readable or compact representation.
///
/// Depending on whether the deserializer is a human readable one or not, the bytes are either
/// decoded from a hex string or a list of bytes.
fn deserialize_bytes<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
    if d.is_human_readable() {
        hex::serde::deserialize(d)
    } else {
        <[u8; 32]>::deserialize(d)
    }
}

impl Serialize for FpFF {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serialize_bytes(self.value.to_repr(), s)
    }
}

impl<'de> Deserialize<'de> for FpFF {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes = deserialize_bytes(d)?;
        match Fp::from_repr(bytes).into() {
            Some(fp) => Ok(FpFF::new(fp)),
            None => Err(D::Error::custom(
                "deserialized bytes don't encode a Pallas field element",
            )),
        }
    }
}

impl Packable for FpFF {}

impl Hash for FpFF {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.value.to_repr().as_ref().iter() {
            state.write_u8(*byte);
        }
    }
}

impl Ord for FpFF {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering { self.value.cmp(&other.value) }
}

impl PartialOrd for FpFF {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for FpFF {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <Fp as Debug>::fmt(&self.value, f)
        }
}

impl Debug for FpFF {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FpFF({:?})", self.value)
    }
}

impl AbstractField for FpFF {
    type F = Self;

    fn zero() -> Self { Self::new(Fp::zero()) }
    fn one() -> Self {  Self::new(Fp::one())}
    fn two() -> Self {
        Self::new(Fp::from(2u64))
    }

    fn neg_one() -> Self {
        Self::new(Fp::zero()-Fp::one())
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self::new(Fp::from(b as u64))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new(Fp::from(n as u64))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new(Fp::from(n as u64))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::new(Fp::from(n as u64))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::new(Fp::from(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::new(Fp::from(n as u64))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(Fp::from(n as u64))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::new(Fp::from(n))
    }

    fn generator() -> Self {
        Self::new(Fp::MULTIPLICATIVE_GENERATOR)
    }
}

impl Field for FpFF {

    type Packing = Self;
    fn is_zero(&self) -> bool {
        self.value.is_zero().into()

    }
    fn try_inverse(&self) -> Option<Self> {
        let inverse = self.value.invert();

        if inverse.is_some().into() {
            Some(Self::new(inverse.unwrap()))
        } else {
            None
        }
    }

    fn order() -> BigUint {
        BigUint::from_str_radix(Fp::MODULUS.trim_start_matches("0x"), 16)
            .expect("Failed to parse MODULUS")
    }
    fn multiplicative_group_factors() -> Vec<(BigUint, usize)> {
        vec![(BigUint::new(vec![0x992d30edu32, 0x094cf91bu32, 0x224698fcu32, 0x00000000u32, 0x00000000u32, 0x00000000u32, 0x40000000u32, 0x00000000u32]), 1),
            (BigUint::from(2u8), 32),
        ]
    }
}

impl PrimeField for FpFF {
    fn as_canonical_biguint(&self) -> BigUint {
        let repr = self.value.to_repr();
        let le_bytes = repr.as_ref();
        BigUint::from_bytes_le(le_bytes)
    }
}

impl Add for FpFF {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }

}

impl AddAssign for FpFF {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sum for FpFF {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::zero())
    }
}

impl Sub for FpFF {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.value.sub(rhs.value))
    }
}

impl SubAssign for FpFF {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Neg for FpFF {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::neg_one()
    }
}

impl Mul for FpFF {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}

impl MulAssign for FpFF {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl Product for FpFF {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y | x * y).unwrap_or(Self::one())
    }
}

impl Div for FpFF {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl Distribution<FpFF> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> FpFF {
        FpFF::new(Fp::random(rng))
    }
}
#[cfg(test)]
mod tests {
    use num_traits::One;
    use p3_field_testing::test_field;
    use super::*;

    type F = FpFF;

    #[test]
    fn test_fp_ff() {

        // rapresentation as canonical big unit tesst
        let f = F::new(Fp::from_u128(100));
        assert_eq!(f.as_canonical_biguint(), BigUint::new(vec![100]));
        println!();

        // is_zero test
        let f = F::from_canonical_u64(0);
        assert!(f.is_zero());

        // The order is zero
        let f = F::new(Fp::from_str_vartime(&F::order().to_str_radix(10)).unwrap());
        assert!(f.is_zero());

        // The generator is 5
        assert_eq!(F::generator().as_canonical_biguint(), BigUint::new(vec![5]));

        let f_1 = F::new(Fp::from_u128(1));
        let f_1_copy = F::new(Fp::from_u128(1));

        let expected_result = F::zero();
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::new(Fp::from_u128(2));
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::new(Fp::from_u128(2));
        let expected_result = F::new(Fp::from_u128(3));
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::new(Fp::from_u128(5));
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_r_minus_1 = F::new(
            Fp::from_str_vartime(&(F::order() - BigUint::one()).to_str_radix(10)).unwrap(),
        );
        let expected_result = F::zero();
        assert_eq!(f_1 + f_r_minus_1, expected_result);

        let f_r_minus_2 = F::new(
            Fp::from_str_vartime(&(F::order() - BigUint::new(vec![2])).to_str_radix(10))
                .unwrap(),
        );
        let expected_result = F::new(
            Fp::from_str_vartime(&(F::order() - BigUint::new(vec![3])).to_str_radix(10))
                .unwrap(),
        );
        assert_eq!(f_r_minus_1 + f_r_minus_2, expected_result);

        let expected_result = F::new(Fp::from_u128(1));
        assert_eq!(f_r_minus_1 - f_r_minus_2, expected_result);

        let expected_result = f_r_minus_1;
        assert_eq!(f_r_minus_2 - f_r_minus_1, expected_result);

        let expected_result = f_r_minus_2;
        assert_eq!(f_r_minus_1 - f_1, expected_result);

        let expected_result = F::new(Fp::from_u128(3));
        assert_eq!(f_2 * f_2 - f_1, expected_result);

        // Generator check
        let expected_multiplicative_group_generator = F::new(Fp::from_u128(5));
        assert_eq!(F::generator(), expected_multiplicative_group_generator);

        let f_serialized = serde_json::to_string(&f).unwrap();
        let f_deserialized: F = serde_json::from_str(&f_serialized).unwrap();
        assert_eq!(f, f_deserialized);

        let f_1_serialized = serde_json::to_string(&f_1).unwrap();
        let f_1_deserialized: F = serde_json::from_str(&f_1_serialized).unwrap();
        let f_1_serialized_again = serde_json::to_string(&f_1_deserialized).unwrap();
        let f_1_deserialized_again: F = serde_json::from_str(&f_1_serialized_again).unwrap();
        assert_eq!(f_1, f_1_deserialized);
        assert_eq!(f_1, f_1_deserialized_again);

        let f_2_serialized = serde_json::to_string(&f_2).unwrap();
        let f_2_deserialized: F = serde_json::from_str(&f_2_serialized).unwrap();
        assert_eq!(f_2, f_2_deserialized);

        let f_r_minus_1_serialized = serde_json::to_string(&f_r_minus_1).unwrap();
        let f_r_minus_1_deserialized: F = serde_json::from_str(&f_r_minus_1_serialized).unwrap();
        assert_eq!(f_r_minus_1, f_r_minus_1_deserialized);

        let f_r_minus_2_serialized = serde_json::to_string(&f_r_minus_2).unwrap();
        let f_r_minus_2_deserialized: F = serde_json::from_str(&f_r_minus_2_serialized).unwrap();
        assert_eq!(f_r_minus_2, f_r_minus_2_deserialized);

    }

    test_field!(crate::fp::FpFF);
}