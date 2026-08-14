//! Exact port of R's Mersenne-Twister RNG + sample() (R 4.5.x semantics).
//!
//! Mirrors src/python_cibersort/rrnd.py (which is verified bit-exact
//! against R 4.5.3). See that file for provenance notes.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;
const TEMPERING_MASK_B: u32 = 0x9d2c5680;
const TEMPERING_MASK_C: u32 = 0xefc60000;
const MT_SCALE: f64 = 2.3283064365386963e-10; // 1/2^32
const I2_32M1: f64 = 2.328306437080797e-10; // 1/(2^32-1)

#[derive(Clone)]
pub struct RRng {
    mt: [u32; N],
    index: usize,
}

#[inline]
fn lcg_next(s: u32) -> u32 {
    s.wrapping_mul(69069).wrapping_add(1)
}

impl RRng {
    /// do_setseed -> RNG_Init(MERSENNE_TWISTER, seed)
    pub fn new(seed: i64) -> Self {
        let mut s = (seed as u64 & 0xffff_ffff) as u32;
        for _ in 0..50 {
            s = lcg_next(s);
        }
        let mut mt = [0u32; N];
        // 1+624 LCG fills; i_seed[0] is the mti slot (overwritten by
        // FixupSeeds with 624), so consume one LCG step first, then
        // mt[0..623] = i_seed[1..624] = the next 624 LCG outputs.
        let mut last = lcg_next(s);
        for slot in mt.iter_mut() {
            last = lcg_next(last);
            *slot = last;
        }
        let mut r = RRng { mt, index: N };
        // FixupSeeds(initial=1): I1 = 624 -> force twist on first draw
        r.index = N;
        r
    }

    fn twist(&mut self) {
        let mt = &mut self.mt;
        let orig = *mt;
        // kk = 0..N-M-1
        for kk in 0..(N - M) {
            let y = (orig[kk] & UPPER_MASK) | (orig[kk + 1] & LOWER_MASK);
            mt[kk] = orig[kk + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
        }
        // kk = N-M..2(N-M)-1: mt[kk-(N-M)] updated by block above
        for kk in (N - M)..(2 * (N - M)) {
            let y = (orig[kk] & UPPER_MASK) | (orig[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk - (N - M)] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
        }
        // kk = 2(N-M)..N-2: mt[kk-(N-M)] updated by previous slice
        for kk in (2 * (N - M))..(N - 1) {
            let y = (orig[kk] & UPPER_MASK) | (orig[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk - (N - M)] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
        }
        // kk = N-1: mixes original mt[N-1] with updated mt[0]
        let y = (orig[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
        self.index = 0;
    }

    #[inline]
    fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & TEMPERING_MASK_B;
        y ^= (y << 15) & TEMPERING_MASK_C;
        y ^= y >> 18;
        y
    }

    /// unif_rand(): fixup(genrand * 2^-32)
    #[inline]
    pub fn unif_rand(&mut self) -> f64 {
        let x = self.genrand_u32() as f64 * MT_SCALE;
        if x <= 0.0 {
            return 0.5 * I2_32M1;
        }
        if 1.0 - x <= 0.0 {
            return 1.0 - 0.5 * I2_32M1;
        }
        x
    }

    #[inline]
    fn rbits(&mut self, bits: u32) -> u64 {
        let mut v: u64 = 0;
        let mut n = 0;
        while n <= bits {
            let v1 = (self.unif_rand() * 65536.0).floor() as u64;
            v = 65536 * v + v1;
            n += 16;
        }
        v & ((1u64 << bits) - 1)
    }

    /// R_unif_index with sample.kind = "Rejection"
    #[inline]
    pub fn r_unif_index(&mut self, dn: u64) -> u64 {
        if dn == 0 {
            return 0;
        }
        let bits = (64 - (dn - 1).leading_zeros()) as u32; // ceil(log2(dn))
        let mut dv = self.rbits(bits);
        while dn <= dv {
            dv = self.rbits(bits);
        }
        dv
    }

    /// sample(n, k), uniform, without replacement (do_sample int branch).
    /// Returns 1-based indices like R.
    pub fn sample_no_replace(&mut self, n: u64, k: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(k);
        if k < 2 {
            for _ in 0..k {
                out.push(self.r_unif_index(n) + 1);
            }
        } else {
            let mut x: Vec<u64> = (0..n).collect();
            let mut nn = n;
            for _ in 0..k {
                let j = self.r_unif_index(nn) as usize;
                out.push(x[j] + 1);
                x[j] = x[(nn - 1) as usize];
                nn -= 1;
            }
        }
        out
    }
}
