"""Bit-exact reimplementation of R's random-number machinery.

Replicates, from R 4.5.x sources (src/main/RNG.c, src/main/random.c):

* ``set.seed(seed)`` with the default ``RNGkind()`` =
  ``("Mersenne-Twister", "Inversion", "Rejection")``:
  - ``RNG_Init``: 50x scrambling ``seed = 69069*seed + 1`` (uint32 wrap),
    then 625 LCG fills of the seed vector, then ``FixupSeeds`` (i_seed[0]=624).
  - ``MT_genrand``: classic MT19937 twist on the *1998* sgenrand layout
    (two 16-bit halves from the 69069 LCG), output scaled by 2^-32.
  - ``unif_rand``: ``fixup`` applied so 0 and 1 are never returned.
* ``R_unif_index`` with ``sample.kind = "Rejection"`` (R >= 3.6 default):
  16-bit chunks via ``rbits()``, mask to ``ceil(log2(n))`` bits,
  rejection loop while ``dv >= n``. ``"Rounding"`` is also provided.
* ``sample(n, k)`` uniform, without replacement (prob == NULL branch of
  ``do_sample``): swap-down partial shuffle, or direct draws when k < 2.

The Mersenne-Twister block twist is vectorised with NumPy; the
value-dependent rejection sampling is necessarily sequential and matches
R's uniform consumption exactly.
"""

from __future__ import annotations

import math

import numpy as np

_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER_MASK = np.uint32(0x80000000)
_LOWER_MASK = np.uint32(0x7FFFFFFF)
_TEMPERING_MASK_B = np.uint32(0x9D2C5680)
_TEMPERING_MASK_C = np.uint32(0xEFC60000)

_MT_SCALE = 2.3283064365386963e-10  # 1 / 2^32  (MT_genrand scaling)
_I2_32M1 = 2.328306437080797e-10    # 1 / (2^32 - 1) (fixup)

UINT32_MASK = 0xFFFFFFFF


def _lcg_next(s: int) -> int:
    """One step of R's seed LCG: seed = 69069 * seed + 1 (mod 2^32)."""
    return (69069 * s + 1) & UINT32_MASK


class RRng:
    """R's default RNG (Mersenne-Twister + Inversion), Rejection sampling."""

    def __init__(self, seed: int | None = None, sample_kind: str = "REJECTION"):
        if sample_kind not in ("REJECTION", "ROUNDING"):
            raise ValueError("sample_kind must be 'REJECTION' or 'ROUNDING'")
        self.sample_kind = sample_kind
        self.mt = np.zeros(_N, dtype=np.uint32)
        self.mti = _N + 1
        if seed is not None:
            self.set_seed(seed)

    # ------------------------------------------------------------------ seed
    def set_seed(self, seed: int) -> None:
        """Replicates do_setseed -> RNG_Init(MERSENNE_TWISTER, seed)."""
        s = seed & UINT32_MASK
        for _ in range(50):  # initial scrambling
            s = _lcg_next(s)
        i_seed = np.empty(1 + _N, dtype=np.uint32)
        for j in range(1 + _N):
            s = _lcg_next(s)
            i_seed[j] = s
        # FixupSeeds(MERSENNE_TWISTER, initial=1): I1 = 624
        i_seed[0] = np.uint32(_N)
        self.mti = _N
        self.mt = i_seed[1:].copy()
        self._index = _N  # force twist on first draw

    # --------------------------------------------------------------- MT core
    def _twist(self) -> None:
        # Vectorised MT twist. Block B (kk=227..622) reads mt[kk-227] AFTER
        # those entries were updated, so it must be split where the read
        # window crosses into block-B territory (kk = 454).
        mt = self.mt
        orig = mt.copy()
        one = np.uint32(1)

        # kk = 0..226: sources are all original
        y = (orig[: _N - _M] & _UPPER_MASK) | (orig[1 : _N - _M + 1] & _LOWER_MASK)
        mt[: _N - _M] = orig[_M :] ^ (y >> one) ^ np.where(y & one, _MATRIX_A, np.uint32(0))

        # kk = 227..453: mt[kk-227] was updated by block A
        y = (orig[_N - _M : 2 * (_N - _M)] & _UPPER_MASK) | (orig[_N - _M + 1 : 2 * (_N - _M) + 1] & _LOWER_MASK)
        mt[_N - _M : 2 * (_N - _M)] = mt[: _N - _M] ^ (y >> one) ^ np.where(y & one, _MATRIX_A, np.uint32(0))

        # kk = 454..622: mt[kk-227] was updated by the previous slice
        y = (orig[2 * (_N - _M) : _N - 1] & _UPPER_MASK) | (orig[2 * (_N - _M) + 1 : _N] & _LOWER_MASK)
        mt[2 * (_N - _M) : _N - 1] = mt[_N - _M : _M - 1] ^ (y >> one) ^ np.where(y & one, _MATRIX_A, np.uint32(0))

        # kk = 623: y mixes original mt[623] with already-updated mt[0]
        y = (orig[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
        mt[_N - 1] = mt[_M - 1] ^ (y >> one) ^ (_MATRIX_A if (y & one) else np.uint32(0))
        self._index = 0

    def _genrand_uint32(self) -> int:
        if self._index >= _N:
            self._twist()
        y = self.mt[self._index]
        self._index += 1
        y ^= y >> np.uint32(11)
        y ^= (y << np.uint32(7)) & _TEMPERING_MASK_B
        y ^= (y << np.uint32(15)) & _TEMPERING_MASK_C
        y ^= y >> np.uint32(18)
        return int(y)

    def unif_rand(self) -> float:
        """R's unif_rand() for MT: fixup(genrand * 2^-32)."""
        x = self._genrand_uint32() * _MT_SCALE
        # fixup(): ensure 0 and 1 are never returned
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if (1.0 - x) <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    # ----------------------------------------------------------- R_unif_index
    def _ru(self) -> float:
        u = 33554432.0
        return (math.floor(u * self.unif_rand()) + self.unif_rand()) / u

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def r_unif_index(self, dn: float) -> int:
        if self.sample_kind == "ROUNDING":
            cut = 2147483647.0  # INT_MAX (MT has 32-bit precision)
            u = self._ru() if dn > cut else self.unif_rand()
            return int(math.floor(dn * u))
        # REJECTION (R >= 3.6 default)
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        dv = self._rbits(bits)
        while dn <= dv:
            dv = self._rbits(bits)
        return int(dv)

    # ---------------------------------------------------------------- sample
    def sample_no_replace(self, n: int, k: int) -> np.ndarray:
        """sample(n, k) with equal probabilities, without replacement.

        Uniform-sampling branch of R's do_sample (random.c).
        Returns 1-based indices like R (dtype int64).
        """
        if k < 0 or n < 1 or k > n:
            raise ValueError("invalid sample size")
        out = np.empty(k, dtype=np.int64)
        if k < 2:
            for i in range(k):
                out[i] = int(self.r_unif_index(n)) + 1
        else:
            x = np.arange(n, dtype=np.int64)
            nn = n
            for i in range(k):
                j = int(self.r_unif_index(nn))
                out[i] = x[j] + 1
                x[j] = x[nn - 1]
                nn -= 1
        return out

    # convenience: draw m uniforms (like runif(m))
    def runif(self, m: int) -> np.ndarray:
        return np.fromiter((self.unif_rand() for _ in range(m)), dtype=np.float64, count=m)
