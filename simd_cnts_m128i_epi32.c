// Contains inlined functions 
//  __m128i X_mm_lzcnt_epi32(__m128i v)
//  __m128i X_mm_tzcnt_epi32(__m128i v)
//  __m128i X_mm_popcnt_epi32(__m128i v)
//
// Note: Depends on simd_cnts_m128i_epi16.c

#ifdef __AVX512VL__
  #define X_mm_lzcnt_epi32 _mm_lzcnt_epi32
  #define X_mm_tzcnt_epi32 _mm_tzcnt_epi32
#else
static inline __m128i X_mm_lzcnt_epi32(__m128i v) {
    // Based on https://stackoverflow.com/questions/58823140/count-leading-zero-bits-for-each-element-in-avx2-vector-emulate-mm256-lzcnt-ep
    // Tested exhaustively for all rounding modes.
    // prevent value from being rounded up to the next power of two
    v = _mm_andnot_si128(_mm_srli_epi32(v, 8), v); // keep 8 MSB
    v = _mm_castps_si128(_mm_cvtepi32_ps(v)); // convert signed integer to float ??
    v = _mm_srli_epi32(v, 23); // shift down the exponent
    v = _mm_subs_epu16(_mm_set1_epi32(158), v); // undo bias
    v = _mm_min_epi16(v, _mm_set1_epi32(32)); // clamp at 32
    return v;
}

#ifdef __BMI__
static inline __m128i X_mm_tzcnt_epi32(__m128i v) {
  uint32_t a[4];
  _mm_storeu_si64(a, v);
  a[0] = (a[0] ? __builtin_ctz(a[0]) : 32);
  a[1] = (a[1] ? __builtin_ctz(a[1]) : 32);
  a[2] = (a[2] ? __builtin_ctz(a[2]) : 32);
  a[3] = (a[3] ? __builtin_ctz(a[3]) : 32);
  return _mm_loadu_si64(a);  
}
#else
#ifdef __SSSE3__
static inline __m128i X_mm_tzcnt_epi32(__m128i v) {
  // Uses the 16 bit function.
  // (a,b) -> a+16 if b = 16
  //          b otherwise
  __m128i temp;
  __m128i res = X_mm_tzcnt_epi16(v);
  __m128i mask = _mm_slli_epi32(_mm_cmpeq_epi16(v, _mm_setzero_si128()), 16);
  temp = _mm_srli_epi32(_mm_and_si128(mask, res), 16);
  return _mm_add_epi32(temp, _mm_and_si128(res, _mm_set1_epi32(0xffff)));
}
#else
static inline __m128i X_mm_tzcnt_epi32(__m128i v) {
  __m128i temp, temp2;
  __m128i res;
  __m128i mask = _mm_cmpeq_epi32(v, _mm_setzero_si128());
  res = _mm_and_si128(mask, _mm_set1_epi32(1));
  mask = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0xffff)), _mm_setzero_si128());
  temp = _mm_and_si128(mask, _mm_srli_epi32(v, 16));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi32(res, _mm_and_si128(mask, _mm_set1_epi32(16)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0xff)), _mm_setzero_si128());
  temp = _mm_and_si128(mask, _mm_srli_epi32(v, 8));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi32(res, _mm_and_si128(mask, _mm_set1_epi32(8)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0xf)), _mm_setzero_si128());
  temp = _mm_and_si128(mask, _mm_srli_epi32(v, 4));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi32(res, _mm_and_si128(mask, _mm_set1_epi32(4)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0x3)), _mm_setzero_si128());
  temp = _mm_and_si128(mask, _mm_srli_epi32(v, 2));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi32(res, _mm_and_si128(mask, _mm_set1_epi32(2)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi32(_mm_and_si128(v, _mm_set1_epi32(0x1)), _mm_setzero_si128());
  res = _mm_add_epi32(res, _mm_and_si128(mask, _mm_set1_epi32(1)));
  return res;
}
#endif
#endif
#endif

#ifdef __AVX512_BITALG__
  #define X_mm_popcnt_epi32 _mm_popcnt_epi32
#else
#ifdef __POPCNT__
static inline __m128i X_mm_popcnt_epi32(__m128i v) {
  uint32_t a[4];
  _mm_storeu_si64(a, v);
  a[0] = __builtin_popcount(a[0]);
  a[1] = __builtin_popcount(a[1]);
  a[2] = __builtin_popcount(a[2]);
  a[3] = __builtin_popcount(a[3]);
  return _mm_loadu_si64(a);  
}
#else
static inline __m128i X_mm_popcnt_epi32(__m128i v) {
  v = _mm_sub_epi32(v, _mm_and_si128(_mm_srli_epi32(v, 1), _mm_set1_epi32(0x55555555)));
  v = _mm_add_epi32(_mm_and_si128(v, _mm_set1_epi32(0x33333333)), _mm_and_si128(_mm_srli_epi32(v, 2), _mm_set1_epi32(0x33333333)));
  v = _mm_and_si128(_mm_add_epi32(v, _mm_srli_epi32(v, 4)), _mm_set1_epi32(0x0f0f0f0f));
  v = _mm_and_si128(_mm_add_epi32(v, _mm_srli_epi32(v, 8)), _mm_set1_epi32(0x00ff00ff));
  v = _mm_add_epi32(_mm_and_si128(v, _mm_set1_epi32(0xffff)), _mm_srli_epi32(v, 16));
  return v;
}
#endif
#endif

