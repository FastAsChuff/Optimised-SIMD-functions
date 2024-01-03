// Contains inlined functions 
//  __m128i X_mm_lzcnt_epi16(__m128i v)
//  __m128i X_mm_tzcnt_epi16(__m128i v)
//  __m128i X_mm_popcnt_epi16(__m128i v)
//  __m128i Y_mm_popcnt_epi16(__m128i v)

#ifndef __SSSE3__
static inline __m128i X_mm_lzcnt_epi16(__m128i v) {
  __m128i temp, temp2;
  __m128i res;
  __m128i mask = _mm_cmpeq_epi16(v, _mm_set1_epi16(0));
  res = _mm_and_si128(mask, _mm_set1_epi16(1));
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0xff00)), _mm_set1_epi16(0));
  temp = _mm_and_si128(mask, v);
  temp2 = _mm_andnot_si128(mask, _mm_srli_epi16(v, 8));
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(8)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0xfff0)), _mm_set1_epi16(0));
  temp = _mm_and_si128(mask, v);
  temp2 = _mm_andnot_si128(mask, _mm_srli_epi16(v, 4));
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(4)));
  v = _mm_or_si128(temp, temp2);
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0xfffc)), _mm_set1_epi16(0));
  temp = _mm_and_si128(mask, v);
  temp2 = _mm_andnot_si128(mask, _mm_srli_epi16(v, 2));
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(2)));
  v = _mm_or_si128(temp, temp2);
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0xfffe)), _mm_set1_epi16(0));
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(1)));
  return res;
}

static inline __m128i X_mm_tzcnt_epi16(__m128i v) {
  __m128i temp, temp2;
  __m128i res;
  __m128i mask = _mm_cmpeq_epi16(v, _mm_set1_epi16(0));
  res = _mm_and_si128(mask, _mm_set1_epi16(1));
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0xff)), _mm_set1_epi16(0));
  temp = _mm_and_si128(mask, _mm_srli_epi16(v, 8));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(8)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0xf)), _mm_set1_epi16(0));
  temp = _mm_and_si128(mask, _mm_srli_epi16(v, 4));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(4)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0x3)), _mm_set1_epi16(0));
  temp = _mm_and_si128(mask, _mm_srli_epi16(v, 2));
  temp2 = _mm_andnot_si128(mask, v);
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(2)));
  v = _mm_or_si128(temp, temp2); 
  mask = _mm_cmpeq_epi16(_mm_and_si128(v, _mm_set1_epi16(0x1)), _mm_set1_epi16(0));
  res = _mm_add_epi16(res, _mm_and_si128(mask, _mm_set1_epi16(1)));
  return res;
}

#else
#ifdef __AVX512VL__
  #define X_mm_lzcnt_epi16 _mm_lzcnt_epi16
  #define X_mm_tzcnt_epi16 _mm_tzcnt_epi16
#else
// From https://www.reddit.com/r/simd/comments/b3k1oa/looking_for_sseavx_bitscan_discussions/
static inline __m128i X_mm_lzcnt_epi16(__m128i v) {
	__m128i byte_lmask = _mm_set1_epi8(0xf);
	
	// find lzcnt for low nibble of each byte
	__m128i low = _mm_and_si128(v, byte_lmask);
	low = _mm_shuffle_epi8(_mm_set_epi32(0x04040404,0x04040404,0x05050505,0x06060710), low);
	// do the same for the high nibble
	__m128i high = _mm_and_si128(_mm_srli_epi16(v, 4), byte_lmask);
	high = _mm_shuffle_epi8(_mm_set_epi32(0,0,0x01010101,0x02020310), high);
	
	// find lzcnt for each byte
	__m128i lz_byte = _mm_min_epu8(low, high);
	
	// now combine to find lzcnt for each word
	low = _mm_add_epi8(lz_byte, _mm_set1_epi16(8));
	high = _mm_srli_epi16(lz_byte, 8);
	return _mm_min_epu8(low, high);
}
static inline __m128i X_mm_tzcnt_epi16(__m128i v) {
	__m128i byte_lmask = _mm_set1_epi8(0xf);
	
	// find tzcnt for low/high nibbles of each byte
	__m128i low = _mm_and_si128(v, byte_lmask);
	low = _mm_shuffle_epi8(_mm_set_epi32(0x00010002,0x00010003,0x00010002,0x00010010), low);
	__m128i high = _mm_and_si128(_mm_srli_epi16(v, 4), byte_lmask);
	high = _mm_shuffle_epi8(_mm_set_epi32(0x04050406,0x04050407,0x04050406,0x04050410), high);
	
	// tzcnt for each byte
	__m128i tz_byte = _mm_min_epu8(low, high);
	
	// combine for tzcnt for each word
	low = tz_byte;
	high = _mm_srli_epi16(_mm_add_epi8(tz_byte, _mm_set1_epi8(8)), 8);
	return _mm_min_epu8(low, high);
}
#endif
#endif

#ifdef __AVX512_BITALG__
  #define X_mm_popcnt_epi16 _mm_popcnt_epi16
  #define Y_mm_popcnt_epi16 _mm_popcnt_epi16
#else
static inline __m128i X_mm_popcnt_epi16(__m128i v) {
// Adapted from scalar 32 bit https://stackoverflow.com/questions/109023/count-the-number-of-set-bits-in-a-32-bit-integer
  v = _mm_sub_epi16(v, _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi16(0x5555)));
  v = _mm_add_epi16(_mm_and_si128(v, _mm_set1_epi16(0x3333)), _mm_and_si128(_mm_srli_epi16(v, 2), _mm_set1_epi16(0x3333)));
  v = _mm_and_si128(_mm_add_epi16(v, _mm_srli_epi16(v, 4)), _mm_set1_epi16(0x0f0f));
  return _mm_srli_epi16(_mm_mullo_epi16(v, _mm_set1_epi16(0x101)), 8);
}
static inline __m128i Y_mm_popcnt_epi16(__m128i v) {
  // Use if all popcounts are < 16.
  v = _mm_sub_epi16(v, _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi16(0x5555)));
  v = _mm_add_epi16(_mm_and_si128(v, _mm_set1_epi16(0x3333)), _mm_and_si128(_mm_srli_epi16(v, 2), _mm_set1_epi16(0x3333)));
  return _mm_srli_epi16(_mm_mullo_epi16(v, _mm_set1_epi16(0x1111)), 12);
}
#endif

