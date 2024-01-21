/* May be slightly faster than the 'Hacker's Delight' bithack pdep32 function on hardware that does not support AVX2/BMI2 extentions
  in the case where it does not make sense to pre-compute a mask object. SIMD code is derived from the following. SSE2 is required as a minimum.
  Author: Simon Goater Jan 2024  
  
    uint32_t pdep32_naive_branchless(uint32_t src, uint32_t mask) {
      uint32_t dest = 0;
      uint32_t bit = 1;
      uint32_t kbit = 1;
      _Bool temp;
      while (bit) {
        temp = ((mask & bit) != 0);
        dest |= bit*((src & kbit) != 0)*temp;
        kbit <<= temp;
        bit <<= 1;
      }
      return dest;
    }

*/
uint32_t pdep32(uint32_t src, uint32_t mask) {
  uint32_t dest[4] = {0};
  uint8_t maskpops[4];
  uint32_t kbits[4];
  uint32_t mask2;
  mask2 = mask - ((mask >> 1) & 0x55555555);
  mask2 = (mask2 & 0x33333333) + ((mask2 >> 2) & 0x33333333);
  mask2 = (mask2 + (mask2 >> 4)) & 0x0F0F0F0F; 
  maskpops[1] = (mask2 >> 16) & 0xf;
  maskpops[2] = (mask2 >> 8) & 0xf;
  maskpops[3] = mask2 & 0xf;  
  maskpops[2] += maskpops[3];
  maskpops[1] += maskpops[2];
  kbits[3] = 1;
  kbits[2] = 1 << maskpops[3];
  kbits[1] = 1 << maskpops[2];
  kbits[0] = 1 << maskpops[1];
  __m128i dest128 = _mm_setzero_si128();
  __m128i kbits128 = _mm_loadu_si128((__m128i*)kbits);
  __m128i bits128 = _mm_setr_epi32(1 << 24, 1 << 16, 1 << 8, 1);
  __m128i mask128 = _mm_set1_epi32(mask);
  __m128i src128 = _mm_set1_epi32(src);
  __m128i temp128;
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
    kbits128 = _mm_add_epi32(kbits128, _mm_andnot_si128(temp128, kbits128));
    bits128 = _mm_add_epi32(bits128, bits128);
    temp128 = _mm_cmpeq_epi32(_mm_and_si128(bits128, mask128), _mm_setzero_si128()); 
    dest128 = _mm_or_si128(dest128, _mm_andnot_si128(temp128, _mm_andnot_si128(_mm_cmpeq_epi32(_mm_and_si128(src128, kbits128), _mm_setzero_si128()), bits128)));
  _mm_storeu_si128((__m128i*)dest, dest128);
  return dest[0] | dest[1] | dest[2] | dest[3];
}

