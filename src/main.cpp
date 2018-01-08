#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <smmintrin.h>

uint32_t hashUint32(uint32_t input)
{
  input = ~input + (input << 15);
  input = input ^ (input >> 12);
  input = input + (input << 2);
  input = input ^ (input >> 4);
  input = input * 2057;
  input = input ^ (input >> 16);

  return input;
}

float bitsToFloat(uint32_t input)
{
  float output;
  memcpy(&output, &input, sizeof(uint32_t));

  return output;
}

float uintToFloat(uint32_t input)
{
  uint32_t output = (0x7F << 23) | (input >> 9);

  return bitsToFloat(output) - 1.f;
}

uint32_t pseudoRandomUint(uint32_t input, uint32_t scramble = 0U)
{
  input ^= scramble;
  input ^= input >> 17;
  input ^= input >> 10;
  input *= 0xb36534e5;
  input ^= input >> 12;
  input ^= input >> 21;
  input *= 0x93fc4795;
  input ^= 0xdf6e307f;
  input ^= input >> 17;
  input *= 1 | scramble >> 18;

  return input;
}

float pseudoRandomFloat(uint32_t input, uint32_t scramble = 0U)
{
  return uintToFloat(pseudoRandomUint(input, scramble));
}

struct vec3f
{
  float x, y, z;
};

float lengthSqr(const vec3f &v)
{
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

float length(const vec3f &v)
{
  return sqrt(lengthSqr(v));
}

vec3f normalise(const vec3f &v)
{
  float linv = 1.f / length(v);

  vec3f out;
  out.x = v.x * linv;
  out.y = v.y * linv;
  out.z = v.z * linv;

  return out;
}

float dot(const vec3f &a, const vec3f &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3f cross(const vec3f &a, const vec3f &b)
{
  vec3f out;
  out.x = a.y * b.z - a.z * b.y;
  out.y = a.z * b.x - a.x * b.z;
  out.z = a.x * b.y - a.y * b.x;

  return out;
}

void basisLinear(const vec3f &nIn, const vec3f &vIn, vec3f *sOut, vec3f *tOut)
{
  *tOut = cross(nIn, vIn);
  *tOut = normalise(*tOut);
  *sOut = cross(*tOut, nIn);
}

void basisVector(const vec3f &nIn, const vec3f &vIn, vec3f *sOut, vec3f *tOut)
{
  // create a yzx shuffle mask
  static const int mask = _MM_SHUFFLE(3, 0, 2, 1);

  // assign constant variables
  static __m128 half = _mm_set1_ps(0.5f);
  static __m128 three = _mm_set1_ps(3.f);
  static __m128 zero = _mm_set1_ps(0.f);

  // load input into aligned memory
  __m128 n = _mm_setr_ps(nIn.x, nIn.y, nIn.z, 0.f);
  __m128 v = _mm_setr_ps(vIn.x, vIn.y, vIn.z, 0.f);

  // shuffle n and v for cross product
  __m128 nYZX = _mm_shuffle_ps(n, n, mask);
  __m128 vYZX = _mm_shuffle_ps(v, v, mask);

  // calculate cross product and shuffle back
  __m128 uZXY = _mm_sub_ps(_mm_mul_ps(n, vYZX), _mm_mul_ps(nYZX, v));
  __m128 u = _mm_shuffle_ps(uZXY, uZXY, mask);

  // normalise result using Newton's method and store as t
  __m128 lsqr = _mm_dp_ps(u, u, 0xFF);
  __m128 sqrt = _mm_rsqrt_ps(lsqr);
  __m128 mult = _mm_mul_ps(_mm_mul_ps(lsqr, sqrt), sqrt);
  __m128 t = _mm_mul_ps(u, _mm_mul_ps(_mm_mul_ps(half, sqrt), _mm_sub_ps(three, mult)));

  // prevent nan by checking if lsqr is greater than zero
  __m128 cmp = _mm_cmpgt_ps(lsqr, zero);
  t = _mm_or_ps(_mm_and_ps(cmp, t), _mm_andnot_ps(cmp, u));

  // shuffle t for cross product
  __m128 tYZX = _mm_shuffle_ps(t, t, mask);

  // calculate cross product and shuffle back
  __m128 sZXY = _mm_sub_ps(_mm_mul_ps(t, nYZX), _mm_mul_ps(tYZX, n));
  __m128 s = _mm_shuffle_ps(sZXY, sZXY, mask);

  // extract into output s vector
  sOut->x = _mm_cvtss_f32(_mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 0)));
  sOut->y = _mm_cvtss_f32(_mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 1)));
  sOut->z = _mm_cvtss_f32(_mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 2)));

  // extract into output t vector
  tOut->x = _mm_cvtss_f32(_mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 0)));
  tOut->y = _mm_cvtss_f32(_mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 1)));
  tOut->z = _mm_cvtss_f32(_mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 0, 0, 2)));
}

int main(int argc, char const *argv[])
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  uint32_t index = 0;
  uint32_t seed = hashUint32(0);

  vec3f n;
  n.x = pseudoRandomFloat(index++, seed) - 0.5f;
  n.y = pseudoRandomFloat(index++, seed) - 0.5f;
  n.z = pseudoRandomFloat(index++, seed) - 0.5f;

  n = normalise(n);

  float scaler = 23.f;

  vec3f tangent;
  tangent.x = (pseudoRandomFloat(index++, seed) - 0.5f) * scaler;
  tangent.y = (pseudoRandomFloat(index++, seed) - 0.5f) * scaler;
  tangent.z = (pseudoRandomFloat(index++, seed) - 0.5f) * scaler;

  vec3f s, t;

  printf("Linear Basis Calc:\n\n");

  basisLinear(n, tangent, &s, &t);

  printf("n: %f %f %f l: %f\n",   n.x, n.y, n.z, length(n));
  printf("s: %f %f %f l: %f\n",   s.x, s.y, s.z, length(s));
  printf("t: %f %f %f l: %f\n\n", t.x, t.y, t.z, length(t));

  printf("tangent: %f %f %f l: %f\n\n", tangent.x, tangent.y, tangent.z, length(tangent));

  printf("cos(st): %f cos(sn): %f cos(nt): %f\n\n", dot(s, t), dot(s, n), dot(n, t));

  printf("Vector Basis Calc:\n\n");

  basisVector(n, tangent, &s, &t);

  printf("n: %f %f %f l: %f\n",   n.x, n.y, n.z, length(n));
  printf("s: %f %f %f l: %f\n",   s.x, s.y, s.z, length(s));
  printf("t: %f %f %f l: %f\n\n", t.x, t.y, t.z, length(t));

  printf("tangent: %f %f %f l: %f\n\n", tangent.x, tangent.y, tangent.z, length(tangent));

  printf("cos(st): %f cos(sn): %f cos(nt): %f\n", dot(s, t), dot(s, n), dot(n, t));

  return 0;
}