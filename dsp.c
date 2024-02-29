#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "dsp.h"
#include "tables.h"

#include <arm_neon.h>

static float dctlookupTranspose[8][8] =
{
  {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, },
  {0.980785f, 0.831470f, 0.555570f, 0.195090f, -0.195090f, -0.555570f, -0.831470f, -0.980785f, },
  {0.923880f, 0.382683f, -0.382683f, -0.923880f, -0.923880f, -0.382683f, 0.382683f, 0.923880f, },
  {0.831470f, -0.195090f, -0.980785f, -0.555570f, 0.555570f, 0.980785f, 0.195090f, -0.831470f, },
  {0.707107f, -0.707107f, -0.707107f, 0.707107f, 0.707107f, -0.707107f, -0.707107f, 0.707107f, },
  {0.555570f, -0.980785f, 0.195090f, 0.831470f, -0.831470f, -0.195090f, 0.980785f, -0.555570f, },
  {0.382683f, -0.923880f, 0.923880f, -0.382683f, -0.382683f, 0.923880f, -0.923880f, 0.382683f, },
  {0.195090f, -0.555570f, 0.831470f, -0.980785f, 0.980785f, -0.831470f, 0.555570f, -0.195090f, },
};

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

static void dct_1d(float *in_data, float *out_data) {
  // Set output vectors to zero
  float32x4_t sumLow = vdupq_n_f32(0.0f);
  float32x4_t sumHigh = vdupq_n_f32(0.0f);

  for (int i = 0; i < 8; ++i) {
    // Load current input data element to all four lanes of a NEON register
    float32x4_t inputValue = vld1q_dup_f32(in_data + i);

    // Load the first half of the current row of the lookup table
    float32x4_t dctLookupLow = vld1q_f32(&dctlookup[i][0]);
    // Load the second half of the current row of the lookup table
    float32x4_t dctLookupHigh = vld1q_f32(&dctlookup[i][4]);

    // Sum
    sumLow = vmlaq_f32(sumLow, dctLookupLow, inputValue);
    sumHigh = vmlaq_f32(sumHigh, dctLookupHigh, inputValue);
  }

  // Store the results back to the output array
  vst1q_f32(out_data, sumLow);
  vst1q_f32(out_data + 4, sumHigh);
}

static void idct_1d(float *in_data, float *out_data) {
    // Set output vectors to zero
    float32x4_t sumLow = vdupq_n_f32(0.0f);
    float32x4_t sumHigh = vdupq_n_f32(0.0f);

    for (int i = 0; i < 8; ++i) {
        // Load the first half of the current row of the lookup transposed table
        float32x4_t dctLookupLow = vld1q_f32(&dctlookupTranspose[i][0]);
        // Load the second half of the current row of the lookup transposed table
        float32x4_t dctLookupHigh = vld1q_f32(&dctlookupTranspose[i][4]);

        // Sum
        sumLow = vmlaq_n_f32(sumLow, dctLookupLow, in_data[i]);
        sumHigh = vmlaq_n_f32(sumHigh, dctLookupHigh, in_data[i]);
    }

    // Store the results back to the output array
    vst1q_f32(&out_data[0], sumLow);
    vst1q_f32(&out_data[4], sumHigh);
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result) {
    // Initialize an empty vector
    uint16x8_t sum = vdupq_n_u16(0);

    // Row 0
    uint8x8_t block1_0 = vld1_u8(&block1[0 * stride]);
    uint8x8_t block2_0 = vld1_u8(&block2[0 * stride]);
    uint8x8_t abdiff_0 = vabd_u8(block1_0, block2_0);
    sum = vaddw_u8(sum, abdiff_0);

    // Row 1
    uint8x8_t block1_1 = vld1_u8(&block1[1 * stride]);
    uint8x8_t block2_1 = vld1_u8(&block2[1 * stride]);
    uint8x8_t abdiff_1 = vabd_u8(block1_1, block2_1);
    sum = vaddw_u8(sum, abdiff_1);

    // Rows 2
    uint8x8_t block1_2 = vld1_u8(&block1[2 * stride]);
    uint8x8_t block2_2 = vld1_u8(&block2[2 * stride]);
    uint8x8_t abdiff_2 = vabd_u8(block1_2, block2_2);
    sum = vaddw_u8(sum, abdiff_2);

    // Row 3
    uint8x8_t block1_3 = vld1_u8(&block1[3 * stride]);
    uint8x8_t block2_3 = vld1_u8(&block2[3 * stride]);
    uint8x8_t abdiff_3 = vabd_u8(block1_3, block2_3);
    sum = vaddw_u8(sum, abdiff_3);

    // Row 4
    uint8x8_t block1_4 = vld1_u8(&block1[4 * stride]);
    uint8x8_t block2_4 = vld1_u8(&block2[4 * stride]);
    uint8x8_t abdiff_4 = vabd_u8(block1_4, block2_4);
    sum = vaddw_u8(sum, abdiff_4);

    // Row 5
    uint8x8_t block1_5 = vld1_u8(&block1[5 * stride]);
    uint8x8_t block2_5 = vld1_u8(&block2[5 * stride]);
    uint8x8_t abdiff_5 = vabd_u8(block1_5, block2_5);
    sum = vaddw_u8(sum, abdiff_5);

    // Row 6
    uint8x8_t block1_6 = vld1_u8(&block1[6 * stride]);
    uint8x8_t block2_6 = vld1_u8(&block2[6 * stride]);
    uint8x8_t abdiff_6 = vabd_u8(block1_6, block2_6);
    sum = vaddw_u8(sum, abdiff_6);

    // Row 7
    uint8x8_t block1_7 = vld1_u8(&block1[7 * stride]);
    uint8x8_t block2_7 = vld1_u8(&block2[7 * stride]);
    uint8x8_t abdiff_7 = vabd_u8(block1_7, block2_7);
    sum = vaddw_u8(sum, abdiff_7);

    // Add the 8 values in the sum vector together to get the total sum
    *result = vaddvq_u16(sum);
}